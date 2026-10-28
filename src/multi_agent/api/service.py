"""任务编排：内存任务表 + 后台线程执行 + 审批恢复 + 审计写入。

本模块不引入 FastAPI 依赖，方便单独复用与单元测试。
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from multi_agent.api.schemas import ApprovalView, TaskStatus, TaskView
from multi_agent.config import Settings
from multi_agent.graph import RunResult, build_graph, resume_task, run_task_full
from multi_agent.registry import AgentRegistry

logger = logging.getLogger("multi_agent.api")

TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED}
)


class TaskNotFound(Exception):
    """任务不存在（或没有挂起的审批）。"""


class TaskConflict(Exception):
    """线程 ID 重复，或当前状态不允许该操作。"""


class TaskUnavailable(Exception):
    """服务正在关闭，暂时不能接收任务或审批。"""


@dataclass(frozen=True)
class StartOutcome:
    """一次任务启动的结果：图实例要留到审批恢复时继续用。"""

    result: RunResult
    app: Any


class Executor(Protocol):
    def start(
        self,
        task: str,
        thread_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> StartOutcome: ...

    def resume(
        self,
        thread_id: str,
        decision: dict[str, Any],
        app: Any,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunResult: ...


class LangGraphExecutor:
    """把现有的图入口包成 HTTP 层可调用的同步执行器。

    每个任务单独 build_graph：每个工作线程拿到自己的 SQLite 连接，
    避免所有任务共用一条连接串行化。
    """

    def __init__(
        self,
        settings: Settings,
        registry: AgentRegistry | None = None,
        app_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._app_factory: Callable[[], Any] = app_factory or (
            lambda: build_graph(settings=self._settings, registry=self._registry)
        )

    def start(
        self,
        task: str,
        thread_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> StartOutcome:
        app = self._app_factory()
        result = run_task_full(
            task,
            settings=self._settings,
            registry=self._registry,
            thread_id=thread_id,
            app=app,
            should_cancel=should_cancel,
        )
        return StartOutcome(result=result, app=app)

    def resume(
        self,
        thread_id: str,
        decision: dict[str, Any],
        app: Any,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunResult:
        return resume_task(
            thread_id,
            decision,
            settings=self._settings,
            registry=self._registry,
            app=app,
            should_cancel=should_cancel,
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _TaskRecord:
    thread_id: str
    task: str
    status: TaskStatus = TaskStatus.QUEUED
    answer: str = ""
    turns: int = 0
    stop_reason: str = ""
    pending: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    app: Any | None = field(default=None, repr=False)
    deadline: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class TaskManager:
    """进程内任务表。服务重启后任务状态丢失，但检查点仍在。"""

    def __init__(
        self,
        executor: Executor,
        *,
        max_workers: int = 4,
        task_timeout: float = 1800.0,
        retention: float = 86_400.0,
        max_records: int = 1000,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not math.isfinite(task_timeout) or task_timeout < 0:
            raise ValueError("task_timeout must be a finite number >= 0")
        if not math.isfinite(retention) or retention < 0:
            raise ValueError("retention must be a finite number >= 0")
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        self._executor = executor
        self._lock = threading.Lock()
        self._records: dict[str, _TaskRecord] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="multi-agent-api"
        )
        self._closed = False
        self._task_timeout = float(task_timeout)
        self._retention = float(retention)
        self._max_records = int(max_records)

    # ---- 查询 -------------------------------------------------------------

    def get(self, thread_id: str) -> TaskView:
        with self._lock:
            self._sweep_locked()
            record = self._records.get(thread_id)
            if record is None:
                raise TaskNotFound("task not found")
            return self._view(record)

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskView]:
        with self._lock:
            self._sweep_locked()
            records = sorted(self._records.values(), key=lambda item: item.created_at)
            return [
                self._view(record)
                for record in records
                if status is None or record.status == status
            ]

    def list_approvals(self) -> list[ApprovalView]:
        with self._lock:
            self._sweep_locked()
            records = sorted(
                (
                    record
                    for record in self._records.values()
                    if record.status == TaskStatus.AWAITING_APPROVAL and record.pending
                ),
                key=lambda item: item.updated_at,
            )
            return [self._approval_view(record) for record in records]

    def get_approval(self, thread_id: str) -> ApprovalView:
        with self._lock:
            self._sweep_locked()
            record = self._records.get(thread_id)
            if record is None:
                raise TaskNotFound("task not found")
            if not record.pending:
                raise TaskNotFound("no pending approval")
            return self._approval_view(record)

    # ---- 提交与审批 -------------------------------------------------------

    def submit(self, task: str, thread_id: str | None = None) -> TaskView:
        with self._lock:
            if self._closed:
                raise TaskUnavailable("task manager is shutting down")
            self._sweep_locked()
            tid = thread_id or f"api-{uuid.uuid4().hex[:8]}"
            if tid in self._records:
                raise TaskConflict("task already exists")
            if len(self._records) >= self._max_records and not any(
                record.status in TERMINAL_STATUSES for record in self._records.values()
            ):
                raise TaskUnavailable(
                    f"task table is full ({self._max_records} active records); "
                    "wait for tasks to finish or raise --max-records"
                )
            record = _TaskRecord(thread_id=tid, task=task, deadline=self._deadline())
            self._records[tid] = record
            self._sweep_locked()
            view = self._view(record)
        try:
            self._pool.submit(self._run, tid, task)
        except RuntimeError as exc:
            with self._lock:
                self._mark_failed(record, "task manager is shutting down")
            raise TaskUnavailable("task manager is shutting down") from exc
        return view

    def approve(self, thread_id: str, decision: dict[str, Any]) -> TaskView:
        with self._lock:
            if self._closed:
                raise TaskUnavailable("task manager is shutting down")
            self._sweep_locked()
            record = self._records.get(thread_id)
            if record is None:
                raise TaskNotFound("task not found")
            if record.status != TaskStatus.AWAITING_APPROVAL or not record.pending:
                raise TaskConflict("task is not awaiting approval")
            app = record.app
            record.status = TaskStatus.RUNNING
            record.pending = None
            record.deadline = self._deadline()
            record.updated_at = _now()
            view = self._view(record)
            if app is None:
                self._mark_failed(record, "task app is unavailable")
                raise TaskConflict("task app is unavailable")
            try:
                # 持锁提交：避免「已批准 → 提交前被取消」的窗口执行到已取消任务
                self._pool.submit(self._resume, thread_id, app, dict(decision))
            except RuntimeError as exc:
                self._mark_failed(record, "task manager is shutting down")
                raise TaskUnavailable("task manager is shutting down") from exc
        return view

    def cancel(self, thread_id: str) -> TaskView:
        with self._lock:
            if self._closed:
                raise TaskUnavailable("task manager is shutting down")
            self._sweep_locked()
            record = self._records.get(thread_id)
            if record is None:
                raise TaskNotFound("task not found")
            if record.status is TaskStatus.CANCELED:
                return self._view(record)
            if record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                raise TaskConflict("task already finished")
            self._mark_canceled(record, "task canceled by client")
            return self._view(record)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _sweep_locked(self) -> None:
        """按超时、审批过期、保留期与容量上限收敛任务表；调用方需持锁。"""
        monotonic_now = time.monotonic()
        wall_now = datetime.now(timezone.utc)

        for record in list(self._records.values()):
            if (
                record.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
                and record.deadline is not None
                and monotonic_now >= record.deadline
            ):
                logger.warning("task %s timed out", record.thread_id)
                self._mark_canceled(record, f"task timed out after {self._task_timeout:g}s")
            elif (
                record.status is TaskStatus.AWAITING_APPROVAL
                and self._retention > 0
                and (wall_now - record.updated_at).total_seconds() >= self._retention
            ):
                logger.warning("approval expired for %s", record.thread_id)
                self._mark_canceled(record, "approval expired")

        if self._retention > 0:
            for thread_id, record in list(self._records.items()):
                if (
                    record.status in TERMINAL_STATUSES
                    and (wall_now - record.updated_at).total_seconds() >= self._retention
                ):
                    del self._records[thread_id]

        while len(self._records) > self._max_records:
            finished = [
                (record.updated_at, thread_id)
                for thread_id, record in self._records.items()
                if record.status in TERMINAL_STATUSES
            ]
            if not finished:
                break
            del self._records[min(finished)[1]]

    # ---- 后台执行 ---------------------------------------------------------

    def _run(self, thread_id: str, task: str) -> None:
        with self._lock:
            record = self._records.get(thread_id)
            if record is None or record.status is TaskStatus.CANCELED:
                return
            record.status = TaskStatus.RUNNING
            record.updated_at = _now()
            cancel_event = record.cancel_event
        try:
            outcome = self._executor.start(
                task, thread_id, should_cancel=cancel_event.is_set
            )
            self._apply(thread_id, outcome.result, outcome.app)
        except Exception as exc:
            logger.exception("task %s failed", thread_id)
            with self._lock:
                current = self._records.get(thread_id)
                if current is not None:
                    self._mark_failed(current, _describe(exc))

    def _resume(self, thread_id: str, app: Any, decision: dict[str, Any]) -> None:
        with self._lock:
            current = self._records.get(thread_id)
            if current is None or current.status is not TaskStatus.RUNNING:
                return  # 已取消 / 超时的任务不得再执行已批准的审批
            cancel_event = current.cancel_event
        try:
            result = self._executor.resume(
                thread_id, decision, app, should_cancel=cancel_event.is_set
            )
            self._apply(thread_id, result, app)
        except Exception as exc:
            logger.exception("task %s failed", thread_id)
            with self._lock:
                current = self._records.get(thread_id)
                if current is not None:
                    self._mark_failed(current, _describe(exc))

    def _apply(self, thread_id: str, result: RunResult, app: Any) -> None:
        with self._lock:
            record = self._records.get(thread_id)
            if record is None or record.status is TaskStatus.CANCELED:
                return
            record.answer = result.answer
            record.turns = result.turns
            record.stop_reason = result.stop_reason
            record.usage = dict(result.usage)
            record.error = None
            record.deadline = None
            record.updated_at = _now()
            if result.pending:
                record.status = TaskStatus.AWAITING_APPROVAL
                record.pending = dict(result.pending)
                record.app = app
            elif result.stop_reason == "awaiting_approval":
                self._mark_failed(record, "awaiting_approval without pending payload")
            elif result.stop_reason == "canceled":
                self._mark_canceled(record, "task canceled")
            elif result.stop_reason == "error":
                self._mark_failed(record, result.answer or "task failed")
            else:
                record.status = TaskStatus.COMPLETED
                record.pending = None
                record.app = None

    # ---- 内部工具（调用方需持锁） -----------------------------------------

    def _mark_failed(self, record: _TaskRecord, error: str) -> None:
        if record.status is TaskStatus.CANCELED:
            return  # 软取消后迟到的失败不应复活任务
        record.status = TaskStatus.FAILED
        record.error = error
        record.pending = None
        record.app = None
        record.deadline = None
        record.updated_at = _now()

    def _mark_canceled(self, record: _TaskRecord, error: str) -> None:
        record.status = TaskStatus.CANCELED
        record.error = error
        record.pending = None
        record.app = None
        record.deadline = None
        record.cancel_event.set()  # 让正在跑的图在下一个节点边界停下
        record.updated_at = _now()

    def _deadline(self) -> float | None:
        return time.monotonic() + self._task_timeout if self._task_timeout > 0 else None

    @staticmethod
    def _view(record: _TaskRecord) -> TaskView:
        return TaskView(
            thread_id=record.thread_id,
            task=record.task,
            status=record.status,
            answer=record.answer,
            turns=record.turns,
            stop_reason=record.stop_reason,
            pending=dict(record.pending) if record.pending else None,
            usage=dict(record.usage),
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _approval_view(record: _TaskRecord) -> ApprovalView:
        return ApprovalView(
            thread_id=record.thread_id,
            task=record.task,
            pending=dict(record.pending or {}),
            updated_at=record.updated_at,
        )


def _describe(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
