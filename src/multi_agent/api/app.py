"""FastAPI 应用工厂：把 TaskManager 暴露为 HTTP 接口。"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from multi_agent.api.schemas import (
    ApprovalDecision,
    ApprovalView,
    TaskStatus,
    TaskSubmit,
    TaskView,
)
from multi_agent.api.service import (
    Executor,
    LangGraphExecutor,
    TaskConflict,
    TaskManager,
    TaskNotFound,
    TaskUnavailable,
)
from multi_agent.config import Settings
from multi_agent.registry import AgentRegistry


def _package_version() -> str:
    """版本号只来自安装元数据（pyproject.toml），避免与 OpenAPI 双源。"""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8 不存在
        return "0.0.0"
    try:
        return version("multi-agent")
    except PackageNotFoundError:  # pragma: no cover - 未安装运行
        return "0.0.0"


_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


class _RateLimiter:
    """单进程固定窗口限流：每个客户端 IP 每分钟的请求数上限。"""

    def __init__(self, per_minute: int) -> None:
        self._limit = per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """返回（是否放行, 超限时的 Retry-After 秒数）。"""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False, max(1, int(self._window - (now - hits[0])) + 1)
            hits.append(now)
            return True, 0


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return ""
    return header[7:].strip()


def create_app(
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    executor: Executor | None = None,
    max_workers: int = 4,
    task_timeout: float = 1800.0,
    retention: float = 86_400.0,
    max_records: int = 1000,
    task_db: str | None = None,
    api_token: str | None = None,
    rate_limit: int = 0,
) -> FastAPI:
    """构建应用；settings 为空时读环境变量（缺 key 会抛 RuntimeError）。

    `api_token` 非空时所有非公开路径要求 `Authorization: Bearer <token>`；
    `rate_limit` 是每客户端 IP 每分钟的请求上限（0 关闭）。
    """
    if rate_limit < 0:
        raise ValueError("rate_limit must be >= 0")
    token = (api_token or "").strip()
    limiter = _RateLimiter(rate_limit) if rate_limit > 0 else None
    settings = settings or Settings.from_env()
    executor = executor or LangGraphExecutor(settings, registry)
    manager = TaskManager(
        executor,
        max_workers=max_workers,
        task_timeout=task_timeout,
        retention=retention,
        max_records=max_records,
        task_db=task_db,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="multi-agent API", version=_package_version(), lifespan=lifespan)
    app.state.manager = manager

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.url.path not in _PUBLIC_PATHS:
            if limiter is not None:
                allowed, retry_after = limiter.allow(
                    request.client.host if request.client else "unknown"
                )
                if not allowed:
                    return JSONResponse(
                        status_code=http_status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "too many requests"},
                        headers={"Retry-After": str(retry_after)},
                    )
            if token:
                provided = _bearer_token(request)
                if not provided or not secrets.compare_digest(provided, token):
                    return JSONResponse(
                        status_code=http_status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "unauthorized"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(TaskNotFound)
    def _not_found(_request: Request, exc: TaskNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=http_status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
        )

    @app.exception_handler(TaskConflict)
    def _conflict(_request: Request, exc: TaskConflict) -> JSONResponse:
        return JSONResponse(
            status_code=http_status.HTTP_409_CONFLICT, content={"detail": str(exc)}
        )

    @app.exception_handler(TaskUnavailable)
    def _unavailable(_request: Request, exc: TaskUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.post("/tasks", response_model=TaskView, status_code=http_status.HTTP_202_ACCEPTED)
    def submit_task(body: TaskSubmit) -> TaskView:
        return manager.submit(body.task, body.thread_id)

    @app.get("/tasks", response_model=list[TaskView])
    def list_tasks(status: TaskStatus | None = None) -> list[TaskView]:
        return manager.list_tasks(status)

    @app.get("/tasks/{thread_id}", response_model=TaskView)
    def get_task(thread_id: str) -> TaskView:
        return manager.get(thread_id)

    @app.delete(
        "/tasks/{thread_id}",
        response_model=TaskView,
        status_code=http_status.HTTP_202_ACCEPTED,
    )
    def cancel_task(thread_id: str) -> TaskView:
        return manager.cancel(thread_id)

    @app.get("/approvals", response_model=list[ApprovalView])
    def list_approvals() -> list[ApprovalView]:
        return manager.list_approvals()

    @app.get("/approvals/{thread_id}", response_model=ApprovalView)
    def get_approval(thread_id: str) -> ApprovalView:
        return manager.get_approval(thread_id)

    @app.post(
        "/approvals/{thread_id}",
        response_model=TaskView,
        status_code=http_status.HTTP_202_ACCEPTED,
    )
    def decide_approval(thread_id: str, body: ApprovalDecision) -> TaskView:
        decision: dict[str, Any] = body.model_dump()
        return manager.approve(thread_id, decision)

    return app
