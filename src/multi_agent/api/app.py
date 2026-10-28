"""FastAPI 应用工厂：把 TaskManager 暴露为 HTTP 接口。"""

from __future__ import annotations

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


def create_app(
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    executor: Executor | None = None,
    max_workers: int = 4,
    task_timeout: float = 1800.0,
    retention: float = 86_400.0,
    max_records: int = 1000,
) -> FastAPI:
    """构建应用；settings 为空时读环境变量（缺 key 会抛 RuntimeError）。"""
    settings = settings or Settings.from_env()
    executor = executor or LangGraphExecutor(settings, registry)
    manager = TaskManager(
        executor,
        max_workers=max_workers,
        task_timeout=task_timeout,
        retention=retention,
        max_records=max_records,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="multi-agent API", version=_package_version(), lifespan=lifespan)
    app.state.manager = manager

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
