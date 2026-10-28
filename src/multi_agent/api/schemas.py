"""HTTP 层的请求 / 响应模型；字段名即线上 JSON 形状。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskSubmit(BaseModel):
    task: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )

    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task must not be blank")
        return value


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=2_000)

    @field_validator("note")
    @classmethod
    def _strip_note(cls, value: str) -> str:
        return value.strip()


class TaskView(BaseModel):
    thread_id: str
    task: str
    status: TaskStatus
    answer: str
    turns: int
    stop_reason: str
    pending: dict[str, Any] | None
    usage: dict[str, int]
    error: str | None
    created_at: datetime
    updated_at: datetime


class ApprovalView(BaseModel):
    thread_id: str
    task: str
    pending: dict[str, Any]
    updated_at: datetime
