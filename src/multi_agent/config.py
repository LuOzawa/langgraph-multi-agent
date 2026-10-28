from __future__ import annotations

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CHECKPOINT_DB = ".multi_agent/checkpoints.sqlite"

_LEVEL_ORDER = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def normalize_log_level(value: str | None) -> str:
    """把日志级别归一化；非法值直接报错，避免静默退回 WARNING。"""
    level = (value or "").strip().upper() or "WARNING"
    if level not in _LEVEL_ORDER:
        raise RuntimeError(
            f"日志级别非法，当前值: {value!r}（可选 {' / '.join(_LEVEL_ORDER)}）"
        )
    return level


def _int_env(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是整数，当前值: {raw!r}") from exc
    if value < 0 or (value == 0 and not allow_zero):
        raise RuntimeError(f"{name} 必须是正整数，当前值: {value}")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是数字，当前值: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} 必须是正数，当前值: {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    supervisor_model: str
    max_turns: int
    worker_max_steps: int
    context_max_tokens: int
    checkpoint_db: str
    audit_log: str
    request_timeout: float
    max_retries: int
    log_level: str
    max_rejections: int = 1

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未设置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
            )
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return cls(
            api_key=api_key,
            base_url=(os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL).rstrip("/"),
            model=model,
            supervisor_model=(os.getenv("MULTI_AGENT_SUPERVISOR_MODEL", "").strip() or model),
            max_turns=_int_env("MULTI_AGENT_MAX_TURNS", 12),
            worker_max_steps=_int_env("MULTI_AGENT_WORKER_STEPS", 25),
            context_max_tokens=_int_env("MULTI_AGENT_CONTEXT_TOKENS", 12000, allow_zero=True),
            checkpoint_db=os.getenv("MULTI_AGENT_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB).strip(),
            audit_log=os.getenv("MULTI_AGENT_AUDIT_LOG", ".multi_agent/approvals.jsonl").strip(),
            request_timeout=_float_env("OPENAI_TIMEOUT", 120.0),
            max_retries=_int_env("OPENAI_MAX_RETRIES", 2),
            log_level=normalize_log_level(os.getenv("MULTI_AGENT_LOG_LEVEL")),
            max_rejections=_int_env("MULTI_AGENT_MAX_REJECTIONS", 1, allow_zero=True),
        )
