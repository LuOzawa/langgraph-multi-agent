from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CHECKPOINT_DB = ".multi_agent/checkpoints.sqlite"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是整数，当前值: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} 必须是正整数，当前值: {value}")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是数字，当前值: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    supervisor_model: str
    max_turns: int
    worker_max_steps: int
    checkpoint_db: str
    request_timeout: float
    max_retries: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
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
            checkpoint_db=os.getenv("MULTI_AGENT_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB).strip(),
            request_timeout=_float_env("OPENAI_TIMEOUT", 120.0),
            max_retries=_int_env("OPENAI_MAX_RETRIES", 2),
            log_level=os.getenv("MULTI_AGENT_LOG_LEVEL", "WARNING").strip().upper() or "WARNING",
        )
