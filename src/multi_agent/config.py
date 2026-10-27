from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    max_turns: int

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未设置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_turns=int(os.getenv("MULTI_AGENT_MAX_TURNS", "12")),
        )
