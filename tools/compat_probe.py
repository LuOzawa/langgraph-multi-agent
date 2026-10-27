#!/usr/bin/env python
"""模型 / 中转兼容性探针。

用法：
    python tools/compat_probe.py --provider deepseek
    python tools/compat_probe.py --base-url https://... --model ... --api-key ...

三个探针：普通对话、工具调用、structured output（json_schema）。
不读项目的 .env，只读 --api-key 或本机 openclaw provider 配置。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

DEFAULT_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


class Route(BaseModel):
    next_agent: str = Field(description="专家名或 FINISH")
    reason: str = Field(description="理由")


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}: 晴 25C"


def _from_openclaw(provider: str, config: Path) -> tuple[str, str, str]:
    data = json.loads(config.read_text(encoding="utf-8"))
    entry = data.get("models", {}).get("providers", {}).get(provider)
    if not entry:
        raise SystemExit(f"openclaw 配置里没有 provider: {provider}")
    models = entry.get("models") or []
    if not models:
        raise SystemExit(f"provider {provider} 没有配置模型")
    return entry["apiKey"], entry["baseUrl"], models[0]["id"]


def _attempt(name: str, fn) -> dict:
    started = time.time()
    try:
        return {"ok": True, "latency_s": round(time.time() - started, 2), "detail": fn()}
    except Exception as exc:  # noqa: BLE001 - 探针要保留原始错误
        return {
            "ok": False,
            "latency_s": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {str(exc)[:220]}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="deepseek", help="openclaw.json 里的 provider 名")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=40)
    args = parser.parse_args()

    if args.api_key:
        api_key, base_url, model = args.api_key, args.base_url or "", args.model or ""
        if not base_url or not model:
            raise SystemExit("使用 --api-key 时需要同时提供 --base-url 与 --model")
    else:
        api_key, base_url, model = _from_openclaw(args.provider, args.config)

    llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url,
        timeout=args.timeout, max_retries=0, max_tokens=64, temperature=0,
    )

    results = {
        "chat": _attempt("chat", lambda: llm.invoke("用三个字回答：你好吗").content[:80]),
        "tool_calling": _attempt(
            "tool_calling",
            lambda: {
                "tool_calls": [
                    call["name"]
                    for call in (
                        llm.bind_tools([get_weather]).invoke("杭州天气怎么样？请调用工具。").tool_calls or []
                    )
                ]
            },
        ),
        "structured_output": _attempt(
            "structured_output",
            lambda: llm.with_structured_output(Route).invoke("任务已完成，请选择 FINISH。").model_dump(),
        ),
    }

    print(json.dumps({"model": model, "base_url": base_url, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
