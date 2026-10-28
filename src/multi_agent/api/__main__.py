"""`multi-agent-api` 启动器：解析参数、延迟导入可选依赖、启动 uvicorn。"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_WORKERS = 4


def _env_int(name: str, default: int, *, low: int, high: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值: {raw!r}") from exc
    _check_range(name, value, low=low, high=high)
    return value


def _check_range(name: str, value: int, *, low: int, high: int | None = None) -> None:
    if high is None:
        if value < low:
            raise ValueError(f"{name} 必须 >= {low}，当前值: {value}")
        return
    if not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low}..{high}，当前值: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="multi-agent-api", description="多 Agent 协作 HTTP API 服务"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MULTI_AGENT_API_HOST", DEFAULT_HOST),
        help=f"监听地址（默认 {DEFAULT_HOST}）",
    )
    parser.add_argument("--port", type=int, default=None, help=f"监听端口（默认 {DEFAULT_PORT}）")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"并发任务线程数（默认 {DEFAULT_MAX_WORKERS}）",
    )
    args = parser.parse_args(argv)

    try:
        port = (
            args.port
            if args.port is not None
            else _env_int("MULTI_AGENT_API_PORT", DEFAULT_PORT, low=1, high=65535)
        )
        _check_range("--port", port, low=1, high=65535)
        workers = (
            args.max_workers
            if args.max_workers is not None
            else _env_int("MULTI_AGENT_API_MAX_WORKERS", DEFAULT_MAX_WORKERS, low=1)
        )
        _check_range("--max-workers", workers, low=1)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    try:
        import uvicorn

        from multi_agent.api.app import create_app
    except ImportError:
        print('缺少 API 依赖：pip install "multi-agent[api]"', file=sys.stderr)
        return 1

    try:
        app = create_app(max_workers=workers)
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    uvicorn.run(app, host=args.host, port=port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
