"""`multi-agent-api` 启动器：解析参数、延迟导入可选依赖、启动 uvicorn。"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from multi_agent.config import Settings, normalize_log_level

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_WORKERS = 4
DEFAULT_TASK_TIMEOUT = 1800.0
DEFAULT_RETENTION = 86_400.0
DEFAULT_MAX_RECORDS = 1000


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


def _env_float(name: str, default: float, *, low: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，当前值: {raw!r}") from exc
    if value < low:
        raise ValueError(f"{name} 必须 >= {low:g}，当前值: {value:g}")
    return value


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
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help=(
            f"单个任务执行超时秒数，0 表示不限（默认 {DEFAULT_TASK_TIMEOUT:g}，"
            "也可用 MULTI_AGENT_API_TASK_TIMEOUT）"
        ),
    )
    parser.add_argument(
        "--retention",
        type=float,
        default=None,
        help=(
            f"任务记录保留秒数，0 表示关闭回收（默认 {DEFAULT_RETENTION:g}，"
            "也可用 MULTI_AGENT_API_RETENTION）"
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            f"任务表容量上限（默认 {DEFAULT_MAX_RECORDS}，"
            "也可用 MULTI_AGENT_API_MAX_RECORDS）"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="DEBUG/INFO/WARNING/ERROR/CRITICAL，默认跟随 MULTI_AGENT_LOG_LEVEL",
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
        task_timeout = (
            args.task_timeout
            if args.task_timeout is not None
            else _env_float("MULTI_AGENT_API_TASK_TIMEOUT", DEFAULT_TASK_TIMEOUT, low=0.0)
        )
        if task_timeout < 0:
            raise ValueError(f"--task-timeout 必须 >= 0，当前值: {task_timeout:g}")
        retention = (
            args.retention
            if args.retention is not None
            else _env_float("MULTI_AGENT_API_RETENTION", DEFAULT_RETENTION, low=0.0)
        )
        if retention < 0:
            raise ValueError(f"--retention 必须 >= 0，当前值: {retention:g}")
        records = (
            args.max_records
            if args.max_records is not None
            else _env_int("MULTI_AGENT_API_MAX_RECORDS", DEFAULT_MAX_RECORDS, low=1)
        )
        _check_range("--max-records", records, low=1)
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
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    try:
        level = normalize_log_level(args.log_level or settings.log_level)
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        app = create_app(
            settings=settings,
            max_workers=workers,
            task_timeout=task_timeout,
            retention=retention,
            max_records=records,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    uvicorn.run(app, host=args.host, port=port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
