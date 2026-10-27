from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import replace

from multi_agent.config import Settings
from multi_agent.graph import SUPERVISOR, build_graph, message_text, run_task_full


def _make_on_event(enabled: bool):
    def on_event(node: str, update: dict) -> None:
        if not enabled:
            return
        messages = update.get("messages") or []
        if node == SUPERVISOR:
            turn = update.get("turn", "?")
            nxt = update.get("next_agent", "?")
            reason = ""
            if messages:
                text = message_text(messages[-1])
                reason = text.split("。", 1)[1] if "。" in text else ""
            suffix = f" — {reason}" if reason else ""
            print(f"  → 第 {turn} 轮：{nxt}{suffix}", flush=True)
        else:
            size = len(message_text(messages[-1])) if messages else 0
            print(f"  ✓ {node} 完成（{size} 字）", flush=True)

    return on_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 协作 CLI")
    parser.add_argument("task", nargs="*", help="要完成的任务；省略则进入交互")
    parser.add_argument("--thread-id", default=None, help="会话线程 ID（默认自动生成）")
    parser.add_argument("--model", default=None, help="覆盖 OPENAI_MODEL")
    parser.add_argument("--max-turns", type=int, default=None, help="覆盖 MULTI_AGENT_MAX_TURNS")
    parser.add_argument("--no-stream", action="store_true", help="不打印每轮进度")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    overrides = {}
    if args.model:
        overrides["model"] = args.model
        if not settings.supervisor_model:
            overrides["supervisor_model"] = args.model
    if args.max_turns:
        overrides["max_turns"] = args.max_turns
    if overrides:
        settings = replace(settings, **overrides)

    level = (args.log_level or settings.log_level or "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        app = build_graph(settings=settings)
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    on_event = _make_on_event(not args.no_stream)

    def run(text: str) -> None:
        result = run_task_full(
            text, settings=settings, app=app, thread_id=thread_id, on_event=on_event
        )
        print(f"\n{result.answer}\n", flush=True)

    task = " ".join(args.task).strip()
    if task:
        run(task)
        return 0

    print(
        f"多 Agent 协作已启动（thread_id={thread_id}）。"
        "输入任务回车执行；/new 新建会话；空行退出。"
    )
    while True:
        try:
            line = input("任务> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            return 0
        if line == "/new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            print(f"已新建会话 thread_id={thread_id}")
            continue
        run(line)


if __name__ == "__main__":
    raise SystemExit(main())
