from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import replace

from multi_agent.config import Settings, normalize_log_level
from multi_agent.graph import (
    SUPERVISOR,
    build_graph,
    message_text,
    pending_approval,
    resume_task,
    run_task_full,
)


def _make_on_event(enabled: bool):
    def on_event(node: str, update: dict) -> None:
        if not enabled:
            return
        messages = update.get("messages") or []
        if node == SUPERVISOR:
            turn = update.get("turn", "?")
            nxt, reason = "?", ""
            if messages:
                text = message_text(messages[-1])
                if "下一步 -> " in text:
                    nxt = text.split("下一步 -> ", 1)[1].split("。", 1)[0]
                reason = text.split("。", 1)[1] if "。" in text else ""
            suffix = f" — {reason}" if reason else ""
            print(f"  → 第 {turn} 轮：{nxt}{suffix}", flush=True)
        else:
            size = len(message_text(messages[-1])) if messages else 0
            print(f"  ✓ {node} 完成（{size} 字）", flush=True)

    return on_event


def _prompt_approval(payload: dict) -> dict:
    agent = payload.get("agent", "?")
    tool = payload.get("tool")
    reason = payload.get("reason", "")
    task = str(payload.get("task", ""))[:160]
    preview = str(payload.get("preview", ""))
    label = f"{agent}/{tool}" if tool else agent
    print(f"\n[需要审批] {label} — {reason}")
    if task:
        print(f"  任务: {task}")
    if tool and payload.get("args"):
        print(f"  参数: {payload['args']}")
    if preview:
        print("  ---- 待确认内容 ----")
        for line in preview.splitlines()[:12]:
            print(f"  {line}")
        print("  --------------------")
    try:
        answer = input("  批准执行？[y/N] ").strip().lower()
    except EOFError:
        return {"approved": False, "note": "非交互环境，默认拒绝"}
    approved = answer in {"y", "yes"}
    note = ""
    if approved:
        try:
            note = input("  备注（可空）: ").strip()
        except EOFError:
            note = ""  # 备注读到 EOF 不影响已经给出的批准
    return {"approved": approved, "note": note}


def _exit_code(result) -> int:
    """失败任务返回非零，便于脚本 / CI 判断。"""
    return 1 if result.stop_reason == "error" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 协作 CLI")
    parser.add_argument("task", nargs="*", help="要完成的任务；省略则进入交互")
    parser.add_argument("--thread-id", default=None, help="会话线程 ID（默认自动生成）")
    parser.add_argument("--model", default=None, help="覆盖 OPENAI_MODEL")
    parser.add_argument("--max-turns", type=int, default=None, help="覆盖 MULTI_AGENT_MAX_TURNS")
    parser.add_argument("--no-stream", action="store_true", help="不打印每轮进度")
    parser.add_argument(
        "--auto-approve", action="store_true", help="自动批准所有人工审批（仅限调试）"
    )
    parser.add_argument("--resume", default=None, help="恢复指定 thread 上挂起的审批")
    parser.add_argument(
        "--log-level", default=None, help="DEBUG / INFO / WARNING / ERROR / CRITICAL"
    )
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
        app = build_graph(settings=settings)
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    on_event = _make_on_event(not args.no_stream)
    approver = None if args.auto_approve else _prompt_approval

    def show(result) -> None:
        if result.pending is not None:
            print(f"\n[待审批] {result.pending}", flush=True)
            print(
                f"处理：/resume y  或  /resume n <原因>；退出后可用 --resume {result.thread_id}\n",
                flush=True,
            )
            return
        tokens = result.usage or {}
        print(f"\n{result.answer}\n", flush=True)
        print(
            f"[thread={result.thread_id} turns={result.turns} "
            f"tokens={tokens.get('total_tokens', 0)}]\n",
            flush=True,
        )

    def run(text: str):
        result = run_task_full(
            text,
            settings=settings,
            app=app,
            thread_id=thread_id,
            on_event=on_event,
            approver=approver,
            auto_approve=args.auto_approve,
        )
        show(result)
        return result

    def resume(decision: dict):
        result = resume_task(
            thread_id,
            decision,
            settings=settings,
            app=app,
            on_event=on_event,
            approver=approver,
            auto_approve=args.auto_approve,
        )
        show(result)
        return result

    if args.resume:
        payload = pending_approval(app, args.resume)
        if payload is None:
            print(f"thread {args.resume} 上没有挂起的审批", file=sys.stderr)
            return 1
        print(f"[待审批] {payload}")
        decision = (
            {"approved": True, "note": "auto-approve"}
            if args.auto_approve
            else _prompt_approval(payload)
        )
        thread_id = args.resume
        return _exit_code(resume(decision))

    task = " ".join(args.task).strip()
    if task:
        return _exit_code(run(task))

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
        if line == "/pending":
            payload = pending_approval(app, thread_id)
            print(f"挂起审批: {payload}" if payload else "当前会话没有挂起的审批")
            continue
        if line.startswith("/resume"):
            parts = line.split(maxsplit=2)
            approved = len(parts) > 1 and parts[1].lower() in {"y", "yes"}
            note = parts[2] if len(parts) > 2 else ""
            try:
                resume({"approved": approved, "note": note})
            except ValueError as exc:  # 没有挂起审批
                print(f"无法恢复：{exc}", file=sys.stderr)
            continue
        run(line)


if __name__ == "__main__":
    raise SystemExit(main())
