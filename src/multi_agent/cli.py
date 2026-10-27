from __future__ import annotations

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage

from multi_agent.config import Settings
from multi_agent.graph import build_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 协作 CLI")
    parser.add_argument("task", nargs="*", help="要完成的任务；省略则进入交互")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
        app = build_graph(settings=settings)
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    task = " ".join(args.task).strip()
    if not task:
        print("多 Agent 协作已启动。输入任务后回车，空行退出。")
        try:
            task = input("任务> ").strip()
        except EOFError:
            return 0
        if not task:
            return 0

    result = app.invoke(
        {
            "task": task,
            "next_agent": "supervisor",
            "turn": 0,
            "messages": [HumanMessage(content=task)],
        }
    )
    for msg in result.get("messages") or []:
        name = getattr(msg, "name", None) or msg.__class__.__name__
        if isinstance(msg, (AIMessage, HumanMessage)):
            print(f"\n[{name}]\n{msg.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
