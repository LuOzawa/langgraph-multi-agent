"""审批审计：把每次人工决策追加写入 JSONL，便于合规追溯。"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_agent.config import Settings

log = logging.getLogger("multi_agent.audit")

_DISABLED = {"", "off", "none", "false"}

# run_parallel 会把多个任务丢进线程池、同写一个审计文件；
# Windows 上 `open("a")` + write 并非原子追加，实测 400 条只剩 151 条、45 行坏 JSON。
# 这里用进程内锁串行化写入（跨进程并发不在本模块保证范围内）。
_WRITE_LOCK = threading.Lock()


def record_approval(
    *,
    settings: Settings,
    thread_id: str,
    payload: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    """写入一条审批记录；路径由 MULTI_AGENT_AUDIT_LOG 控制。"""
    target = (settings.audit_log or "").strip()
    if target.lower() in _DISABLED:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "type": payload.get("type", "approval"),
        "agent": payload.get("agent"),
        "tool": payload.get("tool"),
        "approved": bool(decision.get("approved")),
        "note": str(decision.get("note", "")),
        "task": str(payload.get("task", ""))[:500],
    }
    path = Path(target).expanduser()
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        with _WRITE_LOCK:
            if str(path.parent) not in {"", "."}:
                path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except OSError as exc:  # 审计失败不应中断业务
        log.warning("写审批日志失败（%s）: %s", target, exc)
