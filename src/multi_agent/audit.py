"""审批审计：把每次人工决策追加写入 JSONL，便于合规追溯。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_agent.config import Settings

log = logging.getLogger("multi_agent.audit")

_DISABLED = {"", "off", "none", "false"}


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
    try:
        if str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # 审计失败不应中断业务
        log.warning("写审批日志失败（%s）: %s", target, exc)
