"""任务表持久化：把 TaskManager 的记录写进 SQLite，重启后可继续查询 / 审批。

只存「已经发生的状态」，不存图实例或线程：重启后 `queued` / `running` 视为中断失败，
`awaiting_approval` 保留（审批时用同一 registry 重建图继续）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    thread_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    turns INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT NOT NULL DEFAULT '',
    pending TEXT,
    usage TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

COLUMNS = (
    "thread_id",
    "task",
    "status",
    "answer",
    "turns",
    "stop_reason",
    "pending",
    "usage",
    "error",
    "created_at",
    "updated_at",
)


class TaskStore:
    """SQLite 任务表；`memory` 之类的值由调用方判断后不创建实例。"""

    def __init__(self, path: str) -> None:
        target = Path(path).expanduser()
        if str(target.parent) not in {"", "."}:
            target.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(target), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def upsert(self, row: Mapping[str, Any]) -> None:
        values = tuple(row[column] for column in COLUMNS)
        placeholders = ", ".join("?" for _ in COLUMNS)
        updates = ", ".join(f"{column}=excluded.{column}" for column in COLUMNS[1:])
        with self._lock:
            self._conn.execute(
                f"INSERT INTO tasks ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(thread_id) DO UPDATE SET {updates}",
                values,
            )
            self._conn.commit()

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE thread_id = ?", (thread_id,))
            self._conn.commit()

    def load(self) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(f"SELECT {', '.join(COLUMNS)} FROM tasks")
            return [dict(zip(COLUMNS, row, strict=True)) for row in cursor.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def encode_json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def decode_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:  # 坏数据不应让服务起不来
        return None
