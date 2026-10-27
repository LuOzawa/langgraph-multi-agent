from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool


def _notes_dir() -> Path:
    """笔记目录，默认 ./notes；可用 MULTI_AGENT_NOTES_DIR 指定绝对路径。"""
    return Path(os.getenv("MULTI_AGENT_NOTES_DIR", "notes")).expanduser()


def _safe_stem(title: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title).strip(".")
    return (cleaned or "note")[:80]


def workspace_tools() -> list:
    """每次调用返回新的工具列表，避免多个专家共享可变状态。"""

    @tool
    def save_note(title: str, content: str) -> str:
        """把一条工作笔记保存到本地 notes/ 目录，供后续 Agent 阅读。"""
        directory = _notes_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_stem(title)}.md"
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        return f"已保存笔记: {path}"

    @tool
    def read_note(title: str) -> str:
        """按标题读取 notes/ 下的笔记。"""
        path = _notes_dir() / f"{_safe_stem(title)}.md"
        if not path.exists():
            return f"未找到笔记: {title}"
        return path.read_text(encoding="utf-8")

    @tool
    def list_notes() -> str:
        """列出 notes/ 目录中已有笔记。"""
        directory = _notes_dir()
        if not directory.exists():
            return "暂无笔记"
        names = sorted(p.stem for p in directory.glob("*.md"))
        return "\n".join(names) if names else "暂无笔记"

    return [save_note, read_note, list_notes]
