from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path("notes")


def workspace_tools() -> list:
    @tool
    def save_note(title: str, content: str) -> str:
        """把一条工作笔记保存到本地 notes/ 目录，供后续 Agent 阅读。"""
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)[:80]
        path = NOTES_DIR / f"{safe}.md"
        path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        return f"已保存笔记: {path}"

    @tool
    def read_note(title: str) -> str:
        """按标题读取 notes/ 下的笔记。"""
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)[:80]
        path = NOTES_DIR / f"{safe}.md"
        if not path.exists():
            return f"未找到笔记: {title}"
        return path.read_text(encoding="utf-8")

    @tool
    def list_notes() -> str:
        """列出 notes/ 目录中已有笔记。"""
        if not NOTES_DIR.exists():
            return "暂无笔记"
        names = sorted(p.stem for p in NOTES_DIR.glob("*.md"))
        return "\n".join(names) if names else "暂无笔记"

    return [save_note, read_note, list_notes]
