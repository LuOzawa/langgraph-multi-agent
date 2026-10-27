from __future__ import annotations

from langgraph.graph import MessagesState


class TeamState(MessagesState):
    """共享工作区状态。新增字段时保持可序列化。"""

    task: str
    next_agent: str
    turn: int
