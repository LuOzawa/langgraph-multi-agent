from __future__ import annotations

from langgraph.graph import MessagesState


class TeamState(MessagesState):
    """共享工作区状态。

    字段语义：
    - messages: 业务消息 + supervisor 路由记录（name="supervisor"）。专家节点不会看到路由记录。
    - task: 当前任务原文。
    - turn: supervisor 已调度次数，用于 max_turns 保护（覆盖语义，无 reducer）。
    """

    task: str
    turn: int
    stop_reason: str
