"""图编排共用的常量。

集中在一处，避免 routing / context_window / graph 互相导入形成环。
"""

from __future__ import annotations

FINISH = "FINISH"
SUPERVISOR = "supervisor"
APPROVAL_MARK = "REQUIRES_APPROVAL::"
APPROVAL_EVENT = "__interrupt__"