"""可扩展的 LangGraph 多 Agent 协作框架。"""

from multi_agent.graph import build_graph, run_task
from multi_agent.registry import AgentRegistry, AgentSpec

__all__ = ["AgentRegistry", "AgentSpec", "build_graph", "run_task"]
