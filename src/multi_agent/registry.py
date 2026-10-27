from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool


ToolFactory = Callable[[], Sequence[BaseTool]]


@dataclass(frozen=True)
class AgentSpec:
    """注册一个可被 Supervisor 调度的专家 Agent。"""

    name: str
    description: str
    system_prompt: str
    tools_factory: ToolFactory = field(default=lambda: ())


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        if not spec.name or spec.name == "supervisor":
            raise ValueError("Agent 名称不能为空，且不能叫 supervisor")
        if spec.name in self._agents:
            raise ValueError(f"Agent 已存在: {spec.name}")
        self._agents[spec.name] = spec

    def get(self, name: str) -> AgentSpec:
        return self._agents[name]

    def names(self) -> list[str]:
        return list(self._agents)

    def descriptions_block(self) -> str:
        lines = [f"- {spec.name}: {spec.description}" for spec in self._agents.values()]
        return "\n".join(lines)

    def build_tools(self, spec: AgentSpec) -> list[BaseTool]:
        return list(spec.tools_factory())


def default_registry() -> AgentRegistry:
    from multi_agent.agents.defaults import register_default_agents

    registry = AgentRegistry()
    register_default_agents(registry)
    return registry


def make_llm(settings, *, temperature: float = 0) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
    )
