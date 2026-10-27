from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

ToolFactory = Callable[[], Sequence[BaseTool]]


@dataclass(frozen=True)
class AgentSpec:
    """注册一个可被 Supervisor 调度的专家 Agent。

    model / temperature 为空时跟随全局 OPENAI_MODEL；max_steps 为空时跟随全局
    MULTI_AGENT_WORKER_STEPS（专家内部工具循环的步数上限）。

    requires_approval=True 时，supervisor 选中该专家后先进入人工审批节点，
    批准后才真正执行（适用于上架、改价、发信等高风险动作）。
    tools_factory 每次调用都应返回新的工具列表，避免多个专家共享可变状态。
    """

    name: str
    description: str
    system_prompt: str
    tools_factory: ToolFactory = field(default=lambda: ())
    model: str | None = None
    temperature: float | None = None
    max_steps: int | None = None
    requires_approval: bool = False


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

    def specs(self) -> list[AgentSpec]:
        return list(self._agents.values())

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


def make_llm(settings, *, model: str | None = None, temperature: float = 0) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
