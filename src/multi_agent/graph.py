from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from multi_agent.config import Settings
from multi_agent.registry import AgentRegistry, default_registry, make_llm
from multi_agent.state import TeamState

FINISH = "FINISH"


class RouteDecision(BaseModel):
    next_agent: str = Field(description="下一个专家名称，或 FINISH 表示结束")
    reason: str = Field(description="一句话说明为什么选这个人")


def build_graph(
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
):
    settings = settings or Settings.from_env()
    registry = registry or default_registry()
    llm = make_llm(settings)
    agent_names = registry.names()
    if not agent_names:
        raise ValueError("至少注册一个 Agent")

    workers: dict[str, object] = {}
    for name in agent_names:
        spec = registry.get(name)
        workers[name] = create_react_agent(
            llm,
            tools=registry.build_tools(spec),
            prompt=spec.system_prompt,
        )

    def supervisor_node(state: TeamState) -> Command:
        turn = int(state.get("turn") or 0)
        if turn >= settings.max_turns:
            return Command(goto=END, update={"next_agent": FINISH})

        options = ", ".join(agent_names + [FINISH])
        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 Supervisor，只负责把任务分给专家，自己不写正文。\n"
                        f"可选对象: {options}\n"
                        f"专家职责:\n{registry.descriptions_block()}\n"
                        "规则:\n"
                        "1. 一次只选一个专家，或 FINISH。\n"
                        "2. 信息不足先 researcher，再 writer，再 reviewer。\n"
                        "3. 审查未通过则按意见返工。\n"
                        "4. 审查通过或已有完整答案则 FINISH。"
                    )
                ),
                *state["messages"],
            ]
        )
        nxt = decision.next_agent.strip()
        if nxt not in agent_names and nxt != FINISH:
            nxt = FINISH
        goto = END if nxt == FINISH else nxt
        return Command(
            goto=goto,
            update={
                "next_agent": nxt,
                "turn": turn + 1,
                "messages": [
                    AIMessage(
                        content=f"[supervisor] 下一步 -> {nxt}。{decision.reason}",
                        name="supervisor",
                    )
                ],
            },
        )

    def make_worker(name: str):
        compiled = workers[name]

        def node(state: TeamState) -> Command[Literal["supervisor"]]:
            result = compiled.invoke({"messages": state["messages"]})
            last = result["messages"][-1]
            content = last.content if hasattr(last, "content") else str(last)
            return Command(
                goto="supervisor",
                update={
                    "messages": [AIMessage(content=content, name=name)],
                },
            )

        node.__name__ = f"{name}_node"
        return node

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor_node)
    for name in agent_names:
        graph.add_node(name, make_worker(name))
    graph.add_edge(START, "supervisor")
    return graph.compile()


def run_task(task: str, *, settings: Settings | None = None, registry: AgentRegistry | None = None) -> str:
    app = build_graph(settings=settings, registry=registry)
    result = app.invoke(
        {
            "task": task,
            "next_agent": "supervisor",
            "turn": 0,
            "messages": [HumanMessage(content=task)],
        }
    )
    messages = result.get("messages") or []
    for msg in reversed(messages):
        name = getattr(msg, "name", None)
        if name and name != "supervisor":
            return str(msg.content)
        if isinstance(msg, AIMessage) and name != "supervisor":
            return str(msg.content)
    if messages:
        return str(messages[-1].content)
    return ""
