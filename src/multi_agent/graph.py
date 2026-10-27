"""Supervisor 图编排。

相对初版的改动：
1. 默认带 checkpointer（SQLite；不可用或 memory 模式时退回内存），支持 thread_id 多轮与中断续跑。
2. 专家只看到业务消息，过滤 supervisor 的路由话术。
3. 专家返回原始 AIMessage（保留 content blocks），不再压成字符串。
4. 路由枚举由注册表动态生成；structured output 不可用时退化为 JSON 解析。
5. 单个专家失败不拖垮整轮，错误作为消息交回 supervisor。
6. 每轮写日志，run_task_full 支持 on_event 进度回调。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from multi_agent.config import Settings
from multi_agent.registry import AgentRegistry, default_registry, make_llm
from multi_agent.state import TeamState

try:  # langchain >= 1.0 的新入口；老版本退回 langgraph.prebuilt
    from langchain.agents import create_agent as _create_agent
except Exception:  # pragma: no cover - 取决于安装版本
    _create_agent = None

from langgraph.prebuilt import create_react_agent

log = logging.getLogger("multi_agent")

FINISH = "FINISH"
SUPERVISOR = "supervisor"


def message_text(message: Any) -> str:
    """把 AIMessage.content（str 或 content blocks 列表）转成纯文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _final_ai(messages: list[Any]) -> Any | None:
    """取最后一条没有 tool_calls 的 AIMessage；没有则退回最后一条消息。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return message
    return messages[-1] if messages else None


def _business_messages(state: TeamState) -> list[Any]:
    """专家可见消息：过滤 supervisor 的路由记录，避免污染上下文。"""
    messages = list(state.get("messages") or [])
    visible = [m for m in messages if getattr(m, "name", None) != SUPERVISOR]
    return visible or messages


def _route_schema(agent_names: list[str]) -> type[BaseModel]:
    allowed = tuple(agent_names) + (FINISH,)
    NextAgent = Literal[allowed]  # type: ignore[valid-type]
    return create_model(
        "RouteDecision",
        next_agent=(NextAgent, Field(description="下一个专家名称，或 FINISH 表示结束")),
        reason=(str, Field(description="一句话说明为什么选这个人")),
    )


def _supervisor_prompt(registry: AgentRegistry, agent_names: list[str]) -> SystemMessage:
    options = ", ".join(agent_names + [FINISH])
    return SystemMessage(
        content=(
            "你是 Supervisor，只负责把任务分给专家，自己不写正文。\n"
            f"可选对象: {options}\n"
            f"专家职责:\n{registry.descriptions_block()}\n"
            "规则:\n"
            "1. 一次只选一个专家，或 FINISH。\n"
            "2. 信息不足先 researcher，再 writer，再 reviewer。\n"
            "3. 审查未通过则按意见返工。\n"
            "4. 审查通过或已有完整答案则 FINISH。\n"
            "5. 专家报错时换人重试，不要直接放弃。"
        )
    )


def _decide(
    llm: Any,
    schema: type[BaseModel],
    registry: AgentRegistry,
    agent_names: list[str],
    history: list[Any],
) -> BaseModel:
    prompt = _supervisor_prompt(registry, agent_names)
    try:
        router = llm.with_structured_output(schema)
        return router.invoke([prompt, *history])
    except Exception as exc:  # 中转不支持 function calling / response_format
        log.warning("structured output 不可用（%s: %s），退化为 JSON 解析", type(exc).__name__, exc)
        return _decide_json(llm, schema, prompt, history, agent_names)


def _decide_json(
    llm: Any,
    schema: type[BaseModel],
    prompt: SystemMessage,
    history: list[Any],
    agent_names: list[str],
) -> BaseModel:
    options = ", ".join(agent_names + [FINISH])
    ask = HumanMessage(
        content=(
            "只输出一个 JSON 对象，不要解释、不要代码块："
            '{"next_agent": "<' + options + ' 之一>", "reason": "<一句话>"}'
        )
    )
    raw = llm.invoke([prompt, *history, ask])
    text = message_text(raw)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"无法从模型输出解析路由: {text[:200]!r}")
    data = json.loads(match.group(0))
    return schema(**data)


_SAVER_CACHE: dict[str, Any] = {}


def make_checkpointer(path: str | None):
    """SQLite 检查点；路径为空、memory 或依赖缺失时退回内存。"""
    value = (path or "").strip()
    if not value or value.lower() in {"memory", ":memory:", "none"}:
        return InMemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:  # pragma: no cover - 可选依赖
        log.warning("未安装 langgraph-checkpoint-sqlite，本次使用内存检查点")
        return InMemorySaver()

    db = Path(value).expanduser()
    if str(db.parent) not in {"", "."}:
        db.parent.mkdir(parents=True, exist_ok=True)
    key = str(db.resolve())
    if key not in _SAVER_CACHE:
        connection = sqlite3.connect(key, check_same_thread=False)
        _SAVER_CACHE[key] = SqliteSaver(connection)
        log.debug("检查点数据库: %s", key)
    return _SAVER_CACHE[key]


def build_graph(
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    checkpointer: Any | None = None,
    checkpoint_path: str | None = None,
):
    settings = settings or Settings.from_env()
    registry = registry or default_registry()
    agent_names = registry.names()
    if not agent_names:
        raise ValueError("至少注册一个 Agent")

    supervisor_llm = make_llm(settings, model=settings.supervisor_model, temperature=0)
    route_schema = _route_schema(agent_names)
    worker_graphs: dict[str, Any] = {}

    for name in agent_names:
        spec = registry.get(name)
        llm = make_llm(
            settings,
            model=spec.model,
            temperature=spec.temperature if spec.temperature is not None else 0,
        )
        tools = registry.build_tools(spec)
        if _create_agent is not None:
            worker_graphs[name] = _create_agent(llm, tools, system_prompt=spec.system_prompt)
        else:  # pragma: no cover - 兼容旧版 langgraph
            worker_graphs[name] = create_react_agent(llm, tools=tools, prompt=spec.system_prompt)

    def supervisor_node(state: TeamState) -> Command:
        turn = int(state.get("turn") or 0)
        if turn >= settings.max_turns:
            log.warning("达到 max_turns=%s，强制结束", settings.max_turns)
            return Command(
                goto=END,
                update={
                    "next_agent": FINISH,
                    "stop_reason": "max_turns",
                    "messages": [
                        AIMessage(
                            content=f"[supervisor] 达到最大调度次数 {settings.max_turns}，停止。",
                            name=SUPERVISOR,
                        )
                    ],
                },
            )

        history = _business_messages(state)
        try:
            decision = _decide(supervisor_llm, route_schema, registry, agent_names, history)
        except Exception as exc:
            log.exception("supervisor 路由失败")
            return Command(
                goto=END,
                update={
                    "next_agent": FINISH,
                    "stop_reason": "error",
                    "messages": [
                        AIMessage(
                            content=f"[supervisor] 路由失败，任务终止：{type(exc).__name__}: {exc}",
                            name=SUPERVISOR,
                        )
                    ],
                },
            )

        nxt = str(getattr(decision, "next_agent", "")).strip()
        reason = str(getattr(decision, "reason", "") or "")
        if nxt not in agent_names and nxt != FINISH:
            log.warning("supervisor 返回非法目标 %r，按 FINISH 处理", nxt)
            reason = f"（模型给出非法目标 {nxt!r}）{reason}"
            nxt = FINISH

        log.info("turn %d -> %s | %s", turn + 1, nxt, reason)
        return Command(
            goto=END if nxt == FINISH else nxt,
            update={
                "next_agent": nxt,
                "turn": turn + 1,
                "stop_reason": "finish" if nxt == FINISH else "",
                "messages": [
                    AIMessage(content=f"[supervisor] 下一步 -> {nxt}。{reason}", name=SUPERVISOR)
                ],
            },
        )

    def make_worker(name: str):
        compiled = worker_graphs[name]

        def node(state: TeamState) -> Command:
            log.info("agent %s 开始", name)
            try:
                result = compiled.invoke(
                    {"messages": _business_messages(state)},
                    {"recursion_limit": settings.worker_max_steps},
                )
            except Exception as exc:
                log.exception("agent %s 执行失败", name)
                return Command(
                    goto=SUPERVISOR,
                    update={
                        "messages": [
                            AIMessage(
                                content=f"[{name}] 执行失败：{type(exc).__name__}: {exc}",
                                name=name,
                            )
                        ]
                    },
                )

            last = _final_ai(list(result.get("messages") or []))
            if last is None:
                message = AIMessage(content=f"[{name}] 没有产出", name=name)
            elif hasattr(last, "model_copy"):
                message = last.model_copy(update={"name": name})
            else:  # pragma: no cover
                message = AIMessage(content=message_text(last), name=name)
            log.info("agent %s 完成（%d 字）", name, len(message_text(message)))
            return Command(goto=SUPERVISOR, update={"messages": [message]})

        node.__name__ = f"{name}_node"
        return node

    graph = StateGraph(TeamState)
    graph.add_node(SUPERVISOR, supervisor_node)
    for name in agent_names:
        graph.add_node(name, make_worker(name))
    graph.add_edge(START, SUPERVISOR)

    saver = checkpointer if checkpointer is not None else make_checkpointer(
        checkpoint_path if checkpoint_path is not None else settings.checkpoint_db
    )
    return graph.compile(checkpointer=saver)


@dataclass
class RunResult:
    answer: str
    thread_id: str
    turns: int
    messages: list[Any] = field(default_factory=list)
    stop_reason: str = ""


EventCallback = Callable[[str, dict[str, Any]], None]


def run_config(settings: Settings, thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(2 * settings.max_turns + 5, 25),
    }


def _final_answer(state: dict[str, Any]) -> str:
    messages = list(state.get("messages") or [])
    answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "name", None) != SUPERVISOR:
            answer = message_text(message)
            break
    if not answer and messages:
        answer = message_text(messages[-1])
    if state.get("stop_reason") == "max_turns":
        answer = f"{answer}\n\n（达到最大调度次数上限，结果可能不完整）"
    return answer


def run_task_full(
    task: str,
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    thread_id: str | None = None,
    on_event: EventCallback | None = None,
    app: Any | None = None,
) -> RunResult:
    settings = settings or Settings.from_env()
    app = app or build_graph(settings=settings, registry=registry)
    tid = thread_id or f"task-{uuid.uuid4().hex[:8]}"
    config = run_config(settings, tid)
    inputs = {
        "task": task,
        "next_agent": SUPERVISOR,
        "turn": 0,
        "stop_reason": "",
        "messages": [HumanMessage(content=task)],
    }
    log.info("启动任务 thread_id=%s: %s", tid, task[:80])
    for chunk in app.stream(inputs, config, stream_mode="updates"):
        for node, update in chunk.items():
            if on_event is not None:
                on_event(node, update if isinstance(update, dict) else {"value": update})
    state = app.get_state(config).values or {}
    return RunResult(
        answer=_final_answer(state),
        thread_id=tid,
        turns=int(state.get("turn") or 0),
        messages=list(state.get("messages") or []),
        stop_reason=str(state.get("stop_reason") or ""),
    )


def run_task(
    task: str,
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    thread_id: str | None = None,
) -> str:
    return run_task_full(
        task, settings=settings, registry=registry, thread_id=thread_id
    ).answer
