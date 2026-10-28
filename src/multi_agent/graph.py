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
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, create_model

from multi_agent.audit import record_approval
from multi_agent.config import Settings
from multi_agent.registry import AgentRegistry, default_registry, make_llm
from multi_agent.state import TeamState

_create_agent: Any = None
try:  # langchain >= 1.0 的新入口；老版本退回 langgraph.prebuilt
    from langchain.agents import create_agent as _agent_factory

    _create_agent = _agent_factory
except Exception:  # pragma: no cover - 取决于安装版本
    pass

log = logging.getLogger("multi_agent")

FINISH = "FINISH"
SUPERVISOR = "supervisor"
_APPROVAL_MARK = "REQUIRES_APPROVAL::"
_APPROVED_TOOLS: ContextVar[frozenset[str]] = ContextVar(
    "multi_agent_approved_tools", default=frozenset()
)


def _guarded_tool(original: BaseTool) -> BaseTool:
    """包一层审批闸：未批准时只回标记，不真正执行工具。"""

    def _run(**kwargs: Any) -> Any:
        if original.name not in _APPROVED_TOOLS.get():
            payload = json.dumps(kwargs, ensure_ascii=False, default=str)
            return f"{_APPROVAL_MARK}{original.name}::{payload}"
        return original.invoke(kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
    )


def _approval_markers(messages: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """从工具返回里找出“需要审批”的标记。"""
    found: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content if isinstance(message.content, str) else ""
        if not content.startswith(_APPROVAL_MARK):
            continue
        try:
            _, tool, raw = content.split("::", 2)
            args = json.loads(raw) if raw else {}
        except (ValueError, json.JSONDecodeError):
            continue
        found.append((tool, args if isinstance(args, dict) else {}))
    return found


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


def _count_tokens(llm: Any, messages: list[Any]) -> int:
    try:
        return int(llm.get_num_tokens_from_messages(messages))
    except Exception:  # 假模型/不支持计数的模型，用字符数估算
        return sum(len(message_text(m)) for m in messages) // 2 + 1


def _trim_for_model(messages: list[Any], llm: Any, settings: Settings) -> list[Any]:
    """按 token 预算裁剪历史，保留首条任务消息。

    MULTI_AGENT_CONTEXT_TOKENS=0 时关闭裁剪。
    """
    budget = settings.context_max_tokens
    if budget <= 0 or not messages:
        return messages
    if _count_tokens(llm, messages) <= budget:
        return messages

    trimmed: list[Any]
    try:
        from langchain_core.messages import trim_messages

        trimmed = list(
            trim_messages(
                messages,
                max_tokens=budget,
                strategy="last",
                token_counter=lambda msgs: _count_tokens(llm, msgs),
                start_on="human",
                include_system=True,
            )
        )
    except Exception:
        log.warning("trim_messages 不可用，退回按条数裁剪")
        trimmed = messages[-max(2, budget // 200):]

    first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
    if first_human is not None and all(m is not first_human for m in trimmed):
        trimmed = [
            first_human,
            AIMessage(content="（为控制上下文长度，已省略部分较早消息）", name=SUPERVISOR),
            *trimmed,
        ]
    log.info("上下文裁剪: %d -> %d 条（预算 %d tokens）", len(messages), len(trimmed), budget)
    return trimmed


def _last_output(state: TeamState) -> str:
    """给审批人看的“将要提交的内容”预览。"""
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, AIMessage) and getattr(message, "name", None) not in {
            SUPERVISOR,
            "approval",
        }:
            return message_text(message)[:1500]
    return ""


def _route_schema(agent_names: list[str]) -> type[BaseModel]:
    allowed = tuple(agent_names) + (FINISH,)
    NextAgent = Literal[allowed]  # type: ignore[valid-type]
    return create_model(
        "RouteDecision",
        next_agent=(
            NextAgent,
            Field(description="下一个要执行的专家名；FINISH 表示现有产出已足以收尾"),
        ),
        reason=(
            str,
            Field(
                description=(
                    "一句话，不超过 30 字：已有产出或缺口 → 这一步要它做什么。"
                    "不要复述任务、不要列计划"
                )
            ),
        ),
    )


def _supervisor_prompt(registry: AgentRegistry, agent_names: list[str]) -> SystemMessage:
    options = ", ".join(agent_names + [FINISH])
    # 规则按职责语义判定，不写死默认三专家；默认团队在场时只补一句精确定位
    hint = ""
    if {"researcher", "writer", "reviewer"} <= set(agent_names):
        hint = "（默认团队中：调研型=researcher，产出型=writer，审查型=reviewer）"
    rules = [
        "你是 Supervisor，只负责把任务分给专家；你不写正文、不补内容、不排多步计划。",
        f"可选对象: {options}",
        f"专家职责:\n{registry.descriptions_block()}",
        "角色判定（按上方职责表的实际措辞判断，不要只按名字猜）: "
        "交付事实或要点的是调研型；交付正文或成品的是产出型；"
        "判断通过或驳回的是审查型；执行外部动作的是执行型。" + hint,
        "历史读法: 每条专家消息的 name 就是它的身份；一条失败记录（正文含「执行失败」）"
        "同样代表上一位专家。name 为 approval 的消息是审批结果通知，不是专家产出。"
        "你看不到自己写过的路由理由，只能从专家产出的内容和顺序判断进度。",
        "规则（按顺序判断，先命中先执行）:",
        "1. 一次只选一个对象，或 FINISH；不要一次派多个。",
        "2. FINISH 的判据（满足一条即可）: "
        "(a) 最近一条专家消息是可交付正文——能原样作为答复交给用户，且已覆盖任务里的每个问句；"
        "(b) 审查型专家明确给出通过结论（默认团队写作「审查通过」）。"
        "要点清单、未知项清单、返工分派单、内部待办都不算可交付正文：遇到这类消息，"
        "先派产出型专家写成正文，再 FINISH。任务本身只要求要点或清单时，"
        "结构化的要点可以算正文，但仍须覆盖每个问句。"
        "不要为了「再确认一下」「再补一个角度」而多派一轮。",
        "3. 不连任: 若最后一条专家消息的 name 与这次要选的对象相同，且那条不是失败记录，"
        "改选其他专家或 FINISH；同一专家不得连续出现两次。",
        "4. 按需选人，不必把所有人走一遍: 纯写作或改写交给产出型；"
        "需要读资料、补事实或做测算才派对应的调研型，同一缺口只派一次；"
        "只有任务明确要求审查，或产出是高危交付物时才派审查型。"
        "已有可交付正文能让任务收尾时直接 FINISH（清单类产出不算，见规则 2）。"
        "若职责表里根本没有产出型，退化为：选最合适的一位产出后 FINISH。",
        "5. 审查型专家驳回后，下一位不能还是它: 必须是产出型，"
        "或能补上所指缺口的调研型；拿到新的产出之后，才允许再次派它。",
        "6. 返工有上限: 历史里不含通过结论的审查消息已有两条，就停止返工；"
        "若最近一条不是可交付正文，先派产出型按现有材料写正文，再 FINISH，"
        "把未解决项写进 reason。",
        "7. 专家报错（消息含「执行失败」）时最多换人重试一次；"
        "若已有可交付正文，直接 FINISH。",
        "8. reason 只写一句、不超过 30 字，格式「已有什么/还缺什么 → 派它做什么」；"
        "不复述任务、不列多步计划、不评价文风。",
    ]
    return SystemMessage(content="\n".join(rules))


def _role_of(registry: AgentRegistry, name: str | None) -> str | None:
    if not name:
        return None
    try:
        return registry.get(name).role
    except KeyError:
        return None


def _last_expert_message(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "name", None) not in {
            SUPERVISOR,
            "approval",
        }:
            return message
    return None


def _count_rejections(messages: list[Any], registry: AgentRegistry) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage)
        and _role_of(registry, getattr(message, "name", None)) == "reviewer"
        and _is_rejection(message)
    )


def _first_role(registry: AgentRegistry, agent_names: list[str], role: str) -> str | None:
    for name in agent_names:
        if _role_of(registry, name) == role:
            return name
    return None


def _routing_guard(
    nxt: str,
    state: Mapping[str, Any],
    registry: AgentRegistry,
    agent_names: list[str],
    max_rejections: int,
) -> tuple[str, str]:
    """确定性护栏：不连任、驳回上限。返回（新目标, 说明）。"""
    if nxt == FINISH:
        return nxt, ""
    messages = list(state.get("messages") or [])

    if _role_of(registry, nxt) == "reviewer":
        rejections = _count_rejections(messages, registry)
        if rejections >= max_rejections:
            return FINISH, f"审查已驳回 {rejections} 次，停止返工"

    last = _last_expert_message(messages)
    last_name = getattr(last, "name", None) if last is not None else None
    if last is None or nxt != last_name:
        return nxt, ""
    if "执行失败" in message_text(last):
        return nxt, ""  # 失败后允许同一专家重试一次
    if _role_of(registry, last_name) == "researcher":
        writer = _first_role(registry, agent_names, "writer")
        if writer is not None:
            return writer, "避免连任，改由 writer 成稿"
    return FINISH, "避免连任同一专家"


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


def _parse_first_json(text: str) -> dict[str, Any] | None:
    """从模型输出里取第一个 JSON 对象（容忍代码围栏与前后废话）。"""
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _decide_json(
    llm: Any,
    schema: type[BaseModel],
    prompt: SystemMessage,
    history: list[Any],
    agent_names: list[str],
) -> BaseModel:
    allowed = list(agent_names) + [FINISH]
    allowed_text = ", ".join(allowed)
    ask = HumanMessage(
        content=(
            "只输出一个 JSON 对象，不要解释、不要代码块。\n"
            f"next_agent 只能取以下值之一：{allowed_text}\n"
            '示例：{"next_agent": "FINISH", "reason": "已有可交付正文"}'
        )
    )
    text = message_text(llm.invoke([prompt, *history, ask]))
    data = _parse_first_json(text)
    if data is None:
        raise ValueError(f"无法从模型输出解析路由: {text[:200]!r}")
    try:
        return schema(**data)
    except Exception as exc:  # 非法值：带上合法清单重试一次
        retry = HumanMessage(
            content=(
                f"上一次输出不合法（{exc}）。next_agent 只能是：{allowed_text}。"
                "只输出 JSON 对象。"
            )
        )
        retried = message_text(
            llm.invoke([prompt, *history, ask, AIMessage(content=text), retry])
        )
        data = _parse_first_json(retried)
        if data is None:
            raise ValueError(f"无法从模型输出解析路由: {retried[:200]!r}") from exc
        return schema(**data)


_SAVER_CACHE: dict[str, Any] = {}
_SAVER_LOCK = threading.Lock()


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
    # 每个数据库路径一个连接：SqliteSaver 已用 check_same_thread=False + 内部锁保证线程安全，
    # 按线程拆分只会让连接数随线程数增长且永不关闭。建连必须串行，否则并发 PRAGMA 会互相锁库。
    key = str(db.resolve())
    with _SAVER_LOCK:
        if key not in _SAVER_CACHE:
            connection = sqlite3.connect(key, check_same_thread=False, timeout=30)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                # 别的进程正在切换日志模式：WAL 是持久设置，本轮跳过即可
                log.warning("设置 WAL 失败（%s）: %s", key, exc)
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
    worker_specs: dict[str, Any] = {}
    approval_targets = {
        name: name for name in agent_names
    }  # 默认直连；需要审批的专家在下面改写

    for name in agent_names:
        spec = registry.get(name)
        worker_specs[name] = spec
        if spec.requires_approval:
            approval_targets[name] = f"approve_{name}"
        llm = make_llm(
            settings,
            model=spec.model,
            temperature=spec.temperature if spec.temperature is not None else 0,
        )
        tools = registry.build_tools(spec)
        if spec.approval_tools:
            available = {item.name for item in tools}
            missing = sorted(name for name in spec.approval_tools if name not in available)
            if missing:
                raise ValueError(
                    f"{name} 的 approval_tools 未命中工具: {missing}"
                    f"（可用: {sorted(available)}）"
                )
            guarded = set(spec.approval_tools)
            tools = [_guarded_tool(item) if item.name in guarded else item for item in tools]
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
                    "stop_reason": "max_turns",
                    "messages": [
                        AIMessage(
                            content=f"[supervisor] 达到最大调度次数 {settings.max_turns}，停止。",
                            name=SUPERVISOR,
                        )
                    ],
                },
            )

        history = _trim_for_model(_business_messages(state), supervisor_llm, settings)
        try:
            decision = _decide(supervisor_llm, route_schema, registry, agent_names, history)
        except Exception as exc:
            log.exception("supervisor 路由失败")
            return Command(
                goto=END,
                update={
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

        guarded, note = _routing_guard(
            nxt, state, registry, agent_names, settings.max_rejections
        )
        if guarded != nxt:
            log.info("routing guard: %s -> %s（%s）", nxt, guarded, note)
            reason = f"{reason}（{note}）" if reason else note
            nxt = guarded

        log.info("turn %d -> %s | %s", turn + 1, nxt, reason)
        return Command(
            goto=END if nxt == FINISH else approval_targets[nxt],
            update={
                "turn": turn + 1,
                "stop_reason": "finish" if nxt == FINISH else "",
                "messages": [
                    AIMessage(content=f"[supervisor] 下一步 -> {nxt}。{reason}", name=SUPERVISOR)
                ],
            },
        )

    def make_approval(name: str):
        """高风险专家执行前的人工审批节点。

        进入时调用 interrupt() 挂起；恢复时用 Command(resume={"approved": bool, "note": str})。
        """

        def node(state: TeamState) -> Command:
            pending = dict(state.get("pending_tool") or {})
            tool = str(pending.get("tool") or "")
            if tool:
                payload = {
                    "type": "tool_approval",
                    "agent": name,
                    "tool": tool,
                    "args": pending.get("args") or {},
                    "task": state.get("task", ""),
                    "preview": _last_output(state),
                    "reason": f"工具 {tool} 被标记为高风险，需要人工确认",
                }
            else:
                payload = {
                    "type": "approval",
                    "agent": name,
                    "task": state.get("task", ""),
                    "preview": _last_output(state),
                    "reason": f"{name} 被标记为高风险动作，需要人工确认",
                }
            decision = interrupt(payload)
            if isinstance(decision, bool):
                decision = {"approved": decision}
            approved = bool(isinstance(decision, dict) and decision.get("approved"))
            note = str(decision.get("note", "") if isinstance(decision, dict) else "")
            if approved and tool:
                log.info("工具审批通过: %s/%s %s", name, tool, note)
                approved_now = list(dict.fromkeys([*(state.get("approved_tools") or []), tool]))
                return Command(
                    goto=name,
                    update={
                        "approved_tools": approved_now,
                        "pending_tool": {},
                        "messages": [
                            AIMessage(
                                content=f"[approval] 已批准工具 {tool}。{note}".strip(),
                                name="approval",
                            )
                        ],
                    },
                )
            if approved:
                log.info("审批通过: %s %s", name, note)
                return Command(
                    goto=name,
                    update={
                        "pending_tool": {},
                        "messages": [
                            AIMessage(
                                content=f"[approval] 已批准 {name}。{note}".strip(),
                                name="approval",
                            )
                        ],
                    },
                )
            log.info("审批拒绝: %s %s", name, note)
            target = tool or name
            return Command(
                goto=SUPERVISOR,
                update={
                    "pending_tool": {},
                    "messages": [
                        AIMessage(
                            content=f"[approval] 已拒绝 {target}：{note or '未说明原因'}",
                            name="approval",
                        )
                    ],
                },
            )

        node.__name__ = f"approve_{name}_node"
        return node

    def make_worker(name: str):
        compiled = worker_graphs[name]

        def node(state: TeamState) -> Command:
            log.info("agent %s 开始", name)
            try:
                spec = worker_specs[name]
                budget = spec.max_steps or settings.worker_max_steps
                context = _trim_for_model(_business_messages(state), supervisor_llm, settings)
                approved = set(state.get("approved_tools") or [])
                token = _APPROVED_TOOLS.set(frozenset(approved))
                try:
                    result = compiled.invoke(
                        {"messages": context},
                        {"recursion_limit": budget},
                    )
                finally:
                    _APPROVED_TOOLS.reset(token)
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

            markers = _approval_markers(list(result.get("messages") or []))
            if markers:
                tool_name, tool_args = markers[0]
                log.info("工具需要审批: %s", tool_name)
                return Command(
                    goto=f"approve_{name}",
                    update={
                        "pending_tool": {
                            "agent": name,
                            "tool": tool_name,
                            "args": tool_args,
                        },
                        "messages": [
                            AIMessage(
                                content=f"[{name}] 工具 {tool_name} 需要人工审批，已暂停。",
                                name=name,
                            )
                        ],
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
        spec = worker_specs[name]
        if spec.requires_approval or spec.approval_tools:
            graph.add_node(f"approve_{name}", make_approval(name))
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
    pending: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)


EventCallback = Callable[[str, dict[str, Any]], None]
Approver = Callable[[dict[str, Any]], Any]

_APPROVAL_EVENT = "__interrupt__"


def run_config(settings: Settings, thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(2 * settings.max_turns + 5, 25),
    }


def _is_rejection(message: Any) -> bool:
    """判断一条消息是不是审查驳回，而不是可交付草稿。"""
    text = message_text(message).strip()
    if not text:
        return False
    head = text[:200]
    first = text.splitlines()[0]
    return (
        first.startswith(("# 审查意见", "审查意见"))
        or "审查结论" in head
        or "需返工" in text[-200:]
    )


def _final_answer(state: dict[str, Any]) -> str:
    messages = list(state.get("messages") or [])
    candidates = [
        message
        for message in messages
        if isinstance(message, AIMessage)
        and getattr(message, "name", None) not in {SUPERVISOR, "approval"}
    ]
    drafts = [message for message in candidates if not _is_rejection(message)]
    if drafts:
        answer = message_text(drafts[-1])
    elif candidates:
        answer = message_text(candidates[-1])
    elif messages:
        answer = message_text(messages[-1])
    else:
        answer = ""
    if state.get("stop_reason") == "max_turns":
        if not drafts and answer:
            answer = f"（本次未产出可交付草稿，以下为最后一个专家的输出）\n\n{answer}"
        answer = f"{answer}\n\n（达到最大调度次数上限，结果可能不完整）"
    return answer


def usage_from_state(state: dict[str, Any]) -> dict[str, int]:
    """汇总消息里的 token 用量（模型若返回 usage_metadata）。"""
    prompt = completion = 0
    for message in state.get("messages") or []:
        usage = getattr(message, "usage_metadata", None) or {}
        prompt += int(usage.get("input_tokens") or 0)
        completion += int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _pending_payload(update: Any) -> dict[str, Any] | None:
    items = update if isinstance(update, (list, tuple)) else [update]
    if not items:
        return None
    first = items[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


def run_task_full(
    task: str,
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    thread_id: str | None = None,
    on_event: EventCallback | None = None,
    app: Any | None = None,
    approver: Approver | None = None,
    auto_approve: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    _resume: Any | None = None,
) -> RunResult:
    settings = settings or Settings.from_env()
    app = app or build_graph(settings=settings, registry=registry)
    tid = thread_id or f"task-{uuid.uuid4().hex[:8]}"
    config = run_config(settings, tid)
    inputs = {
        "task": task,
        "turn": 0,
        "stop_reason": "",
        "pending_tool": {},
        "approved_tools": [],
        "messages": [HumanMessage(content=task)],
    }
    if _resume is None:
        log.info("启动任务 thread_id=%s: %s", tid, task[:80])
    else:
        log.info("恢复任务 thread_id=%s", tid)
    next_input: Any = _resume if _resume is not None else inputs

    def snapshot(stop_reason: str, pending: dict[str, Any] | None = None) -> RunResult:
        state = app.get_state(config).values or {}
        return RunResult(
            answer=_final_answer(state),
            thread_id=tid,
            turns=int(state.get("turn") or 0),
            messages=list(state.get("messages") or []),
            stop_reason=stop_reason,
            pending=pending,
            usage=usage_from_state(state),
        )

    while True:
        interrupted: Any = None
        for chunk in app.stream(next_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == _APPROVAL_EVENT:
                    interrupted = update
                    continue
                if on_event is not None:
                    on_event(node, update if isinstance(update, dict) else {"value": update})
            if should_cancel is not None and should_cancel():
                log.info("任务已取消 thread_id=%s（节点边界）", tid)
                return snapshot("canceled")
        if interrupted is None:
            break
        payload = _pending_payload(interrupted) or {}
        log.info("等待人工审批: %s", payload.get("agent"))
        if approver is not None:
            decision = approver(payload)
        elif auto_approve:
            decision = {"approved": True, "note": "auto-approve"}
        else:
            if should_cancel is not None and should_cancel():
                log.info("任务已取消 thread_id=%s（审批前）", tid)
                return snapshot("canceled")
            return snapshot("awaiting_approval", payload)
        if isinstance(decision, bool):
            decision = {"approved": decision}
        record_approval(settings=settings, thread_id=tid, payload=payload, decision=decision)
        next_input = Command(resume=decision)
    state = app.get_state(config).values or {}
    result_usage = usage_from_state(state)
    log.info("任务结束 thread_id=%s tokens=%s", tid, result_usage)
    return RunResult(
        answer=_final_answer(state),
        thread_id=tid,
        turns=int(state.get("turn") or 0),
        messages=list(state.get("messages") or []),
        stop_reason=str(state.get("stop_reason") or ""),
        usage=result_usage,
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


def pending_approval(app: Any, thread_id: str) -> dict[str, Any] | None:
    """读取某个 thread 上挂起的审批载荷；没有则返回 None。"""
    snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return None


def resume_task(
    thread_id: str,
    decision: Any,
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    app: Any | None = None,
    on_event: EventCallback | None = None,
    approver: Approver | None = None,
    auto_approve: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> RunResult:
    """从挂起的审批继续执行。

    decision 形如 {"approved": True, "note": "同意"}，也可直接传 bool。
    没有挂起审批时抛 ValueError（不会起幽灵 run）；决策照旧写审计。
    """
    if isinstance(decision, bool):
        decision = {"approved": decision}
    settings = settings or Settings.from_env()
    app = app or build_graph(settings=settings, registry=registry)
    payload = pending_approval(app, thread_id)
    if payload is None:
        raise ValueError(f"thread {thread_id} 上没有挂起的审批")
    record_approval(settings=settings, thread_id=thread_id, payload=payload, decision=decision)
    return run_task_full(
        "",
        settings=settings,
        registry=registry,
        thread_id=thread_id,
        on_event=on_event,
        app=app,
        approver=approver,
        auto_approve=auto_approve,
        should_cancel=should_cancel,
        _resume=Command(resume=decision),
    )


def run_parallel(
    tasks: list[str],
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    max_workers: int = 4,
    thread_prefix: str = "job",
    on_event: EventCallback | None = None,
    approver: Approver | None = None,
    auto_approve: bool = False,
) -> list[RunResult]:
    """并行跑多个**独立任务**（多 SKU / 多站点）。

    每个任务有自己的 thread_id 与检查点连接；注意这不是图内的 Send 扇出，
    supervisor 仍然是串行调度的，只是任务之间并行。
    """
    settings = settings or Settings.from_env()
    if not tasks:
        return []
    results: list[RunResult | None] = [None] * len(tasks)

    def work(index: int, task: str) -> tuple[int, RunResult]:
        result = run_task_full(
            task,
            settings=settings,
            registry=registry,
            thread_id=f"{thread_prefix}-{uuid.uuid4().hex[:6]}",
            on_event=on_event,
            approver=approver,
            auto_approve=auto_approve,
        )
        return index, result

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(work, i, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    return [item for item in results if item is not None]
