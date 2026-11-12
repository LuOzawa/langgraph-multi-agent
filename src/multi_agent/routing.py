"""Supervisor 的路由决策：候选 schema、提示词、确定性护栏与降级解析。

这里只做"下一棒交给谁"的判断，不碰图结构与执行。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from multi_agent.constants import FINISH, SUPERVISOR
from multi_agent.context_window import message_text
from multi_agent.registry import AgentRegistry
from multi_agent.state import TeamState

log = logging.getLogger("multi_agent")


def _business_messages(state: TeamState) -> list[Any]:
    """专家可见消息：过滤 supervisor 的路由记录，避免污染上下文。"""
    return [
        message
        for message in (state.get("messages") or [])
        if getattr(message, "name", None) != SUPERVISOR
    ]


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


def _last_output(state: TeamState) -> str:
    """给审批人看的“将要提交的内容”预览。"""
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, AIMessage) and getattr(message, "name", None) not in {
            SUPERVISOR,
            "approval",
        }:
            return message_text(message)[:1500]
    return ""


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
    """从模型输出里取第一个 JSON 对象（容忍代码围栏与前后废话）。

    模型经常在 JSON 前解释一句、句子里还带花括号（如「我这样理解 {这一步是路由}」），
    所以不能只看第一个 "{"：逐个起点尝试 raw_decode，解析失败就跳到下一个 "{"。
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


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