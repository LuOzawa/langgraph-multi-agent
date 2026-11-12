"""上下文预算：消息转文本、token 计数与历史裁剪。

裁剪的硬约束：tool_call 与它的 ToolMessage 必须成对保留/成对丢弃。
OpenAI 兼容接口要求 ToolMessage 紧跟产生它的 tool_call，
任何一侧被单独裁掉，模型端都会直接 400。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from multi_agent.config import Settings
from multi_agent.constants import SUPERVISOR

log = logging.getLogger("multi_agent")


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


def _count_tokens(llm: Any, messages: list[Any]) -> int:
    try:
        return int(llm.get_num_tokens_from_messages(messages))
    except Exception:  # 假模型/不支持计数的模型，用字符数估算
        return sum(len(message_text(m)) for m in messages) // 2 + 1


def _drop_orphan_tool_messages(messages: list[Any]) -> list[Any]:
    """修掉裁剪留下的"半截"工具调用。

    两个方向都会让 OpenAI 兼容接口直接 400，必须一起处理：
    1. 有 ToolMessage 却找不到前驱 AIMessage(tool_calls) → 丢掉这条 ToolMessage；
    2. 有 AIMessage(tool_calls) 但响应被裁掉 → 从消息里摘掉这些 tool_calls。
    """
    kept: list[Any] = []
    open_calls: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            open_calls = {
                str(call.get("id"))
                for call in (getattr(message, "tool_calls", None) or [])
                if call.get("id") is not None
            }
            kept.append(message)
        elif isinstance(message, ToolMessage):
            call_id = str(getattr(message, "tool_call_id", "") or "")
            if call_id and call_id in open_calls:
                open_calls.discard(call_id)
                kept.append(message)
            # 否则是孤儿响应，丢弃
        else:
            open_calls = set()
            kept.append(message)

    responded = {
        str(message.tool_call_id)
        for message in kept
        if isinstance(message, ToolMessage) and getattr(message, "tool_call_id", None)
    }

    result: list[Any] = []
    for message in kept:
        if isinstance(message, AIMessage):
            calls = list(getattr(message, "tool_calls", None) or [])
            if calls:
                surviving = [call for call in calls if str(call.get("id")) in responded]
                if len(surviving) != len(calls):
                    message = message.model_copy(update={"tool_calls": surviving})
        result.append(message)
    return result


def _shrink_window(messages: list[Any], llm: Any, budget: int) -> list[Any]:
    """从尾部保留消息，直到不超过预算；至少留一条。"""
    window = list(messages)
    while len(window) > 1 and _count_tokens(llm, window) > budget:
        window.pop(0)
    return window


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
        trimmed = _shrink_window(messages[-max(2, budget // 200):], llm, budget)

    # 无论走哪条路径，裁剪都可能把 tool_call 和它的响应拆散，这里统一收尾
    trimmed = _drop_orphan_tool_messages(trimmed)

    first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
    if first_human is not None and all(m is not first_human for m in trimmed):
        trimmed = [
            first_human,
            AIMessage(content="（为控制上下文长度，已省略部分较早消息）", name=SUPERVISOR),
            *trimmed,
        ]
    log.info("上下文裁剪: %d -> %d 条（预算 %d tokens）", len(messages), len(trimmed), budget)
    return trimmed