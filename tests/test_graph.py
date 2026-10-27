from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

import multi_agent.graph as G
from multi_agent.config import Settings
from multi_agent.registry import default_registry

TRACK: dict = {}


def _reset(decisions=None, fail=False):
    TRACK.clear()
    TRACK.update(
        decisions=list(decisions or []),
        worker_messages=[],
        fail=fail,
        n=0,
        structured_calls=0,
    )


class FakeChatModel(BaseChatModel):
    """supervisor 用 with_structured_output 出路由，专家用普通生成。"""

    role: str = "worker"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.role != "supervisor":
            TRACK["worker_messages"].append(list(messages))
            if TRACK["fail"]:
                raise RuntimeError("boom")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"{self.role}产出"))]
        )

    def bind_tools(self, tools, **kwargs):  # create_agent 会调用
        return self

    def with_structured_output(self, schema, **kwargs):
        def route(_input):
            TRACK["structured_calls"] += 1
            nxt = TRACK["decisions"].pop(0) if TRACK["decisions"] else "FINISH"
            return schema(next_agent=nxt, reason="测试路由")

        return RunnableLambda(route)


def _settings(**overrides) -> Settings:
    base = dict(
        api_key="test",
        base_url="http://example.invalid/v1",
        model="fake",
        supervisor_model="fake",
        max_turns=6,
        worker_max_steps=10,
        context_max_tokens=12000,
        checkpoint_db="memory",
        request_timeout=5,
        max_retries=0,
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    def fake_make_llm(settings, *, model=None, temperature=0):
        TRACK["n"] += 1
        return FakeChatModel(role="supervisor" if TRACK["n"] == 1 else "worker")

    monkeypatch.setattr(G, "make_llm", fake_make_llm)


def _run(decisions, *, thread_id="t1", app=None, fail=False, **settings_kwargs):
    _reset(decisions, fail=fail)
    settings = _settings(**settings_kwargs)
    app = app or G.build_graph(
        settings=settings, registry=default_registry(), checkpointer=InMemorySaver()
    )
    result = G.run_task_full(
        "写一份接入说明", settings=settings, registry=default_registry(),
        thread_id=thread_id, app=app,
    )
    return settings, app, result


def test_full_flow_routes_and_finishes():
    _, _, result = _run(["researcher", "writer", "FINISH"])
    assert result.turns == 3
    assert result.answer == "worker产出"


def test_workers_never_see_supervisor_messages():
    _run(["researcher", "writer", "FINISH"])
    assert TRACK["worker_messages"], "专家应当被调用"
    for messages in TRACK["worker_messages"]:
        assert all(getattr(m, "name", None) != "supervisor" for m in messages)


def test_worker_failure_is_contained():
    _, _, result = _run(["researcher", "FINISH"], fail=True)
    assert "执行失败" in result.answer


def test_max_turns_guard():
    _, _, result = _run(["researcher"] * 10, max_turns=2)
    assert result.turns == 2
    assert "达到最大调度次数" in result.answer


def test_invalid_route_degrades_gracefully():
    _, _, result = _run(["nonexistent"])
    assert "路由失败" in result.answer


def test_checkpointer_keeps_history_per_thread():
    settings = _settings()
    app = G.build_graph(
        settings=settings, registry=default_registry(), checkpointer=InMemorySaver()
    )
    _, _, first = _run(["researcher", "FINISH"], thread_id="p1", app=app)
    _, _, second = _run(["writer", "FINISH"], thread_id="p1", app=app)
    _, _, other = _run(["researcher", "FINISH"], thread_id="p2", app=app)
    assert len(second.messages) > len(first.messages)
    assert len(other.messages) < len(second.messages)


def test_per_agent_step_budget(monkeypatch):
    """每个专家可以用自己的 max_steps，未设置时跟随全局。"""
    recorded: list[tuple[str, int]] = []

    class RecordingWorker:
        def __init__(self, tag: str):
            self.tag = tag

        def invoke(self, inputs, config=None):
            recorded.append((self.tag, int((config or {}).get("recursion_limit", -1))))
            return {"messages": [AIMessage(content=f"{self.tag}产出")]}

    def fake_create_agent(llm, tools, system_prompt=None, **kwargs):
        tag = "a" if "AAA" in (system_prompt or "") else "b"
        return RecordingWorker(tag)

    monkeypatch.setattr(G, "_create_agent", fake_create_agent)
    _reset(["a", "b", "FINISH"])
    settings = _settings(worker_max_steps=7)

    from multi_agent.registry import AgentRegistry, AgentSpec

    registry = AgentRegistry()
    registry.register(AgentSpec(name="a", description="A", system_prompt="AAA", max_steps=3))
    registry.register(AgentSpec(name="b", description="B", system_prompt="BBB"))

    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())
    G.run_task_full("任务", settings=settings, registry=registry, thread_id="budget", app=app)

    assert recorded == [("a", 3), ("b", 7)]


# ---------------------------------------------------------------- 人工审批 / 上下文


def _approval_registry():
    from multi_agent.registry import AgentRegistry, AgentSpec

    registry = AgentRegistry()
    registry.register(AgentSpec(name="writer", description="写文案", system_prompt="WRITE"))
    registry.register(
        AgentSpec(
            name="publisher",
            description="上架商品（高风险）",
            system_prompt="PUBLISH",
            requires_approval=True,
        )
    )
    return registry


def _recording_agents(monkeypatch, called):
    def fake_create_agent(llm, tools, system_prompt=None, **kwargs):
        class Worker:
            def invoke(self, inputs, config=None):
                called.append(system_prompt)
                return {"messages": [AIMessage(content=f"{system_prompt}-产出")]}

        return Worker()

    monkeypatch.setattr(G, "_create_agent", fake_create_agent)


def test_approval_approved_runs_worker(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings()
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "把 SKU-1 上架",
        settings=settings,
        registry=registry,
        thread_id="ap-ok",
        app=app,
        approver=lambda payload: {"approved": True, "note": "同意"},
    )

    assert called == ["PUBLISH"]
    assert result.answer == "PUBLISH-产出"
    assert any("已批准 publisher" in G.message_text(m) for m in result.messages)


def test_approval_rejected_skips_worker(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings()
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "把 SKU-1 上架",
        settings=settings,
        registry=registry,
        thread_id="ap-no",
        app=app,
        approver=lambda payload: {"approved": False, "note": "价格未确认"},
    )

    assert called == []
    assert any("已拒绝 publisher" in G.message_text(m) for m in result.messages)
    assert result.turns == 2  # 路由一次 + 拒绝后 FINISH


def test_pending_approval_is_returned(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings()
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "把 SKU-1 上架", settings=settings, registry=registry, thread_id="ap-wait", app=app
    )

    assert result.stop_reason == "awaiting_approval"
    assert result.pending is not None and result.pending["agent"] == "publisher"
    assert called == []


def test_context_trimming_keeps_task():
    settings = _settings(context_max_tokens=80)
    messages = [HumanMessage(content="任务" * 40)] + [
        AIMessage(content="很长的输出" * 50, name="writer") for _ in range(8)
    ]
    trimmed = G._trim_for_model(messages, FakeChatModel(), settings)

    assert len(trimmed) < len(messages)
    assert trimmed[0].content.startswith("任务")
    assert G._count_tokens(FakeChatModel(), trimmed) <= 80 + 40  # 允许首条任务额外占用
