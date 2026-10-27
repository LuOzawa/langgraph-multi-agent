from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
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
