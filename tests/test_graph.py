from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
        audit_log="off",
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


def test_pending_then_resume(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings()
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    first = G.run_task_full(
        "把 SKU-2 上架", settings=settings, registry=registry, thread_id="ap-resume", app=app
    )
    assert first.stop_reason == "awaiting_approval"

    payload = G.pending_approval(app, "ap-resume")
    assert payload is not None and payload["agent"] == "publisher"

    second = G.resume_task(
        "ap-resume",
        {"approved": True, "note": "复核通过"},
        settings=settings,
        registry=registry,
        app=app,
    )
    assert called == ["PUBLISH"]
    assert second.answer == "PUBLISH-产出"


def test_audit_log_written(monkeypatch, tmp_path):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    audit_file = tmp_path / "approvals.jsonl"
    settings = _settings(audit_log=str(audit_file))
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    G.run_task_full(
        "把 SKU-3 上架",
        settings=settings,
        registry=registry,
        thread_id="ap-audit",
        app=app,
        approver=lambda payload: {"approved": False, "note": "合规未通过"},
    )

    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["agent"] == "publisher"
    assert record["approved"] is False
    assert record["note"] == "合规未通过"
    assert record["thread_id"] == "ap-audit"


def test_parallel_runs_independent_tasks():
    _reset(["writer", "FINISH", "writer", "FINISH"])
    settings = _settings()
    registry = default_registry()
    results = G.run_parallel(
        ["任务一", "任务二"], settings=settings, registry=registry, max_workers=1
    )
    assert len(results) == 2
    assert len({r.thread_id for r in results}) == 2


# ---------------------------------------------------------------- 工具级审批


def test_guarded_tool_blocks_until_approved():
    from multi_agent.tools.ecommerce import calc_landed_cost

    guarded = G._guarded_tool(calc_landed_cost)
    args = {"fob_usd": 10, "freight_usd": 2, "duty_rate": 0.1, "platform_fee_rate": 0.15}

    blocked = guarded.invoke(args)
    assert blocked.startswith("REQUIRES_APPROVAL::calc_landed_cost::")

    token = G._APPROVED_TOOLS.set(frozenset({"calc_landed_cost"}))
    try:
        allowed = guarded.invoke(args)
    finally:
        G._APPROVED_TOOLS.reset(token)
    assert "到岸成本 $13.00" in allowed


def _tool_worker(monkeypatch, calls):
    class ToolWorker:
        def invoke(self, inputs, config=None):
            approved = set(G._APPROVED_TOOLS.get())
            calls.append(sorted(approved))
            if "calc_landed_cost" not in approved:
                return {
                    "messages": [
                        ToolMessage(
                            content='REQUIRES_APPROVAL::calc_landed_cost::{"fob_usd": 10}',
                            tool_call_id="t1",
                        ),
                        AIMessage(content="需要审批"),
                    ]
                }
            return {"messages": [AIMessage(content="已计算 13.00")]}

    monkeypatch.setattr(
        G, "_create_agent", lambda llm, tools, system_prompt=None, **kwargs: ToolWorker()
    )


def _pricing_registry():
    from multi_agent.registry import AgentRegistry, AgentSpec
    from multi_agent.tools.ecommerce import calc_landed_cost

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="pricing",
            description="定价",
            system_prompt="PRICING",
            tools_factory=lambda: [calc_landed_cost],
            approval_tools=("calc_landed_cost",),
        )
    )
    return registry


def test_tool_approval_approved_reruns_worker(monkeypatch):
    calls: list[list[str]] = []
    _tool_worker(monkeypatch, calls)
    _reset(["pricing", "FINISH"])
    settings = _settings()
    registry = _pricing_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "算一下这个 SKU 的售价",
        settings=settings,
        registry=registry,
        thread_id="tool-ok",
        app=app,
        approver=lambda payload: {"approved": True, "note": "同意"},
    )

    assert result.answer == "已计算 13.00"
    assert len(calls) == 2
    assert calls[0] == []
    assert calls[1] == ["calc_landed_cost"]


def test_tool_approval_rejected_stops(monkeypatch):
    calls: list[list[str]] = []
    _tool_worker(monkeypatch, calls)
    _reset(["pricing", "FINISH"])
    settings = _settings()
    registry = _pricing_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "算一下这个 SKU 的售价",
        settings=settings,
        registry=registry,
        thread_id="tool-no",
        app=app,
        approver=lambda payload: {"approved": False, "note": "价格未确认"},
    )

    assert len(calls) == 1
    assert any("已拒绝 calc_landed_cost" in G.message_text(m) for m in result.messages)


def test_sqlite_saver_resumes_from_another_thread(monkeypatch, tmp_path):
    """API 层把 app 留在任务表里，恢复发生在线程池的另一个线程上。"""
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings(checkpoint_db=str(tmp_path / "ckpt.sqlite"))
    registry = _approval_registry()
    box: dict = {}

    def start() -> None:
        box["app"] = G.build_graph(settings=settings, registry=registry)
        box["result"] = G.run_task_full(
            "把 SKU-1 上架",
            settings=settings,
            registry=registry,
            thread_id="x-thread",
            app=box["app"],
        )

    worker = threading.Thread(target=start)
    worker.start()
    worker.join(30)
    assert not worker.is_alive()
    assert box["result"].stop_reason == "awaiting_approval"

    resumed = G.resume_task(
        "x-thread",
        {"approved": True, "note": "同意"},
        settings=settings,
        registry=registry,
        app=box["app"],
    )
    assert resumed.stop_reason == "finish"
    assert resumed.answer == "PUBLISH-产出"
    assert called == ["PUBLISH"]


def test_resume_task_writes_audit_and_requires_pending(monkeypatch, tmp_path):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    audit = tmp_path / "approvals.jsonl"
    settings = _settings(audit_log=str(audit))
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    pending = G.run_task_full(
        "把 SKU-1 上架", settings=settings, registry=registry, thread_id="r-audit", app=app
    )
    assert pending.stop_reason == "awaiting_approval"

    result = G.resume_task(
        "r-audit",
        {"approved": True, "note": "同意"},
        settings=settings,
        registry=registry,
        app=app,
    )
    assert result.answer == "PUBLISH-产出"
    rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["approved"] is True
    assert rows[0]["note"] == "同意"

    with pytest.raises(ValueError, match="没有挂起的审批"):
        G.resume_task("r-audit", {"approved": True}, settings=settings, registry=registry, app=app)


def test_approver_returning_bool_is_accepted(monkeypatch, tmp_path):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    audit = tmp_path / "approvals.jsonl"
    settings = _settings(audit_log=str(audit))
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "把 SKU-1 上架",
        settings=settings,
        registry=registry,
        thread_id="bool-ap",
        app=app,
        approver=lambda payload: True,
    )

    assert result.answer == "PUBLISH-产出"
    rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["approved"] is True


def test_should_cancel_stops_between_nodes():
    _reset(["writer", "writer", "FINISH"])
    settings = _settings()
    registry = default_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())
    seen: list[str] = []

    result = G.run_task_full(
        "写两份说明",
        settings=settings,
        registry=registry,
        thread_id="cancel-between",
        app=app,
        on_event=lambda node, update: seen.append(node),
        should_cancel=lambda: len(seen) >= 1,
    )

    assert result.stop_reason == "canceled"
    assert seen == ["supervisor"]  # 首个节点后即停，专家未执行


def test_resume_task_honors_cancel(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["publisher", "FINISH"])
    settings = _settings()
    registry = _approval_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    pending = G.run_task_full(
        "把 SKU-1 上架", settings=settings, registry=registry, thread_id="cancel-resume", app=app
    )
    assert pending.stop_reason == "awaiting_approval"

    resumed = G.resume_task(
        "cancel-resume",
        {"approved": True, "note": "同意"},
        settings=settings,
        registry=registry,
        app=app,
        should_cancel=lambda: True,
    )

    assert resumed.stop_reason == "canceled"
    assert called == []  # 取消发生在专家执行前


def test_make_checkpointer_is_thread_safe(tmp_path):
    path = str(tmp_path / "concurrent.sqlite")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        savers = list(pool.map(lambda _: G.make_checkpointer(path), range(8)))

    assert len({id(saver) for saver in savers}) == 1  # 同一个实例
    assert len([key for key in G._SAVER_CACHE if key == str(Path(path).resolve())]) == 1


def test_checkpointer_reused_per_database_path(tmp_path):
    path = str(tmp_path / "ckpt.sqlite")
    first = G.make_checkpointer(path)
    from_other_thread: dict = {}

    thread = threading.Thread(target=lambda: from_other_thread.setdefault("saver", G.make_checkpointer(path)))
    thread.start()
    thread.join(10)
    assert not thread.is_alive()

    assert from_other_thread["saver"] is first
    keys = [key for key in G._SAVER_CACHE if key == str(Path(path).resolve())]
    assert len(keys) == 1
    assert G.make_checkpointer("memory") is not G.make_checkpointer("memory")


def test_unknown_approval_tool_rejected():
    from multi_agent.registry import AgentRegistry, AgentSpec

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="pricing",
            description="定价",
            system_prompt="PRICING",
            tools_factory=lambda: (),
            approval_tools=("no_such_tool",),
        )
    )
    with pytest.raises(ValueError, match="approval_tools 未命中工具"):
        G.build_graph(settings=_settings(), registry=registry, checkpointer=InMemorySaver())


def test_final_answer_prefers_draft_over_review_rejection():
    messages = [
        HumanMessage(content="新增专家的改动点？"),
        AIMessage(content="第一段结论：需要改 registry 与 defaults。\n\n1. 注册 AgentSpec\n2. 提供 system_prompt", name="writer"),
        AIMessage(content="# 审查意见\n结论：需返工\n1. 第 2 点缺示例 → 补一行\n需返工", name="reviewer"),
    ]
    answer = G._final_answer({"messages": messages, "stop_reason": "finish"})
    assert answer.startswith("第一段结论")
    assert "审查意见" not in answer


def test_final_answer_keeps_passing_review_with_draft():
    passed = "# 最终定稿\n注册并填 system_prompt 即可。\n审查通过"
    messages = [
        HumanMessage(content="怎么加专家？"),
        AIMessage(content="草稿", name="writer"),
        AIMessage(content=passed, name="reviewer"),
    ]
    assert G._final_answer({"messages": messages, "stop_reason": "finish"}) == passed


def test_final_answer_annotates_missing_draft_on_max_turns():
    messages = [
        HumanMessage(content="任务"),
        AIMessage(content="# 审查意见\n需返工", name="reviewer"),
    ]
    answer = G._final_answer({"messages": messages, "stop_reason": "max_turns"})
    assert "未产出可交付草稿" in answer
    assert "达到最大调度次数上限" in answer


def test_decide_json_retries_invalid_agent():
    class Stub:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages[-1].content)
            if len(self.calls) == 1:
                return AIMessage(content='{"next_agent": "nonexistent", "reason": "猜的"}')
            return AIMessage(content='{"next_agent": "FINISH", "reason": "已有正文"}')

    stub = Stub()
    schema = G._route_schema(["writer"])
    decision = G._decide_json(
        stub, schema, SystemMessage(content="supervisor"), [], ["writer"]
    )
    assert decision.next_agent == "FINISH"
    assert len(stub.calls) == 2
    assert "只能是" in stub.calls[1]


def _guard_registry():
    from multi_agent.registry import AgentRegistry, AgentSpec

    registry = AgentRegistry()
    registry.register(
        AgentSpec(name="researcher", description="调研", system_prompt="R", role="researcher")
    )
    registry.register(
        AgentSpec(name="writer", description="成稿", system_prompt="W", role="writer")
    )
    registry.register(
        AgentSpec(name="reviewer", description="审查", system_prompt="C", role="reviewer")
    )
    return registry


def test_routing_guard_prevents_consecutive_same_agent():
    registry = _guard_registry()
    names = registry.names()

    state = {"messages": [HumanMessage(content="任务"), AIMessage(content="结论", name="researcher")]}
    nxt, note = G._routing_guard("researcher", state, registry, names, 1)
    assert nxt == "writer"
    assert "连任" in note

    nxt, note = G._routing_guard(
        "writer", {"messages": [AIMessage(content="草稿", name="writer")]}, registry, names, 1
    )
    assert nxt == G.FINISH
    assert "连任" in note

    nxt, note = G._routing_guard(
        "writer",
        {"messages": [AIMessage(content="[writer] 执行失败：boom", name="writer")]},
        registry,
        names,
        1,
    )
    assert nxt == "writer"
    assert note == ""


def test_routing_guard_caps_review_rejections():
    registry = _guard_registry()
    names = registry.names()
    state = {
        "messages": [
            AIMessage(content="草稿", name="writer"),
            AIMessage(content="# 审查意见\n结论：需返工", name="reviewer"),
            AIMessage(content="修订稿", name="writer"),
        ]
    }

    assert G._routing_guard("reviewer", state, registry, names, 1)[0] == G.FINISH
    assert G._routing_guard("reviewer", state, registry, names, 2)[0] == "reviewer"
    assert G._routing_guard(G.FINISH, state, registry, names, 1) == (G.FINISH, "")


def test_routing_guard_end_to_end_switches_to_writer(monkeypatch):
    called: list[str] = []
    _recording_agents(monkeypatch, called)
    _reset(["researcher", "researcher", "FINISH"])
    settings = _settings()
    registry = default_registry()
    app = G.build_graph(settings=settings, registry=registry, checkpointer=InMemorySaver())

    result = G.run_task_full(
        "说明一下", settings=settings, registry=registry, thread_id="guard-1", app=app
    )

    assert result.stop_reason == "finish"
    assert len(called) == 2
    assert called[0].startswith("你是研究员")
    assert called[1].startswith("你是写手")
