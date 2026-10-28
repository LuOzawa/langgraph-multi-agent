from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import multi_agent.api.service as service
import multi_agent.graph as graph_module
from multi_agent.api.app import create_app
from multi_agent.api.service import LangGraphExecutor, StartOutcome
from multi_agent.config import Settings
from multi_agent.graph import RunResult
from multi_agent.registry import AgentRegistry, AgentSpec

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _settings(**overrides) -> Settings:
    base: dict[str, Any] = dict(
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


def _completed(answer: str = "完成", *, turns: int = 2) -> RunResult:
    return RunResult(
        answer=answer,
        thread_id="ignored",
        turns=turns,
        stop_reason="finish",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )


def _awaiting(agent: str = "publisher") -> RunResult:
    return RunResult(
        answer="等待审批",
        thread_id="ignored",
        turns=1,
        stop_reason="awaiting_approval",
        pending={"type": "approval", "agent": agent, "reason": "需要人工确认", "task": "上架新品"},
        usage={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    )


class FakeExecutor:
    """按需返回固定结果；记录调用参数，供断言恢复链路的上下文。"""

    def __init__(self, start_result=None, resume_result=None) -> None:
        self._start_result = start_result or (lambda task, thread_id: _completed())
        self._resume_result = resume_result or (lambda thread_id, decision, app: _completed())
        self.starts: list[tuple[str, str]] = []
        self.resumes: list[tuple[str, dict[str, Any], Any]] = []
        self.app = object()

    def start(self, task: str, thread_id: str) -> StartOutcome:
        self.starts.append((task, thread_id))
        result = self._start_result(task, thread_id)
        if isinstance(result, BaseException):
            raise result
        return StartOutcome(result=result, app=self.app)

    def resume(self, thread_id: str, decision: dict[str, Any], app: Any) -> RunResult:
        self.resumes.append((thread_id, decision, app))
        return self._resume_result(thread_id, decision, app)


class BlockingExecutor:
    """start 阻塞到测试放行，用来模拟长任务、取消与超时。"""

    def __init__(self, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.fail = fail
        self.app = object()

    def start(self, task: str, thread_id: str) -> StartOutcome:
        self.started.set()
        self.release.wait(5)
        self.finished.set()
        if self.fail:
            raise RuntimeError("迟到失败")
        return StartOutcome(result=_completed(), app=self.app)

    def resume(self, thread_id: str, decision: dict[str, Any], app: Any) -> RunResult:
        return _completed()


def _client(executor: Any, settings: Settings | None = None, **kwargs) -> TestClient:
    kwargs.setdefault("max_workers", 2)
    app = create_app(settings=settings or _settings(), executor=executor, **kwargs)
    return TestClient(app)


def _wait(client: TestClient, thread_id: str, status: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/tasks/{thread_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] == status:
            return last
        time.sleep(0.01)
    raise AssertionError(f"task {thread_id} 未进入 {status}，最后响应: {last}")


def test_submit_and_get_completed():
    executor = FakeExecutor()
    with _client(executor) as client:
        response = client.post("/tasks", json={"task": "写周报", "thread_id": "t-1"})
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "queued"
        assert body["thread_id"] == "t-1"

        final = _wait(client, "t-1", "completed")
        assert final["answer"] == "完成"
        assert final["turns"] == 2
        assert final["usage"]["total_tokens"] == 30
        assert final["task"] == "写周报"
        assert final["error"] is None

        assert executor.starts == [("写周报", "t-1")]


def test_generated_thread_id():
    with _client(FakeExecutor()) as client:
        response = client.post("/tasks", json={"task": "生成 ID"})
        assert response.status_code == 202, response.text
        thread_id = response.json()["thread_id"]
        assert thread_id.startswith("api-")
        assert THREAD_ID_PATTERN.match(thread_id)


def test_submit_approval_flow():
    executor = FakeExecutor(
        start_result=lambda task, thread_id: _awaiting("publisher"),
        resume_result=lambda thread_id, decision, app: _completed("已上架"),
    )
    with _client(executor) as client:
        response = client.post("/tasks", json={"task": "上架新品", "thread_id": "t-approve"})
        assert response.status_code == 202

        pending = _wait(client, "t-approve", "awaiting_approval")
        assert pending["pending"]["agent"] == "publisher"

        approvals = client.get("/approvals").json()
        assert len(approvals) == 1
        assert approvals[0]["thread_id"] == "t-approve"
        assert client.get("/approvals/t-approve").json()["pending"]["agent"] == "publisher"

        response = client.post("/approvals/t-approve", json={"approved": True, "note": "同意"})
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "running"

        final = _wait(client, "t-approve", "completed")
        assert final["answer"] == "已上架"

        assert len(executor.resumes) == 1
        thread_id, decision, app = executor.resumes[0]
        assert thread_id == "t-approve"
        assert decision == {"approved": True, "note": "同意"}
        assert app is executor.app

        assert client.get("/approvals").json() == []


def test_reject_approval():
    executor = FakeExecutor(
        start_result=lambda task, thread_id: _awaiting(),
        resume_result=lambda thread_id, decision, app: _completed("已拒绝，未执行"),
    )
    with _client(executor) as client:
        client.post("/tasks", json={"task": "上架新品", "thread_id": "t-reject"})
        _wait(client, "t-reject", "awaiting_approval")

        response = client.post(
            "/approvals/t-reject", json={"approved": False, "note": "价格未确认"}
        )
        assert response.status_code == 202, response.text
        final = _wait(client, "t-reject", "completed")
        assert final["answer"] == "已拒绝，未执行"

        assert executor.resumes[0][1] == {"approved": False, "note": "价格未确认"}


def test_real_graph_reject_writes_audit(monkeypatch, tmp_path):
    """拒绝也要留痕：审计由 resume 路径写入，走真实图验证。"""

    _patch_graph(monkeypatch, ["publisher", "FINISH"])
    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="publisher",
            description="上架商品（高风险）",
            system_prompt="PUBLISH",
            requires_approval=True,
        )
    )
    audit = tmp_path / "audit.jsonl"
    settings = _settings(audit_log=str(audit), checkpoint_db=str(tmp_path / "c.sqlite"))

    app = create_app(settings=settings, registry=registry, max_workers=2)
    with TestClient(app) as client:
        client.post("/tasks", json={"task": "把 SKU-2 上架", "thread_id": "it-reject"})
        _wait(client, "it-reject", "awaiting_approval", timeout=5.0)
        response = client.post(
            "/approvals/it-reject", json={"approved": False, "note": "价格未确认"}
        )
        assert response.status_code == 202, response.text
        _wait(client, "it-reject", "completed", timeout=5.0)

    rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "it-reject"
    assert rows[0]["agent"] == "publisher"
    assert rows[0]["approved"] is False
    assert rows[0]["note"] == "价格未确认"


def test_duplicate_thread_id_409():
    with _client(FakeExecutor()) as client:
        assert client.post("/tasks", json={"task": "一", "thread_id": "dup"}).status_code == 202
        response = client.post("/tasks", json={"task": "二", "thread_id": "dup"})
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "task already exists"


def test_unknown_task_404():
    with _client(FakeExecutor()) as client:
        assert client.get("/tasks/nope").status_code == 404
        assert client.get("/approvals/nope").status_code == 404


def test_approval_when_not_awaiting_409():
    executor = FakeExecutor()
    with _client(executor) as client:
        client.post("/tasks", json={"task": "普通任务", "thread_id": "t-done"})
        _wait(client, "t-done", "completed")

        response = client.post("/approvals/t-done", json={"approved": True})
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "task is not awaiting approval"
        assert executor.resumes == []


def test_failed_task_reports_error():
    def boom(task: str, thread_id: str) -> RunResult:
        raise RuntimeError("boom")

    with _client(FakeExecutor(start_result=boom)) as client:
        client.post("/tasks", json={"task": "会失败", "thread_id": "t-fail"})
        final = _wait(client, "t-fail", "failed")
        assert "boom" in final["error"]

        response = client.get("/tasks/t-fail")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"


def test_list_tasks_filter():
    executor = FakeExecutor(
        start_result=lambda task, thread_id: (
            _awaiting() if task == "需审批" else _completed()
        )
    )
    with _client(executor) as client:
        client.post("/tasks", json={"task": "普通", "thread_id": "t-a"})
        client.post("/tasks", json={"task": "需审批", "thread_id": "t-b"})
        _wait(client, "t-a", "completed")
        _wait(client, "t-b", "awaiting_approval")

        everything = client.get("/tasks").json()
        assert [item["thread_id"] for item in everything] == ["t-a", "t-b"]

        awaiting = client.get("/tasks", params={"status": "awaiting_approval"}).json()
        assert [item["thread_id"] for item in awaiting] == ["t-b"]

        assert len(client.get("/approvals").json()) == 1


def test_validation():
    with _client(FakeExecutor()) as client:
        assert client.post("/tasks", json={"task": ""}).status_code == 422
        assert client.post("/tasks", json={"task": "   "}).status_code == 422
        assert (
            client.post("/tasks", json={"task": "ok", "thread_id": "bad id"}).status_code == 422
        )


def test_langgraph_executor_wiring(monkeypatch):
    settings = _settings()
    sentinel_app = object()
    calls: dict[str, Any] = {}

    def fake_run_task_full(task, **kwargs):
        calls["start"] = {"task": task, **kwargs}
        return _completed()

    def fake_resume_task(thread_id, decision, **kwargs):
        calls["resume"] = {"thread_id": thread_id, "decision": decision, **kwargs}
        return _completed()

    monkeypatch.setattr(service, "run_task_full", fake_run_task_full)
    monkeypatch.setattr(service, "resume_task", fake_resume_task)

    factory_calls: list[int] = []

    def app_factory():
        factory_calls.append(1)
        return sentinel_app

    registry = object()
    executor = LangGraphExecutor(settings, registry, app_factory=app_factory)  # type: ignore[arg-type]

    outcome = executor.start("任务", "t-wire")
    assert outcome.app is sentinel_app
    assert factory_calls == [1]

    started = calls["start"]
    assert started["task"] == "任务"
    assert started["settings"] is settings
    assert started["registry"] is registry
    assert started["thread_id"] == "t-wire"
    assert started["app"] is sentinel_app
    assert "approver" not in started
    assert "auto_approve" not in started

    executor.resume("t-wire", {"approved": True}, outcome.app)
    resumed = calls["resume"]
    assert resumed["thread_id"] == "t-wire"
    assert resumed["decision"] == {"approved": True}
    assert resumed["settings"] is settings
    assert resumed["registry"] is registry
    assert resumed["app"] is sentinel_app
    assert factory_calls == [1], "resume 不应重建图"


def test_error_stop_reason_reports_answer_as_error():
    def failed_result(task: str, thread_id: str) -> RunResult:
        return RunResult(
            answer="[supervisor] 路由失败，任务终止：OpenAIConnectionError: Connection error.",
            thread_id=thread_id,
            turns=0,
            stop_reason="error",
        )

    with _client(FakeExecutor(start_result=failed_result)) as client:
        client.post("/tasks", json={"task": "网络挂掉", "thread_id": "t-err"})
        final = _wait(client, "t-err", "failed")
        assert final["stop_reason"] == "error"
        assert final["error"].startswith("[supervisor] 路由失败")
        assert final["pending"] is None


def test_awaiting_approval_without_pending_fails():
    def broken_interrupt(task: str, thread_id: str) -> RunResult:
        return RunResult(
            answer="等待审批",
            thread_id=thread_id,
            turns=1,
            stop_reason="awaiting_approval",
            pending=None,
        )

    with _client(FakeExecutor(start_result=broken_interrupt)) as client:
        client.post("/tasks", json={"task": "缺载荷", "thread_id": "t-nopending"})
        final = _wait(client, "t-nopending", "failed")
        assert final["error"] == "awaiting_approval without pending payload"
        assert final["pending"] is None
        assert client.get("/approvals").json() == []


def test_shutdown_returns_503():
    executor = FakeExecutor()
    app = create_app(settings=_settings(), executor=executor, max_workers=2)
    app.state.manager.shutdown()
    with TestClient(app) as client:
        response = client.post("/tasks", json={"task": "关闭后提交"})
        assert response.status_code == 503, response.text
        assert response.json()["detail"] == "task manager is shutting down"
        assert executor.starts == []


def test_non_finite_lifecycle_values_rejected():
    for kwargs in (
        {"task_timeout": float("nan")},
        {"retention": float("inf")},
        {"task_timeout": -1},
        {"retention": -0.5},
    ):
        with pytest.raises(ValueError):
            create_app(settings=_settings(), executor=FakeExecutor(), **kwargs)


def test_table_full_rejects_new_submit():
    executor = BlockingExecutor()
    with _client(executor, max_workers=1, max_records=1, task_timeout=0) as client:
        assert client.post("/tasks", json={"task": "占位", "thread_id": "t-full1"}).status_code == 202
        assert executor.started.wait(3)

        response = client.post("/tasks", json={"task": "挤不进来", "thread_id": "t-full2"})
        assert response.status_code == 503, response.text
        assert "task table is full" in response.json()["detail"]

        assert client.delete("/tasks/t-full1").status_code == 202
        executor.release.set()
        assert executor.finished.wait(3)
        assert client.post("/tasks", json={"task": "腾出位置", "thread_id": "t-full3"}).status_code == 202


def test_app_version_comes_from_package_metadata():
    from importlib.metadata import version

    app = create_app(settings=_settings(), executor=FakeExecutor())
    assert app.version == version("multi-agent")


def test_cancel_running_task():
    executor = BlockingExecutor()
    with _client(executor) as client:
        client.post("/tasks", json={"task": "长任务", "thread_id": "t-cancel"})
        assert executor.started.wait(3)
        response = client.delete("/tasks/t-cancel")
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "canceled"
        assert response.json()["error"] == "task canceled by client"
        executor.release.set()
        assert executor.finished.wait(3)
        time.sleep(0.05)  # 等后台 _apply 抢锁
        final = client.get("/tasks/t-cancel").json()
        assert final["status"] == "canceled"
        assert final["answer"] == ""  # 迟到结果被丢弃


def test_canceled_task_not_revived_by_late_error():
    executor = BlockingExecutor(fail=True)
    with _client(executor) as client:
        client.post("/tasks", json={"task": "取消后报错", "thread_id": "t-cancel-err"})
        assert executor.started.wait(3)
        assert client.delete("/tasks/t-cancel-err").status_code == 202
        executor.release.set()
        assert executor.finished.wait(3)
        time.sleep(0.05)  # 等后台异常处理抢锁
        final = client.get("/tasks/t-cancel-err").json()
        assert final["status"] == "canceled"
        assert final["error"] == "task canceled by client"
        assert client.delete("/tasks/t-cancel-err").status_code == 202


def test_cancel_finished_task_409():
    with _client(FakeExecutor()) as client:
        client.post("/tasks", json={"task": "已完成", "thread_id": "t-done2"})
        _wait(client, "t-done2", "completed")
        response = client.delete("/tasks/t-done2")
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "task already finished"


def test_cancel_unknown_404():
    with _client(FakeExecutor()) as client:
        response = client.delete("/tasks/ghost")
        assert response.status_code == 404
        assert response.json()["detail"] == "task not found"


def test_cancel_is_idempotent():
    executor = BlockingExecutor()
    with _client(executor) as client:
        client.post("/tasks", json={"task": "取消两次", "thread_id": "t-cancel2"})
        assert executor.started.wait(3)
        assert client.delete("/tasks/t-cancel2").status_code == 202
        again = client.delete("/tasks/t-cancel2")
        assert again.status_code == 202
        assert again.json()["status"] == "canceled"
        executor.release.set()
        assert executor.finished.wait(3)


def test_task_timeout_marks_canceled():
    executor = BlockingExecutor()
    with _client(executor, task_timeout=0.05) as client:
        client.post("/tasks", json={"task": "会超时", "thread_id": "t-timeout"})
        assert executor.started.wait(3)
        final = _wait(client, "t-timeout", "canceled")
        assert final["error"] == "task timed out after 0.05s"
        executor.release.set()
        assert executor.finished.wait(3)


def test_retention_evicts_finished_tasks():
    with _client(FakeExecutor(), retention=0.3) as client:
        client.post("/tasks", json={"task": "短命", "thread_id": "t-evict"})
        _wait(client, "t-evict", "completed")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get("/tasks/t-evict").status_code == 404:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("任务未按 retention 回收")
        assert client.get("/tasks").json() == []


def test_approval_expires_and_releases_app():
    executor = FakeExecutor(start_result=lambda task, thread_id: _awaiting())
    with _client(executor, retention=0.3) as client:
        client.post("/tasks", json={"task": "待审批过期", "thread_id": "t-expire"})
        _wait(client, "t-expire", "awaiting_approval")
        final = _wait(client, "t-expire", "canceled")
        assert final["error"] == "approval expired"
        assert client.get("/approvals").json() == []
        assert client.post("/approvals/t-expire", json={"approved": True}).status_code == 409


def test_max_records_evicts_oldest_finished():
    with _client(FakeExecutor(), max_records=2) as client:
        for thread_id in ["t-cap1", "t-cap2", "t-cap3"]:
            client.post("/tasks", json={"task": thread_id, "thread_id": thread_id})
            _wait(client, thread_id, "completed")
        assert [item["thread_id"] for item in client.get("/tasks").json()] == [
            "t-cap2",
            "t-cap3",
        ]


def test_cancel_between_approve_and_resume_skips_action():
    """批准后、resume 真正开始前后到达的取消，不能让已批准的动作执行。"""

    executor = FakeExecutor(start_result=lambda task, thread_id: _awaiting())
    app = create_app(settings=_settings(), executor=executor, max_workers=2)
    manager = app.state.manager
    submitted: list[tuple[Any, tuple[Any, ...]]] = []

    class DeferredPool:
        def submit(self, fn, *args):
            submitted.append((fn, args))

        def shutdown(self, **kwargs):
            return None

    manager._pool = DeferredPool()
    with TestClient(app) as client:
        assert client.post("/tasks", json={"task": "上架", "thread_id": "t-race"}).status_code == 202
        run_fn, run_args = submitted.pop(0)
        run_fn(*run_args)
        _wait(client, "t-race", "awaiting_approval")

        assert client.post("/approvals/t-race", json={"approved": True}).status_code == 202
        resume_fn, resume_args = submitted.pop(0)  # 已排队但尚未执行
        assert client.delete("/tasks/t-race").status_code == 202

        resume_fn(*resume_args)  # 模拟线程池随后执行
        assert executor.resumes == []  # 取消后不得执行已批准动作
        final = client.get("/tasks/t-race").json()
        assert final["status"] == "canceled"
        assert final["error"] == "task canceled by client"


def test_resolve_host_never_returns_blank():
    from multi_agent.api.__main__ import _resolve_host

    assert _resolve_host(None) == "127.0.0.1"
    assert _resolve_host("") == "127.0.0.1"
    assert _resolve_host("   ") == "127.0.0.1"
    assert _resolve_host(" 0.0.0.0 ") == "0.0.0.0"


def _patch_graph(monkeypatch, decisions):
    """让真实图跑在假模型上：supervisor 走结构化路由，专家直接产出。"""
    called: list[str] = []

    class FakeChatModel(BaseChatModel):
        role: str = "supervisor"

        @property
        def _llm_type(self) -> str:
            return "fake-api"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="产出"))])

        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            def route(_input):
                next_agent = decisions.pop(0) if decisions else "FINISH"
                return schema(next_agent=next_agent, reason="测试路由")

            return RunnableLambda(route)

    def fake_make_llm(settings, *, model=None, temperature=0):
        return FakeChatModel()

    def fake_create_agent(llm, tools, system_prompt=None, **kwargs):
        class Worker:
            def invoke(self, inputs, config=None):
                called.append(system_prompt)
                return {"messages": [AIMessage(content=f"{system_prompt}-产出")]}

        return Worker()

    monkeypatch.setattr(graph_module, "make_llm", fake_make_llm)
    monkeypatch.setattr(graph_module, "_create_agent", fake_create_agent)
    return called


def test_real_graph_approval_round_trip(monkeypatch, tmp_path):
    called = _patch_graph(monkeypatch, ["publisher", "FINISH"])
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
    audit = tmp_path / "audit.jsonl"
    settings = _settings(audit_log=str(audit), checkpoint_db=str(tmp_path / "ckpt.sqlite"))

    app = create_app(settings=settings, registry=registry, max_workers=2)
    with TestClient(app) as client:
        assert (
            client.post("/tasks", json={"task": "把 SKU-1 上架", "thread_id": "it-real"}).status_code
            == 202
        )
        pending = _wait(client, "it-real", "awaiting_approval", timeout=5.0)
        assert pending["pending"]["agent"] == "publisher"
        approved = client.post("/approvals/it-real", json={"approved": True, "note": "同意"})
        assert approved.status_code == 202, approved.text
        final = _wait(client, "it-real", "completed", timeout=5.0)
        assert final["answer"] == "PUBLISH-产出"
        assert client.get("/approvals").json() == []

    assert called == ["PUBLISH"]
    rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["agent"] == "publisher"
    assert rows[0]["approved"] is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
