from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import multi_agent.api.service as service
from multi_agent.api.app import create_app
from multi_agent.api.service import LangGraphExecutor, StartOutcome
from multi_agent.config import Settings
from multi_agent.graph import RunResult

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


def _client(executor: FakeExecutor, settings: Settings | None = None) -> TestClient:
    app = create_app(settings=settings or _settings(), executor=executor, max_workers=2)
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


def test_submit_approval_flow(tmp_path):
    audit = tmp_path / "approvals.jsonl"
    executor = FakeExecutor(
        start_result=lambda task, thread_id: _awaiting("publisher"),
        resume_result=lambda thread_id, decision, app: _completed("已上架"),
    )
    with _client(executor, _settings(audit_log=str(audit))) as client:
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


def test_reject_approval(tmp_path):
    audit = tmp_path / "approvals.jsonl"
    executor = FakeExecutor(
        start_result=lambda task, thread_id: _awaiting(),
        resume_result=lambda thread_id, decision, app: _completed("已拒绝，未执行"),
    )
    with _client(executor, _settings(audit_log=str(audit))) as client:
        client.post("/tasks", json={"task": "上架新品", "thread_id": "t-reject"})
        _wait(client, "t-reject", "awaiting_approval")

        response = client.post(
            "/approvals/t-reject", json={"approved": False, "note": "价格未确认"}
        )
        assert response.status_code == 202, response.text
        final = _wait(client, "t-reject", "completed")
        assert final["answer"] == "已拒绝，未执行"

        assert executor.resumes[0][1] == {"approved": False, "note": "价格未确认"}
        rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["approved"] is False
        assert rows[0]["thread_id"] == "t-reject"


def test_audit_log_written(tmp_path):
    audit = tmp_path / "approvals.jsonl"
    executor = FakeExecutor(start_result=lambda task, thread_id: _awaiting("writer"))
    with _client(executor, _settings(audit_log=str(audit))) as client:
        client.post("/tasks", json={"task": "发布公告", "thread_id": "t-audit"})
        _wait(client, "t-audit", "awaiting_approval")
        client.post("/approvals/t-audit", json={"approved": True, "note": "OK"})
        _wait(client, "t-audit", "completed")

    rows = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["thread_id"] == "t-audit"
    assert row["agent"] == "writer"
    assert row["approved"] is True
    assert row["note"] == "OK"


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
