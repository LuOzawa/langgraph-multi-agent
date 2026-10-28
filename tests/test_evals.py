from __future__ import annotations

import json

import pytest

import evals.run as ev
from multi_agent.graph import RunResult


def _settings(**overrides) -> ev.Settings:
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
    return ev.Settings(**base)


def _result(answer: str, *, stop_reason: str = "finish", tokens: int = 10) -> RunResult:
    return RunResult(
        answer=answer,
        thread_id="eval-x",
        turns=2,
        stop_reason=stop_reason,
        usage={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


def test_score_answer_counts_synonym_groups_case_insensitive():
    points = [["检查点", "checkpoint"], ["恢复", "resume"]]
    score, missing = ev.score_answer("Checkpoint 保存状态，进程重启后可以 resume。", points)
    assert score == 1.0
    assert missing == []

    score, missing = ev.score_answer("只提到检查点。", points)
    assert score == 0.5
    assert missing == [["恢复", "resume"]]


def test_load_cases_skips_comments_and_validates(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '# 注释\n\n{"id": "a", "task": "t", "points": [["x"]]}\n',
        "utf-8",
    )
    cases = ev.load_cases(path)
    assert [case["id"] for case in cases] == ["a"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "b"}\n', "utf-8")
    with pytest.raises(ValueError, match="缺少"):
        ev.load_cases(bad)


def test_evaluate_case_requires_finish_and_threshold():
    case = {"id": "c", "task": "t", "points": [["甲"], ["乙"]]}
    ok = ev.evaluate_case(case, _result("甲 乙"), 1.0, min_score=0.6)
    assert ok["passed"] is True and ok["score"] == 1.0

    half = ev.evaluate_case(case, _result("只有甲"), 1.0, min_score=0.6)
    assert half["passed"] is False and half["missing_points"] == [["乙"]]

    errored = ev.evaluate_case(case, _result("甲 乙", stop_reason="error"), 1.0, min_score=0.6)
    assert errored["passed"] is False


def test_run_cases_aggregates_limit_and_worker_order():
    cases = [
        {"id": "one", "task": "任务一", "points": [["甲"]]},
        {"id": "two", "task": "任务二", "points": [["乙"]]},
        {"id": "three", "task": "任务三", "points": [["丙"]]},
    ]
    seen: list[str] = []

    def runner(task: str, thread_id: str, settings, registry) -> RunResult:
        seen.append(thread_id)
        answer = {"任务一": "甲", "任务二": "无", "任务三": "丙"}[task]
        return _result(answer, tokens=7)

    report = ev.run_cases(
        cases, settings=_settings(), limit=3, workers=2, min_score=0.5, runner=runner
    )

    assert [item["id"] for item in report["results"]] == ["one", "two", "three"]
    assert sorted(seen) == ["eval-one", "eval-three", "eval-two"]
    assert report["summary"]["cases"] == 3
    assert report["summary"]["passed"] == 2
    assert report["summary"]["total_tokens"] == 21
    assert report["results"][1]["missing_points"] == [["乙"]]


def test_run_cases_prefers_judge_scores():
    cases = [{"id": "one", "task": "任务", "points": [["甲"], ["乙"]]}]

    def runner(task: str, thread_id: str, settings, registry) -> RunResult:
        return _result("完全不含关键词")

    def judge(task: str, answer: str, points):
        return 0.9, [["乙"]]

    report = ev.run_cases(
        cases, settings=_settings(), runner=runner, judge=judge, judge_model="judge-x"
    )
    item = report["results"][0]
    assert item["scored_by"] == "llm"
    assert item["score"] == 0.9
    assert item["passed"] is True
    assert item["missing_points"] == [["乙"]]
    assert report["judge_model"] == "judge-x"


def test_llm_judge_parses_fenced_json():
    class FakeLLM:
        def invoke(self, prompt):
            class Out:
                content = '结果如下：\n```json\n{"score": 0.5, "missing": [1]}\n```'

            return Out()

    judge = ev.llm_judge(_settings(), llm=FakeLLM())
    score, missing = judge("任务", "回答", [["甲"], ["乙"]])
    assert score == 0.5
    assert missing == [["乙"]]


def test_llm_judge_falls_back_on_unparseable_output():
    class FakeLLM:
        def invoke(self, prompt):
            class Out:
                content = "我无法判断"

            return Out()

    judge = ev.llm_judge(_settings(), llm=FakeLLM())
    assert judge("任务", "回答", [["甲"]]) is None


def test_llm_judge_ignores_out_of_range_indexes():
    class FakeLLM:
        def invoke(self, prompt):
            class Out:
                content = '{"score": 1.5, "missing": [5]}'

            return Out()

    judge = ev.llm_judge(_settings(), llm=FakeLLM())
    score, missing = judge("任务", "回答", [["甲"]])
    assert score == 1.0
    assert missing == []


def test_main_writes_report_and_strict_exit(monkeypatch, tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id": "a", "task": "t", "points": [["x"]]}\n', "utf-8")
    out = tmp_path / "report.json"
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def fake_run_cases(items, **kwargs):
        return {
            "started_at": "now",
            "model": "fake",
            "summary": {
                "cases": 1,
                "passed": 0,
                "failed": 1,
                "avg_score": 0.0,
                "total_tokens": 0,
                "total_latency_s": 0.0,
                "passed_all": False,
            },
            "results": [
                {
                    "id": "a",
                    "score": 0.0,
                    "passed": False,
                    "stop_reason": "finish",
                    "turns": 1,
                    "tokens": 0,
                    "latency_s": 0.1,
                    "missing_points": [["x"]],
                    "answer": "",
                }
            ],
        }

    monkeypatch.setattr(ev, "run_cases", fake_run_cases)

    assert ev.main(["--cases", str(cases), "--out", str(out)]) == 0
    assert json.loads(out.read_text("utf-8"))["summary"]["passed"] == 0
    assert ev.main(["--cases", str(cases), "--out", str(out), "--strict"]) == 1


def test_main_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ev, "load_dotenv", lambda: None)  # 不读本机 .env
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id": "a", "task": "t", "points": [["x"]]}\n', "utf-8")

    assert ev.main(["--cases", str(cases), "--out", str(tmp_path / "r.json")]) == 1
