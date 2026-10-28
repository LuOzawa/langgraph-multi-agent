from __future__ import annotations

import multi_agent.cli as cli
from multi_agent.graph import RunResult


def _result(stop_reason: str) -> RunResult:
    return RunResult(
        answer="输出", thread_id="t-cli", turns=1, stop_reason=stop_reason, usage={}
    )


def _patch_cli(monkeypatch, result: RunResult) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(cli, "build_graph", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_task_full", lambda *args, **kwargs: result)


def test_cli_exit_code_reflects_task_failure(monkeypatch):
    _patch_cli(monkeypatch, _result("error"))
    assert cli.main(["--no-stream", "会失败的任务"]) == 1


def test_cli_exit_code_zero_on_finish(monkeypatch):
    _patch_cli(monkeypatch, _result("finish"))
    assert cli.main(["--no-stream", "正常任务"]) == 0


def test_prompt_approval_keeps_yes_when_note_hits_eof(monkeypatch):
    answers = iter(["y"])

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration as exc:  # 备注行读到 EOF
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli._prompt_approval({"agent": "publisher"}) == {"approved": True, "note": ""}


def test_prompt_approval_rejects_when_first_answer_hits_eof(monkeypatch):
    def fake_input(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli._prompt_approval({"agent": "publisher"}) == {
        "approved": False,
        "note": "非交互环境，默认拒绝",
    }


def test_prompt_approval_reads_note(monkeypatch):
    answers = iter(["yes", "复核通过"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert cli._prompt_approval({"agent": "publisher"}) == {
        "approved": True,
        "note": "复核通过",
    }
