from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from multi_agent.audit import record_approval
from multi_agent.config import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        api_key="test",
        base_url="http://example.invalid/v1",
        model="fake",
        supervisor_model="fake",
        max_turns=1,
        worker_max_steps=1,
        context_max_tokens=0,
        checkpoint_db="memory",
        audit_log=str(tmp_path / "approvals.jsonl"),
        request_timeout=1,
        max_retries=0,
        log_level="WARNING",
    )


def test_concurrent_approvals_do_not_lose_or_interleave_lines(tmp_path):
    """多线程并发审批时，JSONL 不能丢行、不能写出半行。

    run_parallel 把多个任务丢进线程池，审批记录同写一个文件；
    Windows 上 append 不是原子的，实测 400 条只剩 151 条、其中 45 行坏 JSON。
    """
    settings = _settings(tmp_path)
    total = 400

    def write_one(index: int) -> None:
        record_approval(
            settings=settings,
            thread_id=f"t-{index}",
            payload={"agent": "a", "tool": "x", "task": "y" * 200},
            decision={"approved": index % 2 == 0, "note": "n" * 100},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(total)))

    lines = (tmp_path / "approvals.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == total
    records = [json.loads(line) for line in lines]
    assert {record["thread_id"] for record in records} == {f"t-{i}" for i in range(total)}


def test_approval_log_can_be_disabled(tmp_path):
    settings = replace(_settings(tmp_path), audit_log="off")
    record_approval(
        settings=settings,
        thread_id="t-0",
        payload={"agent": "a"},
        decision={"approved": True},
    )
    assert not (tmp_path / "approvals.jsonl").exists()