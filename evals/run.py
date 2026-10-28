"""离线 / 真实模型的评测入口：跑用例、按要点打分、输出 JSON 报告。

用法（在仓库根目录）：
    python evals/run.py                          # 读 .env / OPENAI_*，跑全部用例
    python evals/run.py --limit 3 --out /tmp/report.json
    python evals/run.py --model deepseek-v4-flash --base-url https://api.deepseek.com/v1
    python evals/run.py --strict                 # 有未通过用例时 exit 1（可用于 CI/回归）

用例格式（JSONL，每行一条）：
    {"id": "ckpt", "task": "问题文本", "points": [["检查点", "checkpoint"], ["恢复", "resume"]]}

`points` 是「要点组」的列表：一组内的词是**同义**，命中任意一个即算该要点满足；
得分 = 满足的组数 / 总组数。默认 `--min-score 0.6` 且 `stop_reason` 必须是 `finish` 才算通过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from multi_agent.config import Settings
from multi_agent.graph import RunResult, run_task_full
from multi_agent.registry import AgentRegistry

DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_OUT = Path(__file__).with_name("report.json")
DEFAULT_MIN_SCORE = 0.6

Runner = Callable[[str, str, Settings, AgentRegistry | None], RunResult]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case = json.loads(line)
        if not case.get("id") or not case.get("task") or not case.get("points"):
            raise ValueError(f"{path}:{lineno} 用例缺少 id / task / points")
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} 里没有用例")
    return cases


def score_answer(answer: str, points: list[list[str]]) -> tuple[float, list[list[str]]]:
    """返回（得分, 未命中的要点组）。子串匹配，忽略大小写与首尾空白。"""
    text = (answer or "").lower()
    missing: list[list[str]] = []
    for group in points:
        if not any(str(word).lower() in text for word in group):
            missing.append(list(group))
    total = len(points)
    return (total - len(missing)) / total if total else 0.0, missing


def evaluate_case(
    case: dict[str, Any],
    result: RunResult,
    latency_s: float,
    *,
    min_score: float,
) -> dict[str, Any]:
    score, missing = score_answer(result.answer, case["points"])
    passed = score >= min_score and result.stop_reason == "finish"
    return {
        "id": case["id"],
        "score": round(score, 3),
        "passed": passed,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "tokens": int((result.usage or {}).get("total_tokens", 0)),
        "latency_s": round(latency_s, 2),
        "missing_points": missing,
        "answer": result.answer,
    }


def _default_runner(task: str, thread_id: str, settings: Settings, registry: AgentRegistry | None) -> RunResult:
    return run_task_full(task, settings=settings, registry=registry, thread_id=thread_id)


def run_cases(
    cases: list[dict[str, Any]],
    *,
    settings: Settings,
    registry: AgentRegistry | None = None,
    limit: int | None = None,
    workers: int = 1,
    min_score: float = DEFAULT_MIN_SCORE,
    runner: Runner | None = None,
) -> dict[str, Any]:
    selected = cases[:limit] if limit else list(cases)
    run = runner or _default_runner

    def one(case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = run(case["task"], f"eval-{case['id']}", settings, registry)
        return evaluate_case(case, result, time.monotonic() - started, min_score=min_score)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval") as pool:
            results = list(pool.map(one, selected))
    else:
        results = [one(case) for case in selected]

    tokens = sum(item["tokens"] for item in results)
    passed = sum(1 for item in results if item["passed"])
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "supervisor_model": settings.supervisor_model,
        "base_url": settings.base_url,
        "min_score": min_score,
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "avg_score": round(sum(item["score"] for item in results) / len(results), 3),
            "total_tokens": tokens,
            "total_latency_s": round(sum(item["latency_s"] for item in results), 2),
            "passed_all": passed == len(results),
        },
        "results": results,
    }


def _settings_from_args(args: argparse.Namespace) -> Settings:
    load_dotenv()
    api_key = (args.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "缺少 API key：设置 OPENAI_API_KEY（或写进 .env），或传 --api-key。见 README「快速开始」。"
        )
    settings = Settings.from_env()
    overrides: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        overrides["base_url"] = args.base_url.rstrip("/")
    if args.model:
        overrides["model"] = args.model
        overrides["supervisor_model"] = args.model
    return replace(settings, **overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多 Agent 评测：跑用例并输出要点命中报告")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条用例")
    parser.add_argument("--workers", type=int, default=1, help="并发用例数（默认 1）")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--strict", action="store_true", help="有未通过用例时返回非零退出码"
    )
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
        settings = _settings_from_args(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"评测启动失败: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report = run_cases(
        cases,
        settings=settings,
        limit=args.limit,
        workers=max(1, args.workers),
        min_score=args.min_score,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    for item in report["results"]:
        flag = "PASS" if item["passed"] else "FAIL"
        print(
            f"[{flag}] {item['id']:<12} score={item['score']:.2f} "
            f"stop={item['stop_reason'] or '-'} turns={item['turns']} "
            f"tokens={item['tokens']} {item['latency_s']}s"
        )
    print(
        f"\n{summary['passed']}/{summary['cases']} 通过，均分 {summary['avg_score']:.2f}，"
        f"tokens {summary['total_tokens']}，耗时 {summary['total_latency_s']}s → {args.out}"
    )
    if args.strict and not summary["passed_all"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
