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
加 `--judge` / `--judge-model` 时改用模型判定覆盖语义（同义改写也能算命中），解析失败自动退回关键词打分。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from multi_agent.config import Settings
from multi_agent.graph import RunResult, message_text, run_task_full
from multi_agent.registry import AgentRegistry, make_llm

DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_OUT = Path(__file__).with_name("report.json")
DEFAULT_MIN_SCORE = 0.6

Runner = Callable[[str, str, Settings, AgentRegistry | None], RunResult]
Judge = Callable[[str, str, list[list[str]]], tuple[float, list[list[str]]] | None]

JUDGE_PROMPT = """你在给一次问答打分。只判断语义是否覆盖要点，不要因为措辞不同而扣分。

任务：
{task}

回答：
{answer}

要点（编号）：
{points}

只输出 JSON，不要解释：
{{"score": 0 到 1 之间的小数, "missing": [未覆盖的要点编号]}}"""


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


def _extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"没有 JSON 对象: {text[:80]!r}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("判分结果不是对象")
    return data


def llm_judge(
    settings: Settings,
    model: str | None = None,
    *,
    llm: Any | None = None,
) -> Judge:
    """用模型判定要点覆盖度；解析失败返回 None，由调用方退回关键词打分。"""
    judge_llm = llm or make_llm(settings, model=model or settings.model)

    def judge(task: str, answer: str, points: list[list[str]]) -> tuple[float, list[list[str]]] | None:
        numbered = "\n".join(
            f"{index}. " + " / ".join(str(word) for word in group)
            for index, group in enumerate(points)
        )
        prompt = JUDGE_PROMPT.format(task=task, answer=answer, points=numbered)
        try:
            raw = judge_llm.invoke(prompt)
            text = getattr(raw, "content", raw)
            data = _extract_json(str(text))
            indexes = data.get("missing") or []
            missing = [
                list(points[index])
                for index in indexes
                if isinstance(index, int) and 0 <= index < len(points)
            ]
            if "score" in data:
                score = float(data["score"])
            else:
                score = 1 - len(missing) / len(points)
            return max(0.0, min(1.0, score)), missing
        except Exception:  # noqa: BLE001 - 判分失败不应让评测挂掉
            return None

    return judge


def transcript_text(result: RunResult) -> str:
    """评测看整段对话，而不是最后一条消息（末条常是「已交付」之类收尾）。"""
    parts = [message_text(message) for message in result.messages]
    text = "\n\n".join(part for part in parts if part.strip())
    return text or result.answer


def evaluate_case(
    case: dict[str, Any],
    result: RunResult,
    latency_s: float,
    *,
    min_score: float,
    judged: tuple[float, list[list[str]]] | None = None,
) -> dict[str, Any]:
    text = transcript_text(result)
    if judged is None:
        score, missing = score_answer(text, case["points"])
        scored_by = "keywords"
    else:
        score, missing = judged
        score = max(0.0, min(1.0, float(score)))
        scored_by = "llm"
    passed = score >= min_score and result.stop_reason == "finish"
    return {
        "id": case["id"],
        "score": round(score, 3),
        "passed": passed,
        "scored_by": scored_by,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "tokens": int((result.usage or {}).get("total_tokens", 0)),
        "latency_s": round(latency_s, 2),
        "missing_points": missing,
        "answer": result.answer,
    }


def _default_runner(task: str, thread_id: str, settings: Settings, registry: AgentRegistry | None) -> RunResult:
    return run_task_full(task, settings=settings, registry=registry, thread_id=thread_id)


def seed_notes(notes_dir: Path) -> None:
    """把项目文档放进评测工作区，模拟「有资料可查」的真实用法。"""
    notes_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    sources = [repo / "README.md", *sorted((repo / "docs").glob("*.md"))]
    for path in sources:
        if path.exists():
            (notes_dir / f"{path.stem}.md").write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )


def run_cases(
    cases: list[dict[str, Any]],
    *,
    settings: Settings,
    registry: AgentRegistry | None = None,
    limit: int | None = None,
    workers: int = 1,
    min_score: float = DEFAULT_MIN_SCORE,
    runner: Runner | None = None,
    judge: Judge | None = None,
    judge_model: str | None = None,
    seed_docs: bool = True,
    only: str | None = None,
) -> dict[str, Any]:
    selected = cases[:limit] if limit else list(cases)
    if only:
        selected = [case for case in selected if case["id"] == only]
        if not selected:
            raise ValueError(f"没有匹配的用例 id: {only}")
    run = runner or _default_runner
    run_id = uuid.uuid4().hex[:8]  # 避免复用同一 thread 的旧检查点
    # 评测不写用户工作区：检查点与笔记都放到临时目录
    workdir = Path(tempfile.mkdtemp(prefix=f"multi-agent-eval-{run_id}-"))
    notes_dir = workdir / "notes"
    if seed_docs:
        seed_notes(notes_dir)
    eval_settings = replace(settings, checkpoint_db=str(workdir / "checkpoints.sqlite"))
    previous_notes = os.environ.get("MULTI_AGENT_NOTES_DIR")
    os.environ["MULTI_AGENT_NOTES_DIR"] = str(notes_dir)

    def one(case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = run(case["task"], f"eval-{run_id}-{case['id']}", eval_settings, registry)
        judged = judge(case["task"], transcript_text(result), case["points"]) if judge else None
        return evaluate_case(
            case,
            result,
            time.monotonic() - started,
            min_score=min_score,
            judged=judged,
        )

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval") as pool:
                results = list(pool.map(one, selected))
        else:
            results = [one(case) for case in selected]
    finally:
        if previous_notes is None:
            os.environ.pop("MULTI_AGENT_NOTES_DIR", None)
        else:
            os.environ["MULTI_AGENT_NOTES_DIR"] = previous_notes
        shutil.rmtree(workdir, ignore_errors=True)

    tokens = sum(item["tokens"] for item in results)
    passed = sum(1 for item in results if item["passed"])
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "supervisor_model": settings.supervisor_model,
        "base_url": settings.base_url,
        "min_score": min_score,
        "judge_model": judge_model,
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
    parser.add_argument("--only", default=None, help="只跑指定 id 的用例")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="不在评测工作区里预置项目文档（默认会放 README 与 docs/）",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发用例数（默认 1）")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="用模型判定要点覆盖度（默认关键词匹配）",
    )
    parser.add_argument(
        "--judge-model", default=None, help="判分模型（默认与 --model 相同）"
    )
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

    judge = None
    judge_model = None
    if args.judge or args.judge_model:
        judge_model = args.judge_model or settings.model
        judge = llm_judge(settings, judge_model)

    report = run_cases(
        cases,
        settings=settings,
        limit=args.limit,
        workers=max(1, args.workers),
        min_score=args.min_score,
        judge=judge,
        judge_model=judge_model,
        seed_docs=not args.no_seed,
        only=args.only,
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
