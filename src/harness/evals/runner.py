"""Eval runner, reports and regression gate.

Offline mode (default, used by CI) replaces the model with the task's scripted trajectory
and checks the harness end to end: prompts, policy, sandbox, tools, budgets. Live mode
uses the models from the agent YAML and measures model + prompt quality; add a judge
model to score rubrics.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness.config import Settings
from harness.core.types import RunStatus
from harness.demo import SCRIPTS
from harness.evals.scorers import Score, deterministic_scores, rubric_score, tools_called
from harness.evals.tasks import EvalSuite, EvalTask
from harness.llm.base import LLMProvider
from harness.runtime import Harness

MAX_AUTO_APPROVALS = 10


class TaskResult(BaseModel):
    suite: str
    task_id: str
    passed: bool
    skipped: bool = False
    score: float
    scores: list[Score] = Field(default_factory=list)
    answer: str | None = None
    stop_reason: str | None = None
    steps: int = 0
    tools: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None


class SuiteResult(BaseModel):
    suite: str
    agent: str
    mode: str
    tasks: list[TaskResult]

    @property
    def scored(self) -> list[TaskResult]:
        return [t for t in self.tasks if not t.skipped]

    @property
    def pass_rate(self) -> float:
        scored = self.scored
        return sum(t.passed for t in scored) / len(scored) if scored else 1.0


async def run_task(
    suite: EvalSuite,
    task: EvalTask,
    settings: Settings,
    *,
    offline: bool,
    judge: LLMProvider | None,
) -> TaskResult:
    agent = task.agent or suite.agent
    overrides: dict[str, LLMProvider] = {}
    if offline:
        if task.script is None:
            return TaskResult(
                suite=suite.suite,
                task_id=task.id,
                passed=True,
                skipped=True,
                score=0.0,
                error="no offline script; run with --live",
            )
        overrides[agent] = SCRIPTS[task.script]()
    # A fresh harness per task: no memory or rate-limit state leaks between tasks.
    harness = Harness.build(settings, llm_overrides=overrides)
    started = time.perf_counter()
    try:
        state = await harness.run(agent, task.input, user_id=f"eval:{suite.suite}")
        approvals = 0
        while (
            state.status == RunStatus.AWAITING_APPROVAL
            and task.auto_approve
            and approvals < MAX_AUTO_APPROVALS
        ):
            for pending in list(state.pending_approvals):
                state = await harness.approvals.decide(
                    state.run_id, pending.call.id, approved=True, reviewer="eval"
                )
                approvals += 1
            state.status = RunStatus.RUNNING
            await harness.state.runs.save(state)
            state = await harness.execute(state.run_id)
    except Exception as exc:
        return TaskResult(
            suite=suite.suite,
            task_id=task.id,
            passed=False,
            score=0.0,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - started,
        )
    finally:
        await harness.aclose()

    scores = deterministic_scores(state, task.expect)
    if task.expect.rubric:
        scores.append(
            await rubric_score(
                judge,
                task.input,
                state.answer or "",
                task.expect.rubric,
                task.expect.rubric_threshold,
            )
        )
    counted = [s for s in scores if not s.skipped]
    return TaskResult(
        suite=suite.suite,
        task_id=task.id,
        passed=all(s.passed for s in counted),
        score=sum(s.score for s in counted) / len(counted) if counted else 1.0,
        scores=scores,
        answer=state.answer,
        stop_reason=state.stop_reason.value if state.stop_reason else None,
        steps=state.step,
        tools=tools_called(state),
        total_tokens=state.usage.total_tokens,
        cost_usd=state.cost_usd,
        latency_s=round(time.perf_counter() - started, 3),
        error=state.error,
    )


async def run_suites(
    suites: list[EvalSuite],
    settings: Settings,
    *,
    offline: bool = True,
    judge: LLMProvider | None = None,
) -> list[SuiteResult]:
    results: list[SuiteResult] = []
    for suite in suites:
        # Sequential on purpose: predictable cost and rate limits for live runs.
        tasks = [
            await run_task(suite, t, settings, offline=offline, judge=judge) for t in suite.tasks
        ]
        results.append(
            SuiteResult(
                suite=suite.suite,
                agent=suite.agent,
                mode="offline" if offline else "live",
                tasks=tasks,
            )
        )
    return results


# ---- baseline & regression gate ---------------------------------------------------


def to_baseline(results: list[SuiteResult]) -> dict[str, Any]:
    return {
        "suites": {
            r.suite: {
                "pass_rate": r.pass_rate,
                "tasks": {
                    t.task_id: {"passed": t.passed, "score": round(t.score, 4)} for t in r.scored
                },
            }
            for r in results
        }
    }


def compare(
    results: list[SuiteResult], baseline: dict[str, Any], tolerance: float = 0.0
) -> list[str]:
    """Regressions vs the baseline: a task that passed now fails, a score dropped by more
    than `tolerance`, or a suite pass rate fell. New tasks are not regressions."""
    problems: list[str] = []
    for r in results:
        base = baseline.get("suites", {}).get(r.suite)
        if base is None:
            continue
        if r.pass_rate + tolerance < float(base["pass_rate"]):
            problems.append(
                f"{r.suite}: pass rate {r.pass_rate:.2f} < baseline {base['pass_rate']:.2f}"
            )
        for t in r.scored:
            b = base["tasks"].get(t.task_id)
            if b is None:
                continue
            if b["passed"] and not t.passed:
                problems.append(f"{r.suite}/{t.task_id}: passed in baseline, now fails")
            elif t.score + tolerance < float(b["score"]):
                problems.append(f"{r.suite}/{t.task_id}: score {t.score:.2f} < {b['score']:.2f}")
    return problems


# ---- reports ---------------------------------------------------------------------


def render_markdown(results: list[SuiteResult], regressions: list[str]) -> str:
    out = [f"# Eval report ({datetime.now(UTC):%Y-%m-%d %H:%M UTC})", ""]
    for r in results:
        out += [
            f"## {r.suite} (agent `{r.agent}`, {r.mode})",
            "",
            f"Pass rate: **{r.pass_rate:.0%}** ({sum(t.passed for t in r.scored)}/"
            f"{len(r.scored)} scored, {len(r.tasks) - len(r.scored)} skipped)",
            "",
            "| task | result | score | steps | tools | tokens | cost $ | time s | notes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for t in r.tasks:
            result = "skip" if t.skipped else ("pass" if t.passed else "**FAIL**")
            notes = "; ".join(f"{s.name}: {s.detail}" for s in t.scores if not s.passed)
            notes = (notes or t.error or "").replace("|", "\\|")[:200]
            out.append(
                f"| {t.task_id} | {result} | {t.score:.2f} | {t.steps} | "
                f"{', '.join(t.tools)} | {t.total_tokens} | {t.cost_usd:.4f} | "
                f"{t.latency_s:.2f} | {notes} |"
            )
        out.append("")
    out += ["## Regressions", ""]
    out += [f"- {p}" for p in regressions] if regressions else ["None."]
    return "\n".join(out) + "\n"


def write_reports(results: list[SuiteResult], regressions: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    md = out_dir / f"eval-{stamp}.md"
    md.write_text(render_markdown(results, regressions), encoding="utf-8")
    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps(
            {
                "results": [r.model_dump(mode="json") for r in results],
                "regressions": regressions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md
