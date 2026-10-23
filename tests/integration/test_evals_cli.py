from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.config import Settings
from harness.core.types import RunState, RunStatus, StopReason
from harness.evals import (
    EvalSuite,
    Expectations,
    SuiteResult,
    TaskResult,
    compare,
    deterministic_scores,
    load_suites,
    render_markdown,
    rubric_score,
    run_suites,
    to_baseline,
)
from harness.llm import FakeLLM, Message, ToolCall

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def unsafe_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SANDBOX_BACKEND", "local_unsafe")
    monkeypatch.setenv("HARNESS_ALLOW_UNSAFE_SANDBOX", "true")
    monkeypatch.setenv("HARNESS_AUDIT_LOG_PATH", str(REPO / "var" / "test-audit.log"))


def _state(answer: str, tools: list[str], steps: int = 2) -> RunState:
    calls = [ToolCall(id=str(i), name=t) for i, t in enumerate(tools)]
    return RunState(
        agent="a",
        input="q",
        answer=answer,
        step=steps,
        status=RunStatus.COMPLETED,
        stop_reason=StopReason.COMPLETED,
        messages=[Message.assistant("", calls)],
    )


def test_deterministic_scorers() -> None:
    exp = Expectations(
        exact="West wins $5.00",
        contains=["west"],
        not_contains=["error"],
        regex=r"\$\d",
        tools_called=["python_exec"],
        tools_not_called=["web_fetch"],
        max_steps=3,
    )
    scores = {
        s.name: s for s in deterministic_scores(_state("West wins $5.00", ["python_exec"]), exp)
    }
    assert all(s.passed for s in scores.values()), scores
    bad = {
        s.name: s for s in deterministic_scores(_state("error: none", ["web_fetch"], steps=9), exp)
    }
    assert not any(s.passed for name, s in bad.items() if name != "stop_reason")
    assert "missing ['west']" in bad["contains"].detail
    assert "took 9" in bad["max_steps"].detail


async def test_rubric_judge_parsing() -> None:
    skipped = await rubric_score(None, "t", "a", "r", 0.7)
    assert skipped.skipped and skipped.passed
    judge = FakeLLM([FakeLLM.text('Sure: {"score": 0.9, "reason": "solid"}')])
    good = await rubric_score(judge, "t", "a", "r", 0.7)
    assert good.passed and good.score == 0.9 and good.detail == "solid"
    assert "<response>" in judge.calls[0][-1].content
    low = await rubric_score(FakeLLM([FakeLLM.text('{"score": 0.2}')]), "t", "a", "r", 0.7)
    assert not low.passed
    junk = await rubric_score(FakeLLM([FakeLLM.text("no json here")]), "t", "a", "r", 0.7)
    assert not junk.passed and "unparseable" in junk.detail


def _result(passed: bool, score: float, rate_tasks: int = 1) -> list[SuiteResult]:
    return [
        SuiteResult(
            suite="s",
            agent="a",
            mode="offline",
            tasks=[TaskResult(suite="s", task_id="t1", passed=passed, score=score)],
        )
    ]


def test_baseline_and_regression_gate() -> None:
    base = to_baseline(_result(True, 1.0))
    assert compare(_result(True, 1.0), base) == []
    problems = compare(_result(False, 0.5), base)
    assert any("now fails" in p for p in problems)
    assert any("pass rate" in p for p in problems)
    assert compare(_result(True, 0.95), base, tolerance=0.1) == []
    assert compare(_result(True, 0.8), base, tolerance=0.1)
    assert compare(_result(True, 1.0), {"suites": {}}) == []
    md = render_markdown(_result(False, 0.0), ["x regressed"])
    assert "**FAIL**" in md and "- x regressed" in md


def test_suite_loading_and_validation(tmp_path: Path) -> None:
    suites = load_suites(REPO / "evals" / "tasks")
    assert suites[0].suite == "data_analyst"
    (tmp_path / "a.yaml").write_text("suite: s\nagent: a\ntasks: []\n")
    (tmp_path / "b.yaml").write_text("suite: s\nagent: a\ntasks: []\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_suites(tmp_path)
    with pytest.raises(Exception, match="id"):
        EvalSuite.model_validate(
            {"suite": "s", "agent": "a", "tasks": [{"id": "Bad Id", "input": "x"}]}
        )


async def test_offline_suite_runs_end_to_end(unsafe_sandbox: None) -> None:
    settings = Settings(_env_file=None)
    results = await run_suites(load_suites(REPO / "evals" / "tasks"), settings, offline=True)
    [suite] = results
    by_id = {t.task_id: t for t in suite.tasks}
    assert by_id["sales-insights"].passed, by_id["sales-insights"]
    assert by_id["pii-table-blocked"].passed, by_id["pii-table-blocked"]
    assert by_id["revenue-by-region-live"].skipped
    baseline = json.loads((REPO / "evals" / "baseline.json").read_text())
    assert compare(results, baseline) == []


def test_cli_demo_offline(unsafe_sandbox: None) -> None:
    result = CliRunner().invoke(
        app, ["run", "data_analyst", "Analyze data/sales.csv and give me insights", "--offline"]
    )
    assert result.exit_code == 0, result.output
    assert "West is the strongest region" in result.output
    assert "status=completed" in result.output


def test_cli_json_output_and_errors(unsafe_sandbox: None) -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["run", "data_analyst", "Analyze data/sales.csv", "--json"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output[out.output.index("{") :])["status"] == "completed"
    bad = runner.invoke(app, ["run", "ghost", "x"])
    assert bad.exit_code != 0 and "unknown agent" in bad.output
    no_script = runner.invoke(app, ["run", "research", "x"])
    assert no_script.exit_code != 0 and "no offline script" in no_script.output


def test_cli_agents_eval_and_audit(unsafe_sandbox: None, tmp_path: Path) -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["agents"])
    assert (
        listed.exit_code == 0 and "data_analyst" in listed.output and "supervisor" in listed.output
    )

    out = runner.invoke(
        app,
        ["eval", "evals/tasks", "--baseline", "evals/baseline.json", "--report-dir", str(tmp_path)],
    )
    assert out.exit_code == 0, out.output
    assert list(tmp_path.glob("eval-*.md")) and list(tmp_path.glob("eval-*.json"))

    audit = REPO / "var" / "test-audit.log"
    runner.invoke(app, ["run", "data_analyst", "Analyze data/sales.csv", "--offline"])
    ok = runner.invoke(app, ["audit-verify", str(audit)])
    assert ok.exit_code == 0 and "chain intact" in ok.output
    tampered = tmp_path / "t.log"
    lines = audit.read_text().splitlines()
    tampered.write_text("\n".join([lines[0].replace("harness", "mallory", 1), *lines[1:]]) + "\n")
    bad = runner.invoke(app, ["audit-verify", str(tampered)])
    assert bad.exit_code == 1 and "tampering" in bad.output
