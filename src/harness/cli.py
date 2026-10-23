"""Command-line interface.

harness run data_analyst "Analyze data/sales.csv and give me insights"
harness eval evals/tasks --baseline evals/baseline.json
harness agents
harness serve
harness audit-verify
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from harness.config import Settings, get_settings
from harness.core.agent_spec import load_agents
from harness.core.types import RunEvent, RunState, RunStatus
from harness.demo import SCRIPTS
from harness.llm.base import LLMProvider
from harness.llm.factory import ModelRef, build_provider

app = typer.Typer(add_completion=False, no_args_is_help=True, help="agent-harness CLI")

# Agents with a scripted offline trajectory (used when no API key is configured).
OFFLINE_SCRIPTS = {"data_analyst": "data_analyst_demo"}


def _has_key(provider: str, s: Settings) -> bool:
    return {
        "anthropic": s.anthropic_api_key is not None,
        "openai": s.openai_api_key is not None,
    }.get(provider, True)


def _echo_event(event: RunEvent) -> None:
    d = event.data
    if event.type == "text_delta":
        typer.echo(d.get("text", ""), nl=False)
    elif event.type == "plan":
        typer.secho(f"\n[step {event.step}] plan: {', '.join(d.get('tools', []))}", fg="cyan")
    elif event.type == "observe":
        status = "error" if d.get("is_error") else "ok"
        typer.secho(
            f"[step {event.step}] {d.get('tool')} -> {status} ({d.get('duration_ms', 0):.0f} ms)",
            fg="red" if d.get("is_error") else "green",
        )
    elif event.type == "approval_required":
        typer.secho(f"\n[approval needed] {d.get('tool')}: {d.get('reason')}", fg="yellow")
    elif event.type == "compacted":
        typer.secho("[context compacted]", fg="magenta")


async def _run(
    agent: str,
    prompt: str,
    *,
    user: str,
    session: str | None,
    offline: bool,
    auto_approve: bool,
    as_json: bool,
) -> RunState:
    from harness.runtime import Harness

    settings = get_settings()
    overrides: dict[str, LLMProvider] = {}
    specs = load_agents(settings.agents_dir)
    spec = specs.get(agent)
    if spec is None:
        raise typer.BadParameter(f"unknown agent {agent!r}; available: {sorted(specs)}")
    if offline or not _has_key(spec.model.provider, settings):
        script = OFFLINE_SCRIPTS.get(agent)
        if script is None:
            raise typer.BadParameter(
                f"no API key for {spec.model.provider} and no offline script for {agent!r}"
            )
        if not as_json:
            typer.secho(
                f"Offline mode: no {spec.model.provider} API key, replaying a scripted model "
                "trajectory. Every other layer (policy, sandbox, tools) runs for real.",
                fg="yellow",
            )
        overrides[agent] = SCRIPTS[script]()
    harness = Harness.build(settings, llm_overrides=overrides)

    if not as_json:

        async def printer(event: RunEvent) -> None:
            _echo_event(event)

        harness.extra_sinks.append(printer)
    try:
        await harness.startup()
        state = await harness.run(agent, prompt, user_id=user, session_id=session)
        while state.status == RunStatus.AWAITING_APPROVAL:
            for pending in list(state.pending_approvals):
                call = pending.call
                if auto_approve:
                    ok = True
                elif sys.stdin.isatty():
                    typer.echo(json.dumps(call.arguments, indent=2)[:2000])
                    ok = typer.confirm(f"Approve {call.name}?", default=False)
                else:
                    typer.secho("Run paused for approval (use --auto-approve).", fg="yellow")
                    return state
                state = await harness.approvals.decide(
                    state.run_id, call.id, approved=ok, reviewer=f"cli:{user}"
                )
            state.status = RunStatus.RUNNING
            await harness.state.runs.save(state)
            state = await harness.execute(state.run_id)
        return state
    finally:
        await harness.aclose()


@app.command()
def run(
    agent: Annotated[str, typer.Argument(help="Agent name (see `harness agents`).")],
    prompt: Annotated[str, typer.Argument(help="The task for the agent.")],
    user: Annotated[str, typer.Option(help="End-user id for memory and policy.")] = "cli",
    session: Annotated[str | None, typer.Option(help="Session id for multi-turn memory.")] = None,
    offline: Annotated[bool, typer.Option(help="Force the scripted offline model.")] = False,
    auto_approve: Annotated[bool, typer.Option(help="Approve gated tool calls.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the final run as JSON.")] = False,
) -> None:
    """Run an agent to completion and print its answer."""
    state = asyncio.run(
        _run(
            agent,
            prompt,
            user=user,
            session=session,
            offline=offline,
            auto_approve=auto_approve,
            as_json=as_json,
        )
    )
    if as_json:
        typer.echo(state.model_dump_json(indent=2, exclude={"messages", "tools"}))
    else:
        # The answer was streamed above as it was generated.
        if not state.answer:
            typer.secho("\n(no answer)", fg="red")
        typer.secho(
            f"\nstatus={state.status.value} stop_reason="
            f"{state.stop_reason.value if state.stop_reason else None} steps={state.step} "
            f"tokens={state.usage.total_tokens} cost=${state.cost_usd:.4f} "
            f"time={state.elapsed_s:.1f}s run_id={state.run_id}",
            fg="bright_black",
        )
        if state.error:
            typer.secho(f"error: {state.error}", fg="red")
    raise typer.Exit(0 if state.status == RunStatus.COMPLETED else 1)


@app.command(name="eval")
def eval_cmd(
    path: Annotated[Path, typer.Argument(help="Task file or directory.")] = Path("evals/tasks"),
    baseline: Annotated[Path | None, typer.Option(help="Fail on regression vs this.")] = None,
    write_baseline: Annotated[Path | None, typer.Option(help="Record results as baseline.")] = None,
    live: Annotated[bool, typer.Option(help="Use real models instead of scripts.")] = False,
    judge: Annotated[
        str | None, typer.Option(help="Judge model for rubrics, e.g. anthropic/claude-opus-5-5")
    ] = None,
    tolerance: Annotated[float, typer.Option(help="Allowed score drop.")] = 0.0,
    report_dir: Annotated[Path, typer.Option(help="Where to write reports.")] = Path(
        "evals/reports"
    ),
) -> None:
    """Run eval suites, write markdown/JSON reports, and gate on regressions."""
    from harness.evals import compare, load_suites, run_suites, to_baseline, write_reports

    settings = get_settings()
    judge_llm = None
    if judge:
        provider, _, model = judge.partition("/")
        judge_llm = build_provider(ModelRef(provider=provider, name=model), settings)
    results = asyncio.run(
        run_suites(load_suites(path), settings, offline=not live, judge=judge_llm)
    )
    regressions: list[str] = []
    if baseline is not None and baseline.is_file():
        regressions = compare(results, json.loads(baseline.read_text()), tolerance)
    report = write_reports(results, regressions, report_dir)
    typer.echo(report.read_text())
    if write_baseline is not None:
        write_baseline.write_text(json.dumps(to_baseline(results), indent=2) + "\n")
        typer.secho(f"baseline written to {write_baseline}", fg="green")
    failed = [t for r in results for t in r.scored if not t.passed]
    if regressions:
        typer.secho(f"{len(regressions)} regression(s) vs baseline", fg="red")
        raise typer.Exit(1)
    if failed and baseline is None:
        raise typer.Exit(1)


@app.command()
def agents() -> None:
    """List configured agents."""
    for spec in load_agents(get_settings().agents_dir).values():
        typer.echo(
            f"{spec.name:<14} {spec.model.provider}/{spec.model.name:<20} "
            f"tools={','.join(spec.tools) or '-'}  {spec.description}"
        )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("harness.api.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command(name="audit-verify")
def audit_verify(path: Annotated[Path | None, typer.Argument()] = None) -> None:
    """Verify the hash chain of a file audit log."""
    from harness.safety import FileAuditLog

    target = path or get_settings().audit_log_path
    ok, line = FileAuditLog(target).verify()
    if ok:
        typer.secho(f"{target}: chain intact", fg="green")
        return
    typer.secho(f"{target}: tampering detected at line {line}", fg="red")
    raise typer.Exit(1)


def main() -> Any:  # pragma: no cover
    return app()
