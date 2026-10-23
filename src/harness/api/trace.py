"""Run trace view: a per-step timeline built from the run's event history.

For full distributed traces (spans across LLM calls, tools and HTTP), export to an
OpenTelemetry backend (Jaeger in docker-compose); this endpoint needs no extra infra.
"""

from __future__ import annotations

from html import escape
from typing import Any

from harness.core.types import RunEvent, RunState


def build_trace(state: RunState, events: list[RunEvent]) -> dict[str, Any]:
    start = events[0].ts if events else state.created_at
    steps: dict[int, dict[str, Any]] = {}
    llm_ms = tool_ms = 0.0
    for e in events:
        if e.type == "text_delta":
            continue
        step = steps.setdefault(e.step, {"step": e.step, "events": []})
        entry = {"t_ms": round((e.ts - start).total_seconds() * 1000, 1), "type": e.type}
        entry.update({k: v for k, v in e.data.items() if k not in ("arguments",)})
        step["events"].append(entry)
        if e.type == "llm_call":
            llm_ms += float(e.data.get("latency_ms", 0))
        elif e.type == "observe":
            tool_ms += float(e.data.get("duration_ms", 0))
    return {
        "run_id": state.run_id,
        "agent": state.agent,
        "status": state.status.value,
        "stop_reason": state.stop_reason.value if state.stop_reason else None,
        "totals": {
            "steps": state.step,
            "elapsed_s": round(state.elapsed_s, 3),
            "llm_ms": round(llm_ms, 1),
            "tool_ms": round(tool_ms, 1),
            "total_tokens": state.usage.total_tokens,
            "cost_usd": round(state.cost_usd, 6),
        },
        "steps": [steps[k] for k in sorted(steps)],
    }


def render_html(trace: dict[str, Any]) -> str:
    rows = []
    for step in trace["steps"]:
        for ev in step["events"]:
            detail = ", ".join(
                f"{k}={v}"
                for k, v in ev.items()
                if k not in ("t_ms", "type") and v not in (None, "")
            )
            rows.append(
                f"<tr><td>{step['step']}</td><td>{ev['t_ms']}</td>"
                f"<td><b>{escape(ev['type'])}</b></td><td>{escape(detail[:400])}</td></tr>"
            )
    totals = " &middot; ".join(f"{escape(k)}: {escape(str(v))}" for k, v in trace["totals"].items())
    return (
        "<!doctype html><meta charset=utf-8><title>Run trace</title>"
        "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}"
        "</style>"
        f"<h1>Run {escape(trace['run_id'])}</h1>"
        f"<p>{escape(trace['agent'])} &middot; {escape(trace['status'])} "
        f"({escape(str(trace['stop_reason']))})</p><p>{totals}</p>"
        "<table><tr><th>step</th><th>t (ms)</th><th>event</th><th>detail</th></tr>"
        + "".join(rows)
        + "</table>"
    )
