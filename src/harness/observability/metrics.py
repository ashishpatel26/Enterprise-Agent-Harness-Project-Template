"""Prometheus metrics, fed from run events."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from harness.core.types import RunEvent

REGISTRY = CollectorRegistry(auto_describe=True)

LLM_CALLS = Counter(
    "harness_llm_calls_total",
    "LLM calls",
    ["provider", "model", "finish_reason"],
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "harness_llm_latency_seconds",
    "LLM call latency",
    ["provider", "model"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160),
    registry=REGISTRY,
)
TOKENS = Counter("harness_tokens_total", "Tokens processed", ["model", "kind"], registry=REGISTRY)
COST = Counter(
    "harness_cost_usd_total", "Model spend in USD", ["agent", "model"], registry=REGISTRY
)
TOOL_CALLS = Counter(
    "harness_tool_calls_total", "Tool calls", ["tool", "outcome"], registry=REGISTRY
)
TOOL_LATENCY = Histogram(
    "harness_tool_latency_seconds",
    "Tool execution latency",
    ["tool"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)
RUNS = Counter("harness_runs_total", "Finished runs", ["agent", "stop_reason"], registry=REGISTRY)
RUN_DURATION = Histogram(
    "harness_run_duration_seconds",
    "Run wall-clock time (per execution segment)",
    ["agent"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
    registry=REGISTRY,
)
RUNS_ACTIVE = Gauge("harness_runs_active", "Runs executing in this process", registry=REGISTRY)
APPROVALS = Counter(
    "harness_approvals_total", "Approval requests and decisions", ["event"], registry=REGISTRY
)
ERRORS = Counter("harness_errors_total", "Run-level errors", ["kind"], registry=REGISTRY)


class MetricsSink:
    """EventSink that turns run events into metrics. `agent` labels cost and runs."""

    def __init__(self, agent: str) -> None:
        self.agent = agent

    async def __call__(self, event: RunEvent) -> None:
        d = event.data
        if event.type == "llm_call":
            model, provider = str(d.get("model")), str(d.get("provider"))
            LLM_CALLS.labels(provider, model, str(d.get("finish_reason"))).inc()
            LLM_LATENCY.labels(provider, model).observe(float(d.get("latency_ms", 0)) / 1000)
            for kind in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            ):
                TOKENS.labels(model, kind.removesuffix("_tokens")).inc(int(d.get(kind, 0)))
            COST.labels(self.agent, model).inc(float(d.get("cost_usd", 0.0)))
        elif event.type == "observe":
            tool = str(d.get("tool"))
            TOOL_CALLS.labels(tool, "error" if d.get("is_error") else "ok").inc()
            TOOL_LATENCY.labels(tool).observe(float(d.get("duration_ms", 0)) / 1000)
        elif event.type == "approval_required":
            APPROVALS.labels("requested").inc()
        elif event.type == "run_finished":
            reason = str(d.get("stop_reason"))
            RUNS.labels(self.agent, reason).inc()
            if reason in ("error", "llm_error", "timeout"):
                ERRORS.labels(reason).inc()
