from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterator

import fakeredis
import pytest

from harness.core.types import EventType, RunEvent
from harness.events import InMemoryEventBus, RedisEventBus
from harness.observability.logging import JsonFormatter, configure_logging, run_id_var
from harness.observability.metrics import LLM_CALLS, REGISTRY, RUNS, TOOL_CALLS, MetricsSink
from harness.observability.tracing import span
from harness.safety import Redactor


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


def _ev(type_: EventType, run: str = "r1", **data: object) -> RunEvent:
    return RunEvent(type=type_, run_id=run, step=1, data=dict(data))


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_event_bus_history_and_replay(backend: str, redis: fakeredis.FakeAsyncRedis) -> None:
    bus = InMemoryEventBus() if backend == "memory" else RedisEventBus(redis)
    ids = [await bus.publish(_ev(t)) for t in ("run_started", "think", "run_finished")]
    await bus.publish(_ev("run_started", run="other"))
    history = await bus.history("r1")
    assert [e.type for _, e in history] == ["run_started", "think", "run_finished"]
    after_first = await bus.history("r1", after=ids[0])
    assert [e.type for _, e in after_first] == ["think", "run_finished"]


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_event_bus_live_subscribe(backend: str, redis: fakeredis.FakeAsyncRedis) -> None:
    bus = InMemoryEventBus() if backend == "memory" else RedisEventBus(redis)
    await bus.publish(_ev("run_started"))
    received: list[str] = []

    async def consume() -> None:
        async for item in bus.subscribe("r1", idle_timeout_s=0.05):
            if item is None:
                continue
            received.append(item[1].type)
            if item[1].type == "run_finished":
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    await bus.publish(_ev("think"))
    await bus.publish(_ev("run_finished"))
    await asyncio.wait_for(task, 2)
    assert received == ["run_started", "think", "run_finished"]


async def test_subscribe_yields_idle_ticks() -> None:
    bus = InMemoryEventBus()
    agen = bus.subscribe("quiet", idle_timeout_s=0.01)
    assert await anext(agen) is None
    await agen.aclose()  # type: ignore[attr-defined]


async def test_in_memory_bus_caps_text_deltas_only() -> None:
    bus = InMemoryEventBus(max_events_per_run=2)
    for _ in range(3):
        await bus.publish(_ev("text_delta", text="x"))
    await bus.publish(_ev("run_finished"))
    types = [e.type for _, e in await bus.history("r1")]
    assert types == ["text_delta", "text_delta", "run_finished"]


async def test_metrics_sink_records_events() -> None:
    sink = MetricsSink("agent-x")
    before = (
        REGISTRY.get_sample_value(
            "harness_runs_total", {"agent": "agent-x", "stop_reason": "completed"}
        )
        or 0.0
    )
    await sink(
        _ev(
            "llm_call",
            provider="fake",
            model="m",
            finish_reason="end_turn",
            latency_ms=120,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
    )
    await sink(_ev("observe", tool="t", is_error=True, duration_ms=5))
    await sink(_ev("approval_required"))
    await sink(_ev("run_finished", stop_reason="completed"))
    await sink(_ev("run_finished", stop_reason="timeout"))
    assert RUNS.labels("agent-x", "completed")._value.get() == before + 1
    assert LLM_CALLS.labels("fake", "m", "end_turn")._value.get() >= 1
    assert TOOL_CALLS.labels("t", "error")._value.get() >= 1


def test_json_logging_redacts_and_correlates() -> None:
    configure_logging("INFO", Redactor(["topsecretvalue"]))
    root = logging.getLogger()
    stream = io.StringIO()
    root.handlers[0].stream = stream  # type: ignore[attr-defined]
    token = run_id_var.set("run_abc")
    try:
        logging.getLogger("t").info("leak topsecretvalue", extra={"api_key": "zzz", "n": 1})
    finally:
        run_id_var.reset(token)
    record = json.loads(stream.getvalue())
    assert record["run_id"] == "run_abc"
    assert record["msg"] == "leak [REDACTED]"
    assert record["api_key"] == "[REDACTED]"
    assert record["n"] == 1


def test_json_formatter_includes_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.getLogger("t").makeRecord("t", logging.ERROR, "f", 1, "failed", (), None)
        import sys

        rec.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in out["exc"]


def test_span_records_errors() -> None:
    with pytest.raises(RuntimeError), span("test.span", attr={"complex": 1}, none=None):
        raise RuntimeError("x")
