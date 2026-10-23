"""Runtime behaviour: leases, crash recovery, cancellation, CLI-style runs."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from harness.core.types import RunStatus, StopReason
from harness.llm import FakeLLM
from harness.runtime import RunConflictError
from harness.safety import InputValidationError

pytestmark = pytest.mark.integration


async def test_run_to_completion(make_harness: Any) -> None:
    llm = FakeLLM([FakeLLM.text("hi there")])
    h = make_harness({"assistant": llm})
    state = await h.run("assistant", "hello", user_id="u1", session_id="ses_1")
    assert state.status == RunStatus.COMPLETED
    # Memory recorded the exchange for the session.
    history = await h.memory.session_history("ses_1")
    assert [m.content for m in history] == ["hello", "hi there"]
    # The second run in the session sees the history.
    llm2 = FakeLLM([FakeLLM.text("again")])
    h.llm_overrides["assistant"] = llm2
    h._agents.clear()
    await h.run("assistant", "and now?", user_id="u1", session_id="ses_1")
    assert [m.content for m in llm2.calls[0][1:]] == ["hello", "hi there", "and now?"]
    await h.aclose()


async def test_input_is_validated(make_harness: Any) -> None:
    h = make_harness({})
    with pytest.raises(InputValidationError):
        await h.create_run("assistant", "\x00\x00")
    with pytest.raises(KeyError, match="unknown agent"):
        await h.create_run("ghost", "hi")


async def test_lease_blocks_concurrent_execution(make_harness: Any) -> None:
    h = make_harness({"assistant": FakeLLM([FakeLLM.text("x")])})
    state = await h.create_run("assistant", "hi")
    assert await h.state.leases.acquire(state.run_id, "other-worker", 30)
    with pytest.raises(RunConflictError):
        await h.execute(state.run_id)


async def test_recover_resumes_interrupted_runs(make_harness: Any) -> None:
    h = make_harness({"assistant": FakeLLM([FakeLLM.text("recovered")])})
    state = await h.create_run("assistant", "hi")
    # Simulate a worker that died mid-run: status RUNNING, no lease held.
    state.status = RunStatus.RUNNING
    await h.state.runs.save(state)
    resumed = await h.recover()
    assert resumed == [state.run_id]
    await asyncio.gather(*h._tasks.values())
    final = await h.get(state.run_id)
    assert final.status == RunStatus.COMPLETED and final.answer == "recovered"


async def test_recover_skips_runs_held_by_live_workers(make_harness: Any) -> None:
    h = make_harness({"assistant": FakeLLM([FakeLLM.text("x")])})
    state = await h.create_run("assistant", "hi")
    state.status = RunStatus.RUNNING
    await h.state.runs.save(state)
    await h.state.leases.acquire(state.run_id, "live-worker", 30)
    assert await h.recover() == []


async def test_cancel_running_task(make_harness: Any) -> None:
    class Hanging(FakeLLM):
        async def chat(self, messages: Any, tools: Any = (), **kw: Any) -> Any:
            await asyncio.sleep(10)

    h = make_harness({"assistant": Hanging()})
    state = await h.create_run("assistant", "hi")
    h.start(state.run_id)
    await asyncio.sleep(0.05)
    cancelled = await h.cancel(state.run_id)
    assert cancelled.status == RunStatus.CANCELLED
    assert cancelled.stop_reason == StopReason.CANCELLED
    # Cancelling a finished run is a no-op.
    assert (await h.cancel(state.run_id)).status == RunStatus.CANCELLED


async def test_cancel_running_elsewhere_conflicts(make_harness: Any) -> None:
    h = make_harness({})
    state = await h.create_run("assistant", "hi")
    state.status = RunStatus.RUNNING
    await h.state.runs.save(state)
    with pytest.raises(RunConflictError):
        await h.cancel(state.run_id)
