from __future__ import annotations

import asyncio

import pytest

from harness.core import (
    AgentLoop,
    ApprovalDecision,
    GuardDecision,
    RunEvent,
    RunLimits,
    RunState,
    RunStatus,
    StopReason,
    ToolOutcome,
)
from harness.llm import FakeLLM, LLMResponse, Message, NonRetryableLLMError, Usage
from tests.helpers import EventLog, MemoryStore, ScriptedGuard, StubTools, echo, slow


def _state(**kw: object) -> RunState:
    return RunState(
        agent="test",
        input="q",
        messages=[Message.system("sys"), Message.user("q")],
        **kw,
    )


async def test_direct_answer_completes() -> None:
    log = EventLog()
    loop = AgentLoop(FakeLLM([FakeLLM.text("42")]), StubTools(), sinks=[log])
    state = await loop.run(_state())
    assert state.status == RunStatus.COMPLETED
    assert state.stop_reason == StopReason.COMPLETED
    assert state.answer == "42"
    assert state.step == 1
    assert log.types()[0] == "run_started" and log.types()[-1] == "run_finished"
    assert "text_delta" in log.types()


async def test_tool_call_then_answer_appends_results_in_order() -> None:
    tools = StubTools({"echo": echo})
    llm = FakeLLM(
        [
            FakeLLM.tool_call("echo", {"x": 1}, call_id="a"),
            FakeLLM.text("done"),
        ]
    )
    log = EventLog()
    state = await AgentLoop(llm, tools, sinks=[log]).run(_state())
    assert state.answer == "done"
    roles = [m.role for m in state.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert state.messages[3].tool_call_id == "a"
    for phase in ("think", "plan", "act", "observe", "reflect", "run_finished"):
        assert phase in log.types()
    # The second LLM call saw the tool result.
    assert llm.calls[1][-1].role == "tool"


async def test_parallel_tool_calls_run_concurrently() -> None:
    started: list[str] = []
    gate = asyncio.Event()

    async def wait(args: dict[str, object]) -> str:
        started.append(str(args["n"]))
        if len(started) == 2:
            gate.set()
        await asyncio.wait_for(gate.wait(), 1)
        return "ok"

    multi = LLMResponse(
        message=Message.assistant(
            "",
            [
                FakeLLM.tool_call("wait", {"n": 1}, call_id="1").message.tool_calls[0],
                FakeLLM.tool_call("wait", {"n": 2}, call_id="2").message.tool_calls[0],
            ],
        ),
        model="fake-1",
        provider="fake",
        finish_reason="tool_use",
    )
    llm = FakeLLM([multi, FakeLLM.text("both done")])
    state = await AgentLoop(llm, StubTools({"wait": wait})).run(_state())
    assert state.answer == "both done"
    assert [m.tool_call_id for m in state.messages if m.role == "tool"] == ["1", "2"]


async def test_max_steps_stops_run() -> None:
    llm = FakeLLM([FakeLLM.tool_call("echo", {})], repeat_last=True)
    state = await AgentLoop(llm, StubTools({"echo": echo}), limits=RunLimits(max_steps=3)).run(
        _state()
    )
    assert state.stop_reason == StopReason.MAX_STEPS
    assert state.status == RunStatus.FAILED
    assert state.step == 3


async def test_token_budget_stops_run() -> None:
    big = FakeLLM.tool_call("echo", {})
    big.usage = Usage(input_tokens=600, output_tokens=0)
    llm = FakeLLM([big], repeat_last=True)
    loop = AgentLoop(llm, StubTools({"echo": echo}), limits=RunLimits(max_total_tokens=1000))
    state = await loop.run(_state())
    assert state.stop_reason == StopReason.TOKEN_BUDGET
    assert state.step == 2


async def test_cost_budget_stops_run() -> None:
    costly = FakeLLM.tool_call("echo", {})
    costly.cost_usd = 0.6
    llm = FakeLLM([costly], repeat_last=True)
    loop = AgentLoop(llm, StubTools({"echo": echo}), limits=RunLimits(max_cost_usd=1.0))
    state = await loop.run(_state())
    assert state.stop_reason == StopReason.COST_BUDGET
    assert state.cost_usd == pytest.approx(1.2)


async def test_wall_clock_timeout() -> None:
    llm = FakeLLM([FakeLLM.tool_call("slow", {})])
    loop = AgentLoop(llm, StubTools({"slow": slow}), limits=RunLimits(timeout_s=0.05))
    state = await loop.run(_state())
    assert state.stop_reason == StopReason.TIMEOUT
    assert state.status == RunStatus.FAILED
    assert state.elapsed_s >= 0.05


async def test_timeout_budget_spans_resumes() -> None:
    state = _state(elapsed_s=10.0)
    loop = AgentLoop(FakeLLM([FakeLLM.text("x")]), StubTools(), limits=RunLimits(timeout_s=5))
    state = await loop.run(state)
    assert state.stop_reason == StopReason.TIMEOUT


async def test_llm_error_is_a_clean_stop() -> None:
    llm = FakeLLM([NonRetryableLLMError("bad key", provider="fake")])
    state = await AgentLoop(llm, StubTools()).run(_state())
    assert state.stop_reason == StopReason.LLM_ERROR
    assert state.error is not None and "bad key" in state.error


async def test_unexpected_exception_is_a_clean_stop() -> None:
    llm = FakeLLM([RuntimeError("bug")])
    state = await AgentLoop(llm, StubTools()).run(_state())
    assert state.stop_reason == StopReason.ERROR
    assert state.error == "RuntimeError: bug"


async def test_refusal_and_truncation() -> None:
    refusal = FakeLLM.text("cannot help")
    refusal.finish_reason = "refusal"
    state = await AgentLoop(FakeLLM([refusal]), StubTools()).run(_state())
    assert state.stop_reason == StopReason.REFUSED

    cut = FakeLLM.text("partial ans")
    cut.finish_reason = "max_tokens"
    state = await AgentLoop(FakeLLM([cut]), StubTools()).run(_state())
    assert state.stop_reason == StopReason.OUTPUT_TRUNCATED
    assert state.answer == "partial ans"


async def test_guard_deny_returns_error_to_model() -> None:
    guard = ScriptedGuard({"echo": GuardDecision(verdict="deny", reason="not allowed")})
    tools = StubTools({"echo": echo})
    llm = FakeLLM([FakeLLM.tool_call("echo", {}), FakeLLM.text("ok, skipped")])
    state = await AgentLoop(llm, tools, guard=guard).run(_state())
    assert tools.calls == []
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.is_error and "Blocked by policy: not allowed" in tool_msg.content
    assert state.answer == "ok, skipped"


async def test_approval_pause_then_approve_resumes() -> None:
    guard = ScriptedGuard({"echo": GuardDecision(verdict="require_approval", reason="dangerous")})
    tools = StubTools({"echo": echo})
    store = MemoryStore()
    log = EventLog()
    llm = FakeLLM([FakeLLM.tool_call("echo", {"v": 1}, call_id="c1"), FakeLLM.text("finished")])
    loop = AgentLoop(llm, tools, guard=guard, store=store, sinks=[log])

    state = await loop.run(_state())
    assert state.status == RunStatus.AWAITING_APPROVAL
    assert state.stop_reason == StopReason.AWAITING_APPROVAL
    assert state.pending_approvals[0].call.id == "c1"
    assert "approval_required" in log.types()
    assert tools.calls == []

    # Simulate a process restart: reload from the store, record a decision, resume.
    restored = await store.load(state.run_id)
    assert restored is not None
    restored.decisions["c1"] = ApprovalDecision(call_id="c1", approved=True, reviewer="alice")
    restored.pending_approvals = []
    final = await loop.run(restored)
    assert final.status == RunStatus.COMPLETED
    assert len(tools.calls) == 1
    assert final.decisions == {}


async def test_approval_denied_feeds_error_back() -> None:
    guard = ScriptedGuard({"echo": GuardDecision(verdict="require_approval", reason="r")})
    tools = StubTools({"echo": echo})
    llm = FakeLLM([FakeLLM.tool_call("echo", {}, call_id="c1"), FakeLLM.text("understood")])
    loop = AgentLoop(llm, tools, guard=guard)
    state = await loop.run(_state())
    state.decisions["c1"] = ApprovalDecision(
        call_id="c1", approved=False, reviewer="bob", comment="too risky"
    )
    state = await loop.run(state)
    assert tools.calls == []
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "Denied by human reviewer bob: too risky" in tool_msg.content
    assert state.answer == "understood"


async def test_resume_after_crash_skips_finished_tool_calls() -> None:
    tools = StubTools({"echo": echo})
    call_a = FakeLLM.tool_call("echo", {"n": "a"}, call_id="a").message.tool_calls[0]
    call_b = FakeLLM.tool_call("echo", {"n": "b"}, call_id="b").message.tool_calls[0]
    state = _state(status=RunStatus.RUNNING, step=1)
    state.messages.append(Message.assistant("", [call_a, call_b]))
    state.tool_results["a"] = ToolOutcome(content="cached a")  # finished before the crash

    state = await AgentLoop(FakeLLM([FakeLLM.text("ok")]), tools).run(state)
    assert [c.id for c in tools.calls] == ["b"]
    contents = [m.content for m in state.messages if m.role == "tool"]
    assert contents[0] == "cached a"
    assert state.tool_results == {}


async def test_checkpoints_are_written() -> None:
    store = MemoryStore()
    llm = FakeLLM([FakeLLM.tool_call("echo", {}), FakeLLM.text("x")])
    state = await AgentLoop(llm, StubTools({"echo": echo}), store=store).run(_state())
    assert store.saves >= 4
    restored = await store.load(state.run_id)
    assert restored is not None and restored.status == RunStatus.COMPLETED


async def test_cancellation_is_recorded_and_reraised() -> None:
    store = MemoryStore()
    loop = AgentLoop(
        FakeLLM([FakeLLM.tool_call("slow", {})]), StubTools({"slow": slow}), store=store
    )
    state = _state()
    task = asyncio.create_task(loop.run(state))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.stop_reason == StopReason.CANCELLED
    assert state.status == RunStatus.CANCELLED
    saved = await store.load(state.run_id)
    assert saved is not None and saved.status == RunStatus.CANCELLED


async def test_failing_sink_does_not_break_run() -> None:
    async def broken(_: RunEvent) -> None:
        raise RuntimeError("sink down")

    state = await AgentLoop(FakeLLM([FakeLLM.text("ok")]), StubTools(), sinks=[broken]).run(
        _state()
    )
    assert state.status == RunStatus.COMPLETED


async def test_terminal_state_is_not_rerun() -> None:
    state = _state(status=RunStatus.COMPLETED)
    llm = FakeLLM()
    assert (await AgentLoop(llm, StubTools()).run(state)) is state
    assert llm.calls == []


async def test_empty_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="no messages"):
        await AgentLoop(FakeLLM(), StubTools()).run(RunState(agent="a", input="q"))


async def test_context_compaction_happens_once_over_budget() -> None:
    log = EventLog()
    state = _state()
    for i in range(200):
        state.messages.append(Message.user(f"filler {i} " + "z" * 200))
    llm = FakeLLM([FakeLLM.text("done")])
    loop = AgentLoop(llm, StubTools(), limits=RunLimits(max_context_tokens=2_000), sinks=[log])
    state = await loop.run(state)
    assert "compacted" in log.types()
    assert state.status == RunStatus.COMPLETED
