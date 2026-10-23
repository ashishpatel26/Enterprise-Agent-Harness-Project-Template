"""Agno engine: Agno drives the loop; the harness governs tools, approvals and budgets."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from harness.config import Settings
from harness.core.types import ApprovalDecision, RunState, RunStatus, StopReason
from harness.engines.agno_engine import load_native_model
from harness.llm import FakeLLM, LLMResponse, Message, ToolCall, Usage
from harness.runtime import Harness
from harness.sandbox.local_unsafe import UnsafeLocalSandbox

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _harness(tmp_path: Path, llm: FakeLLM, extra: str = "") -> Harness:
    agents = tmp_path / "agents"
    agents.mkdir(exist_ok=True)
    (agents / "a.yaml").write_text(
        "name: agno_bot\nengine: agno\nmodel: {provider: fake}\n"
        "tools: [file_read, python_exec, sql_query]\n" + extra
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "hello.txt").write_text("hello from agno land")
    settings = Settings(
        _env_file=None,
        agents_dir=agents,
        policy_path=REPO / "policies" / "default.yaml",
        audit_log_path=tmp_path / "audit.log",
        workspace_root=tmp_path,
    )
    return Harness.build(
        settings, sandbox=UnsafeLocalSandbox(allow=True), llm_overrides={"agno_bot": llm}
    )


def _events(h: Harness) -> list[str]:
    seen: list[str] = []

    async def sink(event: Any) -> None:
        seen.append(event.type)

    h.extra_sinks.append(sink)
    return seen


async def test_agno_runs_tools_through_harness_pipeline(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeLLM.tool_call("file_read", {"path": "data/hello.txt"}, call_id="r1"),
            FakeLLM.tool_call(
                "sql_query", {"database": "data/x.db", "query": "select * from customers"}
            ),
            lambda msgs: FakeLLM.text("final: " + msgs[-3].content.split("\n")[1].strip()),
        ]
    )
    h = _harness(tmp_path, llm)
    events = _events(h)
    state = await h.run("agno_bot", "read the file")
    assert state.status == RunStatus.COMPLETED, state.error
    assert state.metadata["engine"] == "agno"
    assert state.answer is not None and "hello from agno land" in state.answer
    # The harness model saw the real tool results, in order, via the Agno loop.
    tool_msgs = [m for m in llm.calls[2] if m.role == "tool"]
    assert "hello from agno land" in tool_msgs[0].content
    assert "Blocked by policy" in tool_msgs[1].content
    # The run record contains the full transcript, usage and events.
    assert [c.name for m in state.messages for c in m.tool_calls] == ["file_read", "sql_query"]
    assert state.usage.total_tokens > 0 and state.step == 3
    for e in ("run_started", "think", "llm_call", "act", "observe", "run_finished"):
        assert e in events
    stored = await h.get(state.run_id)
    assert stored.status == RunStatus.COMPLETED
    await h.aclose()


async def test_agno_max_steps_budget(tmp_path: Path) -> None:
    llm = FakeLLM([FakeLLM.tool_call("file_read", {"path": "data/hello.txt"})], repeat_last=True)
    h = _harness(tmp_path, llm, "limits: {max_steps: 2}\n")
    state = await h.run("agno_bot", "loop forever")
    assert state.stop_reason == StopReason.MAX_STEPS
    assert state.status == RunStatus.FAILED
    await h.aclose()


async def test_agno_inline_approval(tmp_path: Path) -> None:
    async def run(approved: bool) -> RunState:
        llm = FakeLLM(
            [
                FakeLLM.tool_call("python_exec", {"code": "print(6*7)"}, call_id="p"),
                lambda msgs: FakeLLM.text(next(m for m in msgs if m.role == "tool").content),
            ]
        )
        h = _harness(tmp_path, llm)

        async def reviewer(state: RunState, call: ToolCall, reason: str) -> ApprovalDecision:
            return ApprovalDecision(
                call_id=call.id, approved=approved, reviewer="carol", comment="checked"
            )

        h.inline_approver = reviewer
        state = await h.run("agno_bot", "compute")
        await h.aclose()
        return state

    approved = await run(True)
    assert approved.status == RunStatus.COMPLETED and "42" in (approved.answer or "")
    denied = await run(False)
    assert "Denied by human reviewer carol: checked" in (denied.answer or "")


async def test_agno_approval_via_api_while_waiting(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            FakeLLM.tool_call("python_exec", {"code": "print('ok')"}, call_id="p"),
            FakeLLM.text("approved and ran"),
        ]
    )
    h = _harness(tmp_path, llm)
    state = await h.create_run("agno_bot", "compute")
    task = h.start(state.run_id)
    for _ in range(200):
        current = await h.get(state.run_id)
        if current.status == RunStatus.AWAITING_APPROVAL:
            break
        await asyncio.sleep(0.02)
    call_id = current.pending_approvals[0].call.id
    await h.decide(state.run_id, call_id, approved=True, reviewer="dave")
    final = await asyncio.wait_for(task, 10)
    assert final.status == RunStatus.COMPLETED and final.answer == "approved and ran"
    await h.aclose()


async def test_agno_llm_error_is_clean_stop(tmp_path: Path) -> None:
    from harness.llm import NonRetryableLLMError

    h = _harness(tmp_path, FakeLLM([NonRetryableLLMError("bad key", provider="fake")]))
    state = await h.run("agno_bot", "x")
    assert state.status == RunStatus.FAILED
    assert state.stop_reason in (StopReason.LLM_ERROR, StopReason.ERROR)
    await h.aclose()


async def test_agno_preserves_raw_blocks_within_tool_loop(tmp_path: Path) -> None:
    first = LLMResponse(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="t1", name="file_read", arguments={"path": "data/hello.txt"})],
            raw_content=[{"type": "thinking", "thinking": "", "signature": "sig"}],
            raw_provider="anthropic",
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
        model="fake-1",
        provider="fake",
        finish_reason="tool_use",
    )
    llm = FakeLLM([first, FakeLLM.text("done")])
    h = _harness(tmp_path, llm)
    await h.run("agno_bot", "x")
    replayed = next(m for m in llm.calls[1] if m.role == "assistant" and m.tool_calls)
    assert replayed.raw_content == first.message.raw_content
    await h.aclose()


def test_native_model_loader() -> None:
    with pytest.raises(ValueError, match="invalid agno model"):
        load_native_model("nonsense")
    with pytest.raises(ModuleNotFoundError):
        load_native_model("no.such.module:Cls:model")
