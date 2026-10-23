from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from harness.config import Settings
from harness.core.types import RunStatus, StopReason, ToolContext
from harness.llm import FakeLLM, LLMResponse, Message, ToolCall, Usage
from harness.orchestration import MAX_DEPTH, child_run_id
from harness.runtime import Harness
from harness.sandbox.local_unsafe import UnsafeLocalSandbox

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]

TEAM = {
    "lead.yaml": """
name: lead
description: supervisor
prompt: {template: supervisor@v1, instructions: Coordinate the team.}
tools: [delegate]
sub_agents: [research, review]
model: {provider: fake, name: fake-1}
""",
    "research.yaml": """
name: research
description: Finds facts.
tools: [file_read]
model: {provider: fake, name: fake-1}
""",
    "review.yaml": """
name: review
description: Reviews drafts critically.
tools: []
model: {provider: fake, name: fake-1}
""",
    "gated.yaml": """
name: gated
description: Runs code (needs approval).
tools: [python_exec]
model: {provider: fake, name: fake-1}
""",
    "boss.yaml": """
name: boss
prompt: {template: supervisor@v1}
tools: [delegate]
sub_agents: [gated, lead]
model: {provider: fake, name: fake-1}
""",
}


def _build(tmp_path: Path, llms: dict[str, Any], agents: dict[str, str] | None = None) -> Harness:
    d = tmp_path / "agents"
    d.mkdir(exist_ok=True)
    for name, body in (agents or TEAM).items():
        (d / name).write_text(body)
    settings = Settings(
        _env_file=None,
        agents_dir=d,
        policy_path=REPO / "policies" / "default.yaml",
        audit_log_path=tmp_path / "audit.log",
        workspace_root=tmp_path,
    )
    return Harness.build(settings, sandbox=UnsafeLocalSandbox(allow=True), llm_overrides=llms)


def _parallel(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant("", list(calls)),
        model="fake-1",
        provider="fake",
        finish_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


async def test_supervisor_delegates_in_parallel_and_rolls_up_cost(tmp_path: Path) -> None:
    def research_answer(msgs: Sequence[Message]) -> LLMResponse:
        # The sub-agent sees only its own task, never the supervisor's conversation.
        assert [m.role for m in msgs] == ["system", "user"]
        assert "Coordinate the team" not in msgs[0].content
        r = FakeLLM.text(f"facts about: {msgs[-1].content}")
        r.cost_usd = 0.5
        return r

    lead_llm = FakeLLM(
        [
            _parallel(
                ToolCall(id="d1", name="delegate", arguments={"agent": "research", "task": "A"}),
                ToolCall(id="d2", name="delegate", arguments={"agent": "review", "task": "B"}),
            ),
            lambda msgs: FakeLLM.text(
                "Combined: " + " | ".join(m.content for m in msgs if m.role == "tool")
            ),
        ]
    )
    harness = _build(
        tmp_path,
        {
            "lead": lead_llm,
            "research": FakeLLM([research_answer]),
            "review": FakeLLM([FakeLLM.text("looks good")]),
        },
    )
    state = await harness.run("lead", "Write a brief", user_id="u1")
    assert state.status == RunStatus.COMPLETED
    assert state.answer == "Combined: facts about: A | looks good"
    assert "## Sub-agents you can delegate to" in state.messages[0].content
    assert "- research: Finds facts." in state.messages[0].content

    children = [r for r in await harness.state.runs.list_runs() if r.parent_run_id]
    assert {c.agent for c in children} == {"research", "review"}
    assert all(c.parent_run_id == state.run_id for c in children)
    assert all(c.metadata["depth"] == 1 for c in children)
    assert all(c.metadata["owner"] == state.metadata["owner"] for c in children)
    assert all(c.user_id == "u1" for c in children)
    # Child spend rolls up into the parent.
    assert state.cost_usd == pytest.approx(0.5)
    child_tokens = sum(c.usage.total_tokens for c in children)
    assert state.usage.total_tokens >= child_tokens + 20
    await harness.aclose()


async def test_only_listed_sub_agents_are_reachable(tmp_path: Path) -> None:
    lead_llm = FakeLLM(
        [
            FakeLLM.tool_call("delegate", {"agent": "gated", "task": "x"}),
            FakeLLM.text("ok"),
        ]
    )
    harness = _build(tmp_path, {"lead": lead_llm})
    state = await harness.run("lead", "go")
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.is_error and "not a sub-agent of lead" in tool_msg.content


async def test_child_failure_and_approval_pause_are_reported(tmp_path: Path) -> None:
    boss_llm = FakeLLM(
        [
            FakeLLM.tool_call("delegate", {"agent": "gated", "task": "run code"}),
            FakeLLM.text("reported"),
        ]
    )
    gated_llm = FakeLLM([FakeLLM.tool_call("python_exec", {"code": "print(1)"})])
    harness = _build(tmp_path, {"boss": boss_llm, "gated": gated_llm})
    state = await harness.run("boss", "go")
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.is_error and "needs human approval" in tool_msg.content
    assert state.status == RunStatus.COMPLETED
    child_id = tool_msg.content.split("(run ")[1].rstrip(").").split(")")[0]
    child = await harness.get(child_id)
    assert child.status == RunStatus.AWAITING_APPROVAL


async def test_delegation_is_idempotent_per_call(tmp_path: Path) -> None:
    research = FakeLLM([FakeLLM.text("only once")])
    harness = _build(tmp_path, {"research": research, "lead": FakeLLM()})
    parent = await harness.create_run("lead", "parent task")
    ctx = ToolContext(
        run_id=parent.run_id, session_id="s", user_id="u", agent="lead", step=1, call_id="c9"
    )
    call = ToolCall(id="c9", name="delegate", arguments={"agent": "research", "task": "t"})
    first = await harness.registry.execute(call, ctx)
    again = await harness.registry.execute(call, ctx)  # e.g. replay after a crash
    assert first.content == again.content == "only once"
    assert first.metadata["child_run_id"] == child_run_id(parent.run_id, "c9")
    assert len(research.calls) == 1


async def test_depth_limit(tmp_path: Path) -> None:
    harness = _build(tmp_path, {"lead": FakeLLM()})
    parent = await harness.create_run("lead", "deep", metadata={"depth": MAX_DEPTH})
    ctx = ToolContext(
        run_id=parent.run_id, session_id="s", user_id="u", agent="lead", step=1, call_id="c"
    )
    out = await harness.registry.execute(
        ToolCall(id="c", name="delegate", arguments={"agent": "research", "task": "t"}), ctx
    )
    assert out.is_error and "depth limit" in out.content


async def test_child_errors_surface_with_stop_reason(tmp_path: Path) -> None:
    lead_llm = FakeLLM(
        [FakeLLM.tool_call("delegate", {"agent": "review", "task": "t"}), FakeLLM.text("done")]
    )
    review = FakeLLM([RuntimeError("model crashed")])
    harness = _build(tmp_path, {"lead": lead_llm, "review": review})
    state = await harness.run("lead", "go")
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "did not finish (error)" in tool_msg.content
    assert state.stop_reason == StopReason.COMPLETED


@pytest.mark.parametrize(
    ("agents", "fragment"),
    [
        ({"a.yaml": "name: a\ntools: [nope]\n"}, "unknown tool 'nope'"),
        ({"a.yaml": "name: a\ntools: [delegate]\nsub_agents: [ghost]\n"}, "unknown sub-agent"),
        ({"a.yaml": "name: a\nsub_agents: [b]\n", "b.yaml": "name: b\n"}, "not the 'delegate'"),
        ({"a.yaml": "name: a\ntools: [delegate]\nsub_agents: [a]\n"}, "cannot delegate to itself"),
    ],
)
def test_invalid_team_definitions_fail_fast(
    tmp_path: Path, agents: dict[str, str], fragment: str
) -> None:
    with pytest.raises(ValueError, match="invalid agent definitions") as exc:
        _build(tmp_path, {}, agents)
    assert fragment in str(exc.value)
