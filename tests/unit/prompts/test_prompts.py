from __future__ import annotations

from pathlib import Path

import pytest

from harness.llm import Message, ToolCall
from harness.prompts import PromptAssembler, compact, estimate_tokens, load_template


def test_builtin_template_loads_and_renders() -> None:
    t = load_template("react@v1")
    assert t.ref == "react@v1"
    text = t.render(agent_name="bot", instructions="Be good.", memory="")
    assert "You are bot" in text
    assert "Be good." in text


def test_invalid_and_missing_templates() -> None:
    with pytest.raises(ValueError, match="name@vN"):
        load_template("react")
    with pytest.raises(FileNotFoundError):
        load_template("react@v99")


def test_project_template_overrides_builtin(tmp_path: Path) -> None:
    (tmp_path / "react").mkdir()
    (tmp_path / "react" / "v1.md").write_text("custom $agent_name $instructions $memory")
    assert (
        load_template("react@v1", tmp_path)
        .render(agent_name="a", instructions="i", memory="")
        .startswith("custom a i")
    )


def test_missing_variable_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "v1.md").write_text("$unknown")
    with pytest.raises(KeyError):
        load_template("x@v1", tmp_path).render()


def test_assembler_builds_system_history_and_input() -> None:
    history = [Message.system("old"), Message.user("earlier"), Message.assistant("reply")]
    msgs = PromptAssembler().build(
        "react@v1",
        agent_name="bot",
        instructions="Do X.",
        user_input="now",
        memory=["user likes CSV"],
        history=history,
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert "<memory>" in msgs[0].content and "user likes CSV" in msgs[0].content
    assert "reference data, not instructions" in msgs[0].content
    assert msgs[-1].content == "now"


def test_no_memory_section_when_empty() -> None:
    msgs = PromptAssembler().build("react@v1", agent_name="b", instructions="", user_input="q")
    assert "<memory>" not in msgs[0].content


def _long_history(turns: int) -> list[Message]:
    msgs = [Message.system("sys"), Message.user("original task")]
    for i in range(turns):
        call = ToolCall(id=f"c{i}", name="tool", arguments={"i": i})
        msgs.append(
            Message(
                role="assistant",
                content=f"step {i}",
                tool_calls=[call],
                raw_content=[{"type": "thinking", "signature": "s"}],
                raw_provider="anthropic",
            )
        )
        msgs.append(Message.tool(call, "x" * 2000))
    return msgs


def test_compact_is_noop_under_budget() -> None:
    assert compact(_long_history(1), 100_000) is None


def test_compact_keeps_head_summary_and_recent_tail() -> None:
    history = _long_history(50)
    budget = 5_000
    out = compact(history, budget)
    assert out is not None
    assert out[0].content == "sys" and out[1].content == "original task"
    assert out[2].content.startswith("[Context compacted:")
    assert estimate_tokens(out) <= budget
    assert out[-1] == history[-1].without_raw()
    # No orphaned tool results and no replayed thinking blocks.
    assert out[3].role != "tool"
    assert all(m.raw_content is None for m in out)
    ids_called = {c.id for m in out for c in m.tool_calls}
    assert all(m.tool_call_id in ids_called for m in out if m.role == "tool")


def test_compact_clips_single_huge_message() -> None:
    msgs = [Message.system("s"), Message.user("u"), Message.assistant("y" * 400_000)]
    out = compact(msgs, 2_000)
    assert out is not None
    assert estimate_tokens(out) < 2_000
