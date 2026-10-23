from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.core import Agent, AgentSpec, RunState, RunStatus, load_agents
from harness.llm import FakeLLM, Message
from tests.helpers import MemoryStore, StubTools, echo


class FakeMemory:
    def __init__(self, fail_record: bool = False) -> None:
        self.recorded: list[str] = []
        self.fail_record = fail_record

    async def recall(self, *, query: str, user_id: str, k: int) -> list[str]:
        return [f"fact about {query}"][:k]

    async def session_history(self, session_id: str) -> list[Message]:
        return [Message.user("previous q"), Message.assistant("previous a")]

    async def record_run(self, state: RunState) -> None:
        if self.fail_record:
            raise RuntimeError("db down")
        self.recorded.append(state.run_id)


def _spec(**kw: object) -> AgentSpec:
    return AgentSpec.model_validate({"name": "tester", "tools": ["echo"], **kw})


async def test_new_run_assembles_prompt_with_memory_and_history() -> None:
    store = MemoryStore()
    agent = Agent(_spec(), FakeLLM(), StubTools({"echo": echo}), memory=FakeMemory(), store=store)
    state = await agent.new_run("hello", user_id="u1", session_id="ses_1")
    assert state.status == RunStatus.PENDING
    assert state.session_id == "ses_1"
    assert [t.name for t in state.tools] == ["echo"]
    assert "fact about hello" in state.messages[0].content
    assert [m.content for m in state.messages[1:]] == ["previous q", "previous a", "hello"]
    assert store.saves == 1


async def test_run_records_memory_and_tolerates_memory_failure() -> None:
    mem = FakeMemory()
    agent = Agent(_spec(), FakeLLM([FakeLLM.text("hi")]), StubTools({"echo": echo}), memory=mem)
    state = await agent.run("q")
    assert state.answer == "hi" and mem.recorded == [state.run_id]

    agent = Agent(
        _spec(),
        FakeLLM([FakeLLM.text("hi")]),
        StubTools({"echo": echo}),
        memory=FakeMemory(fail_record=True),
    )
    assert (await agent.run("q")).status == RunStatus.COMPLETED


async def test_memory_disabled_skips_recall() -> None:
    agent = Agent(
        _spec(memory={"enabled": False}), FakeLLM(), StubTools({"echo": echo}), memory=FakeMemory()
    )
    state = await agent.new_run("hello", session_id="s")
    assert "<memory>" not in state.messages[0].content
    assert len(state.messages) == 2


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"name": "Bad Name"})
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"name": "ok", "unknown_field": 1})
    spec = AgentSpec.model_validate({"name": "ok"})
    assert spec.model.name == "claude-opus-5-5"


def test_load_agents_from_directory(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: alpha\ntools: [file_read]\nlimits: {max_steps: 5}\n")
    (tmp_path / "b.yml").write_text("name: beta\n")
    specs = load_agents(tmp_path)
    assert set(specs) == {"alpha", "beta"}
    assert specs["alpha"].limits.max_steps == 5


def test_load_agents_rejects_duplicates_and_bad_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: same\n")
    (tmp_path / "b.yaml").write_text("name: same\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_agents(tmp_path)
    (tmp_path / "b.yaml").write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="invalid agent definition"):
        load_agents(tmp_path)
