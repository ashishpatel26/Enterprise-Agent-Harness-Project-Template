from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import fakeredis
import pytest

from harness.config import Settings
from harness.core import RunState, RunStatus
from harness.core.types import StopReason
from harness.llm import Message, ToolCall
from harness.memory import (
    Episode,
    HashingEmbedder,
    InMemoryEpisodicMemory,
    InMemorySessionMemory,
    InMemoryVectorMemory,
    MemoryItem,
    MemoryManager,
    PgVectorMemory,
    RedisEpisodicMemory,
    RedisSessionMemory,
    build_memory,
)
from harness.memory.embeddings import cosine


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    e = HashingEmbedder(64)
    [a1], [a2] = (
        await e.embed(["Quarterly revenue by region"]),
        await e.embed(["Quarterly revenue by region"]),
    )
    assert a1 == a2
    assert abs(sum(v * v for v in a1) - 1.0) < 1e-9
    [empty] = await e.embed([""])
    assert empty == [0.0] * 64
    assert cosine(empty, a1) == 0.0


async def test_vector_memory_ranks_relevant_items_and_scopes_by_user() -> None:
    mem = InMemoryVectorMemory(HashingEmbedder(256))
    await mem.add(MemoryItem(user_id="u1", content="The West region has the highest revenue"))
    await mem.add(MemoryItem(user_id="u1", content="Favourite colour is green"))
    await mem.add(MemoryItem(user_id="u2", content="West region revenue secret of another user"))
    hits = await mem.search("u1", "which region has the highest revenue", k=5)
    assert hits[0].content.startswith("The West region")
    assert all(h.user_id == "u1" for h in hits)
    assert await mem.search("u1", "revenue", k=5, min_score=0.99) == []
    assert await mem.delete_user("u1") == 2
    assert await mem.search("u1", "revenue", k=5) == []


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_session_memory_caps_and_roundtrips(
    backend: str, redis: fakeredis.FakeAsyncRedis
) -> None:
    mem = (
        InMemorySessionMemory(max_messages=3)
        if backend == "memory"
        else RedisSessionMemory(redis, max_messages=3)
    )
    await mem.append("s", [Message.user(str(i)) for i in range(5)])
    await mem.append("s", [])
    assert [m.content for m in await mem.get("s")] == ["2", "3", "4"]
    await mem.clear("s")
    assert await mem.get("s") == []


def _finished(user: str = "u1", status: RunStatus = RunStatus.COMPLETED) -> RunState:
    call = ToolCall(id="c", name="python_exec", arguments={})
    return RunState(
        agent="analyst",
        user_id=user,
        session_id="ses",
        input="Which region sells most?",
        answer="West leads with 31% of revenue." if status == RunStatus.COMPLETED else None,
        status=status,
        stop_reason=StopReason.COMPLETED if status == RunStatus.COMPLETED else StopReason.ERROR,
        step=2,
        messages=[
            Message.user("Which region sells most?"),
            Message.assistant("", [call]),
            Message.tool(call, "boom", is_error=True),
        ],
    )


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_episodic_memory(backend: str, redis: fakeredis.FakeAsyncRedis) -> None:
    mem = InMemoryEpisodicMemory() if backend == "memory" else RedisEpisodicMemory(redis)
    ep = Episode.from_state(_finished())
    assert ep.tools_used == ["python_exec"] and ep.tool_errors == 1
    await mem.record(ep)
    await mem.record(Episode.from_state(_finished(user="u2")))
    recent = await mem.recent("u1")
    assert [e.run_id for e in recent] == [ep.run_id]


def _manager() -> MemoryManager:
    return MemoryManager(
        InMemorySessionMemory(), InMemoryVectorMemory(HashingEmbedder()), InMemoryEpisodicMemory()
    )


async def test_manager_records_and_recalls() -> None:
    mgr = _manager()
    await mgr.record_run(_finished())
    recalled = await mgr.recall(query="which region sells the most", user_id="u1", k=3)
    assert recalled and "West leads" in recalled[0]
    history = await mgr.session_history("ses")
    assert [m.role for m in history] == ["user", "assistant"]
    assert await mgr.recall(query="region", user_id="someone-else", k=3) == []


async def test_failed_runs_only_go_to_episodic() -> None:
    mgr = _manager()
    await mgr.record_run(_finished(status=RunStatus.FAILED))
    assert await mgr.session_history("ses") == []
    assert len(await mgr.episodic.recent("u1")) == 1


async def test_recall_degrades_on_backend_failure_and_timeout() -> None:
    class Broken(InMemoryVectorMemory):
        async def search(self, *a: object, **k: object) -> list[MemoryItem]:
            raise ConnectionError("db down")

    class Slow(InMemorySessionMemory):
        async def get(self, session_id: str) -> list[Message]:
            await asyncio.sleep(5)
            return []

    mgr = MemoryManager(Slow(), Broken(HashingEmbedder()), InMemoryEpisodicMemory(), timeout_s=0.05)
    assert await mgr.recall(query="x", user_id="u", k=3) == []
    assert await mgr.session_history("s") == []


async def test_remember_truncates() -> None:
    mgr = _manager()
    await mgr.remember("u1", "x" * 5000, source="test")
    [item] = await mgr.long_term.search("u1", "x", k=1)
    assert len(item.content) == 1000 and item.metadata == {"source": "test"}


def test_build_memory_backends() -> None:
    mem = build_memory(Settings(_env_file=None))
    assert isinstance(mem.long_term, InMemoryVectorMemory)
    red = build_memory(
        Settings(_env_file=None, storage_backend="redis", database_url="postgresql://x/y")
    )
    assert isinstance(red.long_term, PgVectorMemory)
    assert isinstance(red.short_term, RedisSessionMemory)


def test_pgvector_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError, match="identifier"):
        PgVectorMemory("postgresql://x/y", HashingEmbedder(), table="x; DROP TABLE y")


@pytest.mark.postgres
@pytest.mark.skipif(
    not os.environ.get("HARNESS_TEST_DATABASE_URL"), reason="HARNESS_TEST_DATABASE_URL not set"
)
async def test_pgvector_end_to_end() -> None:
    mem = PgVectorMemory(
        os.environ["HARNESS_TEST_DATABASE_URL"], HashingEmbedder(64), table="test_memories"
    )
    try:
        await mem.ensure_schema()
        await mem.delete_user("pg-user")
        await mem.add(MemoryItem(user_id="pg-user", content="West region revenue is highest"))
        await mem.add(MemoryItem(user_id="pg-user", content="Unrelated note about colours"))
        hits = await mem.search("pg-user", "highest revenue region", k=2)
        assert hits[0].content.startswith("West")
        assert hits[0].score is not None and hits[0].score > hits[1].score  # type: ignore[operator]
        assert await mem.delete_user("pg-user") == 2
    finally:
        await mem.aclose()
