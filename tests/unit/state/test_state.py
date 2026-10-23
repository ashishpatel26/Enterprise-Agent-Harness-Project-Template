from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis
import pytest

from harness.config import Settings
from harness.core import AgentLoop, RunState, RunStatus, ToolContext
from harness.core.types import StopReason, ToolOutcome
from harness.llm import FakeLLM, Message, ToolCall
from harness.state import (
    IdempotentExecutor,
    InMemoryIdempotencyStore,
    InMemoryLeases,
    InMemoryStateStore,
    RedisIdempotencyStore,
    RedisLeases,
    RedisStateStore,
    build_state_backends,
)
from harness.tools import Permission, ToolRegistry


def _state(**kw: object) -> RunState:
    return RunState(agent="a", input="q", messages=[Message.user("q")], **kw)


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest, redis: fakeredis.FakeAsyncRedis) -> object:
    return InMemoryStateStore() if request.param == "memory" else RedisStateStore(redis)


@pytest.fixture(params=["memory", "redis"])
def leases(request: pytest.FixtureRequest, redis: fakeredis.FakeAsyncRedis) -> object:
    return InMemoryLeases() if request.param == "memory" else RedisLeases(redis)


async def test_save_load_roundtrip(store: InMemoryStateStore) -> None:
    s = _state(session_id="ses_1")
    await store.save(s)
    loaded = await store.load(s.run_id)
    assert loaded == s
    assert await store.load("missing") is None


async def test_checkpoint_is_a_snapshot(store: InMemoryStateStore) -> None:
    s = _state()
    await store.save(s)
    s.messages.append(Message.user("mutated after save"))
    loaded = await store.load(s.run_id)
    assert loaded is not None and len(loaded.messages) == 1


async def test_list_by_status_and_session(store: InMemoryStateStore) -> None:
    a = _state(session_id="s1", status=RunStatus.RUNNING)
    b = _state(session_id="s1", status=RunStatus.COMPLETED)
    c = _state(session_id="s2", status=RunStatus.RUNNING)
    for s in (a, b, c):
        await store.save(s)
    running = {r.run_id for r in await store.list_runs(status=RunStatus.RUNNING)}
    assert running == {a.run_id, c.run_id}
    assert {r.run_id for r in await store.list_runs(session_id="s1")} == {a.run_id, b.run_id}
    assert len(await store.list_runs(limit=2)) == 2

    # Status transitions move the run between indexes.
    a.status = RunStatus.COMPLETED
    await store.save(a)
    assert {r.run_id for r in await store.list_runs(status=RunStatus.RUNNING)} == {c.run_id}

    await store.delete(a.run_id)
    assert await store.load(a.run_id) is None
    assert a.run_id not in {r.run_id for r in await store.list_runs()}


async def test_terminal_runs_get_ttl(redis: fakeredis.FakeAsyncRedis) -> None:
    store = RedisStateStore(redis, terminal_ttl_s=100)
    s = _state(status=RunStatus.COMPLETED)
    await store.save(s)
    assert 0 < await redis.ttl(f"harness:run:{s.run_id}") <= 100


async def test_leases_are_exclusive_reentrant_and_releasable(leases: InMemoryLeases) -> None:
    assert await leases.acquire("r1", "w1", 30)
    assert not await leases.acquire("r1", "w2", 30)
    assert await leases.acquire("r1", "w1", 30)  # re-entrant
    assert await leases.renew("r1", "w1", 30)
    assert not await leases.renew("r1", "w2", 30)
    await leases.release("r1", "w2")  # not the owner: no effect
    assert not await leases.acquire("r1", "w2", 30)
    await leases.release("r1", "w1")
    assert await leases.acquire("r1", "w2", 30)


async def test_expired_lease_can_be_taken_over() -> None:
    leases = InMemoryLeases()
    assert await leases.acquire("r1", "dead-worker", 0.0)
    assert await leases.acquire("r1", "w2", 30)


# ---- idempotency ------------------------------------------------------------------


def _side_effect_registry(counter: list[int]) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(permission=Permission.WRITE)
    async def send(to: str, ctx: ToolContext) -> str:
        """Send something (side effect)."""
        counter.append(1)
        return f"sent to {to} key={ctx.idempotency_key}"

    @reg.tool()
    async def read() -> str:
        """Read-only."""
        counter.append(1)
        return "data"

    return reg


CTX = ToolContext(run_id="r", session_id="s", user_id="u", agent="a", step=1, call_id="c1")


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_non_idempotent_tool_runs_at_most_once(
    backend: str, redis: fakeredis.FakeAsyncRedis
) -> None:
    counter: list[int] = []
    store = InMemoryIdempotencyStore() if backend == "memory" else RedisIdempotencyStore(redis)
    ex = IdempotentExecutor(_side_effect_registry(counter), store)
    call = ToolCall(id="c1", name="send", arguments={"to": "bob"})
    first = await ex.execute(call, CTX)
    again = await ex.execute(call, CTX)
    assert len(counter) == 1
    assert first.content == again.content == "sent to bob key=r:c1"
    assert again.metadata["idempotency"] == "replayed"


async def test_interrupted_side_effect_is_not_rerun() -> None:
    counter: list[int] = []
    store = InMemoryIdempotencyStore()
    await store.begin("r:c1")  # crashed mid-call: marker, no result
    ex = IdempotentExecutor(_side_effect_registry(counter), store)
    out = await ex.execute(ToolCall(id="c1", name="send", arguments={"to": "bob"}), CTX)
    assert out.is_error and "outcome is unknown" in out.content
    assert counter == []


async def test_idempotent_tools_bypass_the_store() -> None:
    counter: list[int] = []
    ex = IdempotentExecutor(_side_effect_registry(counter), InMemoryIdempotencyStore())
    for _ in range(2):
        await ex.execute(ToolCall(id="c1", name="read", arguments={}), CTX)
    assert len(counter) == 2
    assert [s.name for s in ex.specs()] == ["read", "send"]


# ---- crash recovery end to end ----------------------------------------------------


async def test_run_resumes_from_checkpoint_after_crash(redis: fakeredis.FakeAsyncRedis) -> None:
    """Worker 1 dies after the tool result is checkpointed; worker 2 finishes the run
    from the stored state without re-executing the tool."""
    counter: list[int] = []
    reg = _side_effect_registry(counter)
    store = RedisStateStore(redis)
    executor = IdempotentExecutor(reg, RedisIdempotencyStore(redis))

    crash = RuntimeError("worker killed")
    llm1 = FakeLLM([FakeLLM.tool_call("send", {"to": "bob"}, call_id="c1"), crash])
    state = _state()
    state.messages.insert(0, Message.system("sys"))
    state.tools = reg.specs()
    crashed = await AgentLoop(llm1, executor, store=store).run(state)
    assert crashed.stop_reason == StopReason.ERROR
    assert len(counter) == 1

    # Recovery: reload, reset to running and drive with a fresh worker.
    restored = await store.load(state.run_id)
    assert restored is not None
    restored.status = RunStatus.RUNNING
    llm2 = FakeLLM([FakeLLM.text("sent!")])
    final = await AgentLoop(llm2, executor, store=store).run(restored)
    assert final.status == RunStatus.COMPLETED
    assert final.answer == "sent!"
    assert len(counter) == 1
    assert [m.role for m in llm2.calls[0]][-2:] == ["assistant", "tool"]


async def test_resume_mid_act_uses_tool_results_scratchpad() -> None:
    state = _state(status=RunStatus.RUNNING, step=1)
    call = ToolCall(id="c1", name="send", arguments={"to": "x"})
    state.messages.append(Message.assistant("", [call]))
    state.tool_results["c1"] = ToolOutcome(content="already sent")
    counter: list[int] = []
    final = await AgentLoop(
        FakeLLM([FakeLLM.text("ok")]),
        IdempotentExecutor(_side_effect_registry(counter), InMemoryIdempotencyStore()),
    ).run(state)
    assert counter == []
    assert final.messages[-2].content == "already sent"


def test_build_state_backends() -> None:
    mem = build_state_backends(Settings(_env_file=None))
    assert isinstance(mem.runs, InMemoryStateStore)
    red = build_state_backends(Settings(_env_file=None, storage_backend="redis"))
    assert isinstance(red.runs, RedisStateStore)
    assert isinstance(red.leases, RedisLeases)
