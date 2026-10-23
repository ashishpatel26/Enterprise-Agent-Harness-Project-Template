"""Run state persistence and run leases.

A run is checkpointed after every step and every tool result. Leases make sure only one
worker drives a given run at a time; a worker that dies stops renewing its lease, so
another worker can pick the run up (`list_runs(RunStatus.RUNNING)` + `acquire`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any, Protocol

import redis.asyncio as aioredis

from harness.core.types import RunState, RunStatus


class StateStore(Protocol):
    async def save(self, state: RunState) -> None: ...

    async def load(self, run_id: str) -> RunState | None: ...

    async def list_runs(
        self, *, status: RunStatus | None = None, session_id: str | None = None, limit: int = 100
    ) -> list[RunState]: ...

    async def delete(self, run_id: str) -> None: ...


class LeaseManager(Protocol):
    async def acquire(self, run_id: str, owner: str, ttl_s: float) -> bool: ...

    async def renew(self, run_id: str, owner: str, ttl_s: float) -> bool: ...

    async def release(self, run_id: str, owner: str) -> None: ...


# ---- in-memory (single process: dev and tests) ----------------------------------


class InMemoryStateStore:
    def __init__(self) -> None:
        self._runs: dict[str, str] = {}

    async def save(self, state: RunState) -> None:
        # Serialize on save so later in-place mutation can't alter the checkpoint.
        self._runs[state.run_id] = state.model_dump_json()

    async def load(self, run_id: str) -> RunState | None:
        raw = self._runs.get(run_id)
        return RunState.model_validate_json(raw) if raw else None

    async def list_runs(
        self, *, status: RunStatus | None = None, session_id: str | None = None, limit: int = 100
    ) -> list[RunState]:
        runs = [RunState.model_validate_json(r) for r in self._runs.values()]
        runs = [
            r
            for r in runs
            if (status is None or r.status == status)
            and (session_id is None or r.session_id == session_id)
        ]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]

    async def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


class InMemoryLeases:
    def __init__(self) -> None:
        self._leases: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, run_id: str, owner: str, ttl_s: float) -> bool:
        async with self._lock:
            current = self._leases.get(run_id)
            now = time.monotonic()
            if current and current[0] != owner and current[1] > now:
                return False
            self._leases[run_id] = (owner, now + ttl_s)
            return True

    async def renew(self, run_id: str, owner: str, ttl_s: float) -> bool:
        async with self._lock:
            current = self._leases.get(run_id)
            if not current or current[0] != owner:
                return False
            self._leases[run_id] = (owner, time.monotonic() + ttl_s)
            return True

    async def release(self, run_id: str, owner: str) -> None:
        async with self._lock:
            current = self._leases.get(run_id)
            if current and current[0] == owner:
                del self._leases[run_id]


# ---- Redis (multi-process / multi-node) -----------------------------------------


def make_redis(url: str, *, timeout_s: float = 5.0) -> aioredis.Redis:
    """Redis client with explicit socket timeouts: no call can hang forever."""
    client: aioredis.Redis = aioredis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=timeout_s,
        socket_connect_timeout=timeout_s,
        health_check_interval=30,
    )
    return client


class RedisStateStore:
    """Run JSON at `{p}:run:{id}`; indexes: per-status sets, per-session sorted sets,
    and a global sorted set by creation time. Terminal runs expire after `ttl_s`."""

    def __init__(
        self, redis: aioredis.Redis, *, prefix: str = "harness", terminal_ttl_s: int = 30 * 86400
    ) -> None:
        self.r = redis
        self.p = prefix
        self.terminal_ttl_s = terminal_ttl_s

    def _key(self, run_id: str) -> str:
        return f"{self.p}:run:{run_id}"

    def _status_key(self, status: RunStatus) -> str:
        return f"{self.p}:runs:status:{status.value}"

    async def save(self, state: RunState) -> None:
        score = state.created_at.timestamp()
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.set(self._key(state.run_id), state.model_dump_json())
            for status in RunStatus:
                if status != state.status:
                    pipe.srem(self._status_key(status), state.run_id)
            pipe.sadd(self._status_key(state.status), state.run_id)
            pipe.zadd(f"{self.p}:session:{state.session_id}:runs", {state.run_id: score})
            pipe.zadd(f"{self.p}:runs:all", {state.run_id: score})
            if state.status.terminal:
                pipe.expire(self._key(state.run_id), self.terminal_ttl_s)
            await pipe.execute()

    async def load(self, run_id: str) -> RunState | None:
        raw = await self.r.get(self._key(run_id))
        return RunState.model_validate_json(raw) if raw else None

    async def _load_many(self, ids: Sequence[Any]) -> list[RunState]:
        if not ids:
            return []
        raws = await self.r.mget([self._key(i) for i in ids])
        return [RunState.model_validate_json(r) for r in raws if r]

    async def list_runs(
        self, *, status: RunStatus | None = None, session_id: str | None = None, limit: int = 100
    ) -> list[RunState]:
        if session_id is not None:
            ids = await self.r.zrevrange(f"{self.p}:session:{session_id}:runs", 0, -1)
        elif status is not None:
            ids = list(await self.r.smembers(self._status_key(status)))
        else:
            ids = await self.r.zrevrange(f"{self.p}:runs:all", 0, limit - 1)
        runs = await self._load_many(ids)
        if status is not None:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]

    async def delete(self, run_id: str) -> None:
        state = await self.load(run_id)
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.delete(self._key(run_id))
            for status in RunStatus:
                pipe.srem(self._status_key(status), run_id)
            pipe.zrem(f"{self.p}:runs:all", run_id)
            if state:
                pipe.zrem(f"{self.p}:session:{state.session_id}:runs", run_id)
            await pipe.execute()


class RedisLeases:
    """`SET NX PX` leases; renew/release check ownership inside WATCH transactions."""

    def __init__(self, redis: aioredis.Redis, *, prefix: str = "harness") -> None:
        self.r = redis
        self.p = prefix

    def _key(self, run_id: str) -> str:
        return f"{self.p}:lease:{run_id}"

    async def acquire(self, run_id: str, owner: str, ttl_s: float) -> bool:
        key = self._key(run_id)
        if await self.r.set(key, owner, nx=True, px=int(ttl_s * 1000)):
            return True
        # Re-entrant for the current owner.
        return await self.renew(run_id, owner, ttl_s)

    async def _if_owner(self, run_id: str, owner: str, ttl_s: float | None) -> bool:
        key = self._key(run_id)
        async with self.r.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.get(key) != owner:
                    await pipe.unwatch()  # type: ignore[no-untyped-call]
                    return False
                pipe.multi()  # type: ignore[no-untyped-call]
                if ttl_s is None:
                    pipe.delete(key)
                else:
                    pipe.pexpire(key, int(ttl_s * 1000))
                await pipe.execute()
                return True
            except aioredis.WatchError:
                return False

    async def renew(self, run_id: str, owner: str, ttl_s: float) -> bool:
        return await self._if_owner(run_id, owner, ttl_s)

    async def release(self, run_id: str, owner: str) -> None:
        await self._if_owner(run_id, owner, None)
