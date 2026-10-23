"""Run event bus: durable per-run event history plus live fan-out for SSE.

Each event gets a monotonically increasing id per run, so a client that reconnects with
`Last-Event-ID` gets exactly the events it missed. The Redis implementation uses one
Redis Stream per run, which works across API replicas: the worker executing a run
publishes, and any replica can serve the stream.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Protocol

import redis.asyncio as aioredis

from harness.core.types import RunEvent

TERMINAL_EVENTS = {"run_finished"}


class EventBus(Protocol):
    async def publish(self, event: RunEvent) -> str: ...

    async def history(
        self, run_id: str, after: str | None = None
    ) -> list[tuple[str, RunEvent]]: ...

    def subscribe(
        self, run_id: str, after: str | None = None, *, idle_timeout_s: float = 15.0
    ) -> AsyncIterator[tuple[str, RunEvent] | None]:
        """Yield (id, event) as they arrive; yield None after `idle_timeout_s` without
        events (lets the caller send keep-alives or check for disconnects)."""
        ...


class InMemoryEventBus:
    def __init__(self, max_events_per_run: int = 5000) -> None:
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._cond = asyncio.Condition()
        self.max_events = max_events_per_run

    async def publish(self, event: RunEvent) -> str:
        async with self._cond:
            events = self._events[event.run_id]
            # text deltas beyond the cap are dropped; structural events are always kept
            if len(events) >= self.max_events and event.type == "text_delta":
                return str(len(events))
            events.append(event)
            self._cond.notify_all()
            return str(len(events))

    def _has_more(self, run_id: str, seen: int) -> bool:
        return len(self._events.get(run_id, [])) > seen

    async def history(self, run_id: str, after: str | None = None) -> list[tuple[str, RunEvent]]:
        start = int(after) if after else 0
        events = self._events.get(run_id, [])
        return [(str(i + 1), e) for i, e in enumerate(events) if i + 1 > start]

    async def subscribe(
        self, run_id: str, after: str | None = None, *, idle_timeout_s: float = 15.0
    ) -> AsyncIterator[tuple[str, RunEvent] | None]:
        cursor = int(after) if after else 0
        while True:
            batch = await self.history(run_id, str(cursor))
            if not batch:
                timed_out = False
                async with self._cond:
                    try:
                        await asyncio.wait_for(
                            self._cond.wait_for(partial(self._has_more, run_id, cursor)),
                            idle_timeout_s,
                        )
                    except TimeoutError:
                        timed_out = True
                # Yield outside the lock so a slow consumer never blocks publishers.
                if timed_out:
                    yield None
                continue
            for event_id, event in batch:
                cursor = int(event_id)
                yield event_id, event


class RedisEventBus:
    def __init__(
        self,
        redis: aioredis.Redis,
        *,
        prefix: str = "harness",
        max_events_per_run: int = 5000,
        ttl_s: int = 7 * 86400,
    ) -> None:
        self.r = redis
        self.p = prefix
        self.max_events = max_events_per_run
        self.ttl_s = ttl_s

    def _key(self, run_id: str) -> str:
        return f"{self.p}:events:{run_id}"

    async def publish(self, event: RunEvent) -> str:
        key = self._key(event.run_id)
        async with self.r.pipeline(transaction=False) as pipe:
            pipe.xadd(key, {"e": event.model_dump_json()}, maxlen=self.max_events, approximate=True)
            pipe.expire(key, self.ttl_s)
            event_id, _ = await pipe.execute()
        return str(event_id)

    async def history(self, run_id: str, after: str | None = None) -> list[tuple[str, RunEvent]]:
        start = f"({after}" if after else "-"
        entries: Any = await self.r.xrange(self._key(run_id), min=start, max="+")
        return [(str(i), RunEvent.model_validate_json(f["e"])) for i, f in entries]

    async def subscribe(
        self, run_id: str, after: str | None = None, *, idle_timeout_s: float = 15.0
    ) -> AsyncIterator[tuple[str, RunEvent] | None]:
        cursor = after or "0-0"
        key = self._key(run_id)
        while True:
            resp: Any = await self.r.xread(
                {key: cursor}, count=100, block=int(idle_timeout_s * 1000)
            )
            if not resp:
                yield None
                continue
            for _, entries in resp:
                for event_id, fields in entries:
                    cursor = str(event_id)
                    yield cursor, RunEvent.model_validate_json(fields["e"])
