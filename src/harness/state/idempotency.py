"""At-most-once execution for tools with side effects.

The loop's per-step checkpoint already skips tool calls that finished before a crash.
The remaining gap is a crash *during* a call: the side effect may or may not have
happened, and no result was saved. For idempotent tools re-running is harmless. For
non-idempotent tools this executor records an "in progress" marker before running; if
it finds a marker with no result, it refuses to re-run and tells the model the outcome
is unknown, instead of risking a duplicate payment, email or write.
"""

from __future__ import annotations

from typing import Protocol

import redis.asyncio as aioredis

from harness.core.types import ToolContext, ToolOutcome
from harness.llm.types import ToolCall, ToolSpec
from harness.tools.registry import ToolRegistry

_IN_PROGRESS = "__in_progress__"


class IdempotencyStore(Protocol):
    async def begin(self, key: str) -> str | None:
        """Mark `key` in progress. Returns the previous value if one existed."""
        ...

    async def complete(self, key: str, outcome_json: str) -> None: ...


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def begin(self, key: str) -> str | None:
        previous = self.data.get(key)
        if previous is None:
            self.data[key] = _IN_PROGRESS
        return previous

    async def complete(self, key: str, outcome_json: str) -> None:
        self.data[key] = outcome_json


class RedisIdempotencyStore:
    def __init__(
        self, redis: aioredis.Redis, *, prefix: str = "harness", ttl_s: int = 7 * 86400
    ) -> None:
        self.r = redis
        self.p = prefix
        self.ttl_s = ttl_s

    async def begin(self, key: str) -> str | None:
        k = f"{self.p}:idem:{key}"
        if await self.r.set(k, _IN_PROGRESS, nx=True, ex=self.ttl_s):
            return None
        value = await self.r.get(k)
        return None if value is None else str(value)

    async def complete(self, key: str, outcome_json: str) -> None:
        await self.r.set(f"{self.p}:idem:{key}", outcome_json, ex=self.ttl_s)


class IdempotentExecutor:
    """Wraps a `ToolRegistry`; implements the loop's `ToolExecutor` contract."""

    def __init__(self, registry: ToolRegistry, store: IdempotencyStore) -> None:
        self.registry = registry
        self.store = store

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        return self.registry.specs(names)

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        tool = self.registry.get(call.name)
        if tool is None or tool.idempotent:
            return await self.registry.execute(call, ctx)

        key = ctx.idempotency_key
        previous = await self.store.begin(key)
        if previous == _IN_PROGRESS:
            return ToolOutcome(
                content=(
                    f"A previous attempt of {call.name} was interrupted and its outcome is "
                    "unknown. It was not re-run to avoid duplicating side effects. Check the "
                    "result another way before trying again."
                ),
                is_error=True,
                metadata={"idempotency": "unknown_outcome"},
            )
        if previous is not None:
            outcome = ToolOutcome.model_validate_json(previous)
            outcome.metadata["idempotency"] = "replayed"
            return outcome
        outcome = await self.registry.execute(call, ctx)
        await self.store.complete(key, outcome.model_dump_json())
        return outcome
