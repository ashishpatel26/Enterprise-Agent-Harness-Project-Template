"""Run/session state persistence, checkpointing, leases and idempotency."""

from __future__ import annotations

from dataclasses import dataclass

from harness.config import Settings
from harness.state.idempotency import (
    IdempotencyStore,
    IdempotentExecutor,
    InMemoryIdempotencyStore,
    RedisIdempotencyStore,
)
from harness.state.store import (
    InMemoryLeases,
    InMemoryStateStore,
    LeaseManager,
    RedisLeases,
    RedisStateStore,
    StateStore,
    make_redis,
)


@dataclass
class StateBackends:
    runs: StateStore
    leases: LeaseManager
    idempotency: IdempotencyStore


def build_state_backends(settings: Settings) -> StateBackends:
    if settings.storage_backend == "redis":
        r = make_redis(settings.redis_url.get_secret_value())
        return StateBackends(RedisStateStore(r), RedisLeases(r), RedisIdempotencyStore(r))
    return StateBackends(InMemoryStateStore(), InMemoryLeases(), InMemoryIdempotencyStore())


__all__ = [
    "IdempotencyStore",
    "IdempotentExecutor",
    "InMemoryIdempotencyStore",
    "InMemoryLeases",
    "InMemoryStateStore",
    "LeaseManager",
    "RedisIdempotencyStore",
    "RedisLeases",
    "RedisStateStore",
    "StateBackends",
    "StateStore",
    "build_state_backends",
    "make_redis",
]
