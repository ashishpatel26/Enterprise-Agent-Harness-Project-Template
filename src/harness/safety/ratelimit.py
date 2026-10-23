"""Sliding-window rate limits (per API key, per user, per tool)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

import redis.asyncio as aioredis


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_s: float


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_s: float) -> RateLimitResult: ...


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def hit(self, key: str, limit: int, window_s: float) -> RateLimitResult:
        now = time.monotonic()
        async with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return RateLimitResult(False, 0, max(0.0, q[0] + window_s - now))
            q.append(now)
            return RateLimitResult(True, limit - len(q), 0.0)


class RedisRateLimiter:
    """Sliding log in a sorted set; shared across all API replicas."""

    def __init__(self, redis: aioredis.Redis, *, prefix: str = "harness") -> None:
        self.r = redis
        self.p = prefix

    async def hit(self, key: str, limit: int, window_s: float) -> RateLimitResult:
        k = f"{self.p}:rl:{key}"
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex}"
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(k, 0, now - window_s)
            pipe.zadd(k, {member: now})
            pipe.zcard(k)
            pipe.zrange(k, 0, 0, withscores=True)
            pipe.expire(k, int(window_s) + 1)
            _, _, count, oldest, _ = await pipe.execute()
        if int(count) > limit:
            await self.r.zrem(k, member)
            oldest_ts = float(oldest[0][1]) if oldest else now
            return RateLimitResult(False, 0, max(0.0, oldest_ts + window_s - now))
        return RateLimitResult(True, limit - int(count), 0.0)
