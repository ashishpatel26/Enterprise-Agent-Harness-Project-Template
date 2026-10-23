"""Episodic memory: compact records of finished runs (what was asked, what happened)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import redis.asyncio as aioredis
from pydantic import BaseModel

from harness.core.types import RunState


class Episode(BaseModel):
    run_id: str
    session_id: str
    user_id: str
    agent: str
    input: str
    answer: str | None
    stop_reason: str | None
    steps: int
    tools_used: list[str]
    tool_errors: int
    total_tokens: int
    cost_usd: float
    elapsed_s: float
    created_at: datetime

    @classmethod
    def from_state(cls, s: RunState) -> Episode:
        tools = [c.name for m in s.messages for c in m.tool_calls]
        return cls(
            run_id=s.run_id,
            session_id=s.session_id,
            user_id=s.user_id,
            agent=s.agent,
            input=s.input,
            answer=s.answer,
            stop_reason=s.stop_reason.value if s.stop_reason else None,
            steps=s.step,
            tools_used=tools,
            tool_errors=sum(1 for m in s.messages if m.role == "tool" and m.is_error),
            total_tokens=s.usage.total_tokens,
            cost_usd=s.cost_usd,
            elapsed_s=s.elapsed_s,
            created_at=s.created_at,
        )


class EpisodicMemory(Protocol):
    async def record(self, episode: Episode) -> None: ...

    async def recent(self, user_id: str, limit: int = 20) -> list[Episode]: ...


class InMemoryEpisodicMemory:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    async def record(self, episode: Episode) -> None:
        self._episodes.append(episode)

    async def recent(self, user_id: str, limit: int = 20) -> list[Episode]:
        mine = [e for e in self._episodes if e.user_id == user_id]
        return sorted(mine, key=lambda e: e.created_at, reverse=True)[:limit]


class RedisEpisodicMemory:
    def __init__(
        self,
        redis: aioredis.Redis,
        *,
        prefix: str = "harness",
        max_per_user: int = 500,
        ttl_s: int = 90 * 86400,
    ) -> None:
        self.r = redis
        self.p = prefix
        self.max_per_user = max_per_user
        self.ttl_s = ttl_s

    async def record(self, episode: Episode) -> None:
        key = f"{self.p}:episodes:{episode.user_id}"
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.zadd(key, {episode.model_dump_json(): episode.created_at.timestamp()})
            pipe.zremrangebyrank(key, 0, -self.max_per_user - 1)
            pipe.expire(key, self.ttl_s)
            await pipe.execute()

    async def recent(self, user_id: str, limit: int = 20) -> list[Episode]:
        raw = await self.r.zrevrange(f"{self.p}:episodes:{user_id}", 0, limit - 1)
        return [Episode.model_validate_json(str(r)) for r in raw]
