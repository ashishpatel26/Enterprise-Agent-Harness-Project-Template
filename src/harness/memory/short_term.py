"""Short-term memory: per-session conversation history (user turns + final answers)."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

import redis.asyncio as aioredis

from harness.llm.types import Message


class SessionMemory(Protocol):
    async def append(self, session_id: str, messages: list[Message]) -> None: ...

    async def get(self, session_id: str) -> list[Message]: ...

    async def clear(self, session_id: str) -> None: ...


class InMemorySessionMemory:
    def __init__(self, max_messages: int = 40) -> None:
        self._data: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=max_messages))

    async def append(self, session_id: str, messages: list[Message]) -> None:
        self._data[session_id].extend(m.model_dump_json() for m in messages)

    async def get(self, session_id: str) -> list[Message]:
        return [Message.model_validate_json(m) for m in self._data.get(session_id, ())]

    async def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionMemory:
    """Capped Redis list per session with a sliding TTL."""

    def __init__(
        self,
        redis: aioredis.Redis,
        *,
        prefix: str = "harness",
        max_messages: int = 40,
        ttl_s: int = 7 * 86400,
    ) -> None:
        self.r = redis
        self.p = prefix
        self.max_messages = max_messages
        self.ttl_s = ttl_s

    def _key(self, session_id: str) -> str:
        return f"{self.p}:session:{session_id}:messages"

    async def append(self, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        key = self._key(session_id)
        async with self.r.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(m.model_dump_json() for m in messages))
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl_s)
            await pipe.execute()

    async def get(self, session_id: str) -> list[Message]:
        raw = await self.r.lrange(self._key(session_id), 0, -1)
        return [Message.model_validate_json(m) for m in raw]

    async def clear(self, session_id: str) -> None:
        await self.r.delete(self._key(session_id))
