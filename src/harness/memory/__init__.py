"""Short-term, long-term and episodic memory behind one facade."""

from __future__ import annotations

from harness.config import Settings
from harness.memory.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from harness.memory.episodic import (
    Episode,
    EpisodicMemory,
    InMemoryEpisodicMemory,
    RedisEpisodicMemory,
)
from harness.memory.long_term import InMemoryVectorMemory, MemoryItem, PgVectorMemory, VectorMemory
from harness.memory.manager import MemoryManager
from harness.memory.short_term import InMemorySessionMemory, RedisSessionMemory, SessionMemory


def build_memory(settings: Settings, embedder: Embedder | None = None) -> MemoryManager:
    """In-memory backends for `storage_backend=memory`; Redis + pgvector for `redis`.

    Call `await manager.long_term.ensure_schema()` once at startup when using pgvector.
    """
    embedder = embedder or HashingEmbedder(settings.embedding_dim)
    if settings.storage_backend == "redis":
        from harness.state.store import make_redis

        r = make_redis(settings.redis_url.get_secret_value())
        long_term: VectorMemory
        if settings.database_url is not None:
            long_term = PgVectorMemory(settings.database_url.get_secret_value(), embedder)
        else:
            long_term = InMemoryVectorMemory(embedder)
        return MemoryManager(RedisSessionMemory(r), long_term, RedisEpisodicMemory(r))
    return MemoryManager(
        InMemorySessionMemory(), InMemoryVectorMemory(embedder), InMemoryEpisodicMemory()
    )


__all__ = [
    "Embedder",
    "Episode",
    "EpisodicMemory",
    "HashingEmbedder",
    "InMemoryEpisodicMemory",
    "InMemorySessionMemory",
    "InMemoryVectorMemory",
    "MemoryItem",
    "MemoryManager",
    "OpenAIEmbedder",
    "PgVectorMemory",
    "RedisEpisodicMemory",
    "RedisSessionMemory",
    "SessionMemory",
    "VectorMemory",
    "build_memory",
]
