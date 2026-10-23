"""Long-term memory: semantic search over remembered facts, scoped per user."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from harness.memory.embeddings import Embedder, cosine


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    score: float | None = None


class VectorMemory(Protocol):
    async def add(self, item: MemoryItem) -> None: ...

    async def search(
        self, user_id: str, query: str, k: int, min_score: float = 0.0
    ) -> list[MemoryItem]: ...

    async def delete_user(self, user_id: str) -> int: ...


class InMemoryVectorMemory:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._items: list[tuple[MemoryItem, list[float]]] = []

    async def add(self, item: MemoryItem) -> None:
        [vec] = await self.embedder.embed([item.content])
        self._items.append((item, vec))

    async def search(
        self, user_id: str, query: str, k: int, min_score: float = 0.0
    ) -> list[MemoryItem]:
        [q] = await self.embedder.embed([query])
        scored = [
            item.model_copy(update={"score": cosine(q, vec)})
            for item, vec in self._items
            if item.user_id == user_id
        ]
        scored = [s for s in scored if (s.score or 0.0) >= min_score]
        return sorted(scored, key=lambda i: i.score or 0.0, reverse=True)[:k]

    async def delete_user(self, user_id: str) -> int:
        before = len(self._items)
        self._items = [(i, v) for i, v in self._items if i.user_id != user_id]
        return before - len(self._items)


class PgVectorMemory:
    """Postgres + pgvector. Cosine distance with an HNSW index; rows scoped by user_id.

    SQL is built with the table name only after `__init__` validates it as a plain
    identifier; every value goes through query parameters.
    """

    def __init__(self, dsn: str, embedder: Embedder, *, table: str = "harness_memories") -> None:
        if not table.isidentifier():
            raise ValueError("table must be a plain identifier")
        self.dsn = dsn
        self.embedder = embedder
        self.table = table
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool

            async def configure(conn: Any) -> None:
                await register_vector_async(conn)

            pool = AsyncConnectionPool(
                self.dsn,
                min_size=1,
                max_size=10,
                timeout=10,
                open=False,
                configure=configure,
                kwargs={"options": "-c statement_timeout=10000"},
            )
            await pool.open(wait=True, timeout=10)
            self._pool = pool
        return self._pool

    async def ensure_schema(self) -> None:
        """Idempotent DDL; safe to call on every startup."""
        from psycopg_pool import AsyncConnectionPool

        # The vector type must exist before connections register it, so bootstrap the
        # extension on a plain connection first.
        async with (
            AsyncConnectionPool(self.dsn, min_size=1, max_size=1, open=False) as boot,
            boot.connection() as conn,
        ):
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pool = await self._get_pool()
        t, dim = self.table, self.embedder.dim
        async with pool.connection() as conn:
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
            await conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_user_idx ON {t} (user_id)")
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_embedding_idx ON {t} "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    async def add(self, item: MemoryItem) -> None:
        import numpy as np

        [vec] = await self.embedder.embed([item.content])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {self.table} (id, user_id, content, metadata, embedding, created_at)"  # noqa: S608
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (
                    item.id,
                    item.user_id,
                    item.content,
                    json.dumps(item.metadata),
                    np.array(vec, dtype=np.float32),
                    item.created_at,
                ),
            )

    async def search(
        self, user_id: str, query: str, k: int, min_score: float = 0.0
    ) -> list[MemoryItem]:
        import numpy as np

        [q] = await self.embedder.embed([query])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT id, user_id, content, metadata, created_at, "  # noqa: S608
                f"1 - (embedding <=> %s) AS score FROM {self.table} "
                "WHERE user_id = %s ORDER BY embedding <=> %s LIMIT %s",
                (np.array(q, dtype=np.float32), user_id, np.array(q, dtype=np.float32), k),
            )
            rows = await cur.fetchall()
        items = [
            MemoryItem(
                id=r[0],
                user_id=r[1],
                content=r[2],
                metadata=r[3] or {},
                created_at=r[4],
                score=float(r[5]),
            )
            for r in rows
        ]
        return [i for i in items if (i.score or 0.0) >= min_score]

    async def delete_user(self, user_id: str) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"DELETE FROM {self.table} WHERE user_id = %s",  # noqa: S608
                (user_id,),
            )
            return int(cur.rowcount)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
