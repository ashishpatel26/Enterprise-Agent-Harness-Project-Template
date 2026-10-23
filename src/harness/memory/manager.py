"""`MemoryManager`: one facade over short-term, long-term and episodic memory.

Implements the agent's `MemoryHook`. Every backend call has a timeout. Recall degrades to
"no memory" on failure (an agent can still work without it); recording failures are
logged and surfaced to the caller, which treats them as non-fatal.
"""

from __future__ import annotations

import asyncio
import logging

from harness.core.types import RunState, RunStatus
from harness.llm.types import Message
from harness.memory.episodic import Episode, EpisodicMemory
from harness.memory.long_term import MemoryItem, VectorMemory
from harness.memory.short_term import SessionMemory

log = logging.getLogger(__name__)

_MAX_MEMORY_CHARS = 1_000


class MemoryManager:
    def __init__(
        self,
        short_term: SessionMemory,
        long_term: VectorMemory,
        episodic: EpisodicMemory,
        *,
        timeout_s: float = 5.0,
        min_score: float = 0.2,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.episodic = episodic
        self.timeout_s = timeout_s
        self.min_score = min_score

    async def recall(self, *, query: str, user_id: str, k: int) -> list[str]:
        try:
            async with asyncio.timeout(self.timeout_s):
                items = await self.long_term.search(user_id, query, k, self.min_score)
        except Exception:
            log.warning("memory recall failed; continuing without memory", exc_info=True)
            return []
        return [i.content for i in items]

    async def session_history(self, session_id: str) -> list[Message]:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self.short_term.get(session_id)
        except Exception:
            log.warning("session history unavailable", exc_info=True)
            return []

    async def remember(self, user_id: str, content: str, **metadata: str) -> None:
        async with asyncio.timeout(self.timeout_s):
            await self.long_term.add(
                MemoryItem(user_id=user_id, content=content[:_MAX_MEMORY_CHARS], metadata=metadata)
            )

    async def record_run(self, state: RunState) -> None:
        async with asyncio.timeout(self.timeout_s):
            await self.episodic.record(Episode.from_state(state))
            if state.status != RunStatus.COMPLETED or not state.answer:
                return
            await self.short_term.append(
                state.session_id, [Message.user(state.input), Message.assistant(state.answer)]
            )
            await self.long_term.add(
                MemoryItem(
                    id=state.run_id,
                    user_id=state.user_id,
                    content=f"Task: {state.input}\nOutcome: {state.answer}"[:_MAX_MEMORY_CHARS],
                    metadata={"run_id": state.run_id, "agent": state.agent},
                )
            )
