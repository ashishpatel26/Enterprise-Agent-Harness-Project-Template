"""Fallback across providers: try each in order until one succeeds."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from harness.llm.base import LLMError, LLMProvider
from harness.llm.types import LLMResponse, Message, StreamEvent, ToolSpec

log = logging.getLogger(__name__)


class FallbackProvider(LLMProvider):
    """Primary first, then each fallback. Any `LLMError` (after the wrapped provider's
    own retries) moves on to the next provider; the last error is re-raised."""

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        super().__init__(providers[0].model)
        self.providers = list(providers)
        self.name = providers[0].name

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        last: LLMError | None = None
        for i, provider in enumerate(self.providers):
            try:
                return await provider.chat(
                    messages, tools, max_tokens=max_tokens, temperature=temperature
                )
            except LLMError as exc:
                last = exc
                if i + 1 < len(self.providers):
                    log.warning(
                        "provider failed, falling back",
                        extra={"failed": provider.name, "next": self.providers[i + 1].name},
                    )
        assert last is not None
        raise last

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        last: LLMError | None = None
        for provider in self.providers:
            started = False
            try:
                async for event in provider.stream(
                    messages, tools, max_tokens=max_tokens, temperature=temperature
                ):
                    started = True
                    yield event
                return
            except LLMError as exc:
                # Once output reached the caller, switching providers would splice two
                # different answers together; surface the error instead.
                if started:
                    raise
                last = exc
        assert last is not None
        raise last

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()
