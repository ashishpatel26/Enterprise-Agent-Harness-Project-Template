"""The `LLMProvider` interface every model adapter implements."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence

from harness.llm.types import LLMResponse, Message, StreamEvent, ToolSpec


class LLMError(Exception):
    """Base error for provider failures."""

    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message)
        self.provider = provider


class RetryableLLMError(LLMError):
    """Transient failure (rate limit, 5xx, timeout, connection reset). Safe to retry."""

    def __init__(
        self, message: str, *, provider: str = "unknown", retry_after: float | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after = retry_after


class NonRetryableLLMError(LLMError):
    """Permanent failure (bad request, auth, content policy). Retrying will not help."""


class LLMProvider(abc.ABC):
    """Chat + tool-calling + streaming over a provider-neutral schema."""

    name: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return one complete assistant turn."""

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a turn. Default: one `chat` call replayed as events.

        Adapters with native streaming override this.
        """
        response = await self.chat(messages, tools, max_tokens=max_tokens, temperature=temperature)
        if response.message.content:
            yield StreamEvent(type="text_delta", text=response.message.content)
        for call in response.message.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(type="done", response=response)

    async def aclose(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Release network resources. Override when the adapter holds a client."""
