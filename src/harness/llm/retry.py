"""Retries with exponential backoff + full jitter, and a hard per-call timeout."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

from harness.llm.base import LLMProvider, RetryableLLMError
from harness.llm.types import LLMResponse, Message, StreamEvent, ToolSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    timeout_s: float = 60.0

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Full jitter (AWS architecture blog): uniform(0, min(cap, base * 2^attempt)).

        A server-provided Retry-After wins when present, capped at max_delay_s.
        """
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_delay_s)
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return random.uniform(0, ceiling)  # noqa: S311 - jitter, not crypto


async def call_with_retries[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    provider: str = "unknown",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `fn` with a timeout per attempt; retry only `RetryableLLMError` and timeouts."""
    attempt = 0
    while True:
        try:
            async with asyncio.timeout(policy.timeout_s):
                return await fn()
        except TimeoutError as exc:
            err: RetryableLLMError = RetryableLLMError(
                f"call exceeded {policy.timeout_s}s", provider=provider
            )
            err.__cause__ = exc
        except RetryableLLMError as exc:
            err = exc
        if attempt >= policy.max_retries:
            raise err
        delay = policy.backoff(attempt, err.retry_after)
        log.warning(
            "llm call failed, retrying",
            extra={"provider": provider, "attempt": attempt + 1, "delay_s": round(delay, 3)},
        )
        attempt += 1
        await sleep(delay)


class RetryingProvider(LLMProvider):
    """Wraps a provider with timeouts and retries. Streams are retried only before
    the first event is emitted, so callers never see duplicated output."""

    def __init__(self, inner: LLMProvider, policy: RetryPolicy | None = None) -> None:
        super().__init__(inner.model)
        self.inner = inner
        self.policy = policy or RetryPolicy()
        self.name = inner.name

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        return await call_with_retries(
            lambda: self.inner.chat(
                messages, tools, max_tokens=max_tokens, temperature=temperature
            ),
            self.policy,
            provider=self.inner.name,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async def first() -> tuple[StreamEvent, AsyncIterator[StreamEvent]]:
            it = aiter(
                self.inner.stream(messages, tools, max_tokens=max_tokens, temperature=temperature)
            )
            return await anext(it), it

        event, it = await call_with_retries(first, self.policy, provider=self.inner.name)
        yield event
        async for event in it:
            yield event

    async def aclose(self) -> None:
        await self.inner.aclose()
