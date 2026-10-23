from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from harness.llm import FakeLLM, Message, NonRetryableLLMError, RetryableLLMError, StreamEvent
from harness.llm.fallback import FallbackProvider
from harness.llm.retry import RetryingProvider, RetryPolicy, call_with_retries

FAST = RetryPolicy(max_retries=3, base_delay_s=0.0, max_delay_s=0.0, timeout_s=1.0)


async def test_retryable_errors_are_retried_then_succeed() -> None:
    llm = RetryingProvider(
        FakeLLM([RetryableLLMError("429"), RetryableLLMError("503"), FakeLLM.text("ok")]), FAST
    )
    resp = await llm.chat([Message.user("x")])
    assert resp.message.content == "ok"


async def test_non_retryable_error_is_not_retried() -> None:
    fake = FakeLLM([NonRetryableLLMError("400"), FakeLLM.text("never")])
    with pytest.raises(NonRetryableLLMError):
        await RetryingProvider(fake, FAST).chat([])
    assert fake.remaining == 1


async def test_retries_exhausted_raises_last_error() -> None:
    fake = FakeLLM([RetryableLLMError(str(i)) for i in range(4)])
    with pytest.raises(RetryableLLMError, match="3"):
        await RetryingProvider(fake, FAST).chat([])


async def test_timeout_is_converted_to_retryable_and_retried() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(5)
        return "ok"

    policy = RetryPolicy(max_retries=1, base_delay_s=0, max_delay_s=0, timeout_s=0.05)
    assert await call_with_retries(flaky, policy) == "ok"
    assert attempts == 2


async def test_timeout_exhausted_raises_retryable() -> None:
    async def hang() -> str:
        await asyncio.sleep(5)
        return "never"

    with pytest.raises(RetryableLLMError, match="exceeded"):
        await call_with_retries(hang, RetryPolicy(max_retries=0, timeout_s=0.01))


async def test_retry_after_is_honoured() -> None:
    delays: list[float] = []

    async def record(d: float) -> None:
        delays.append(d)

    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableLLMError("429", retry_after=2.5)
        return "ok"

    await call_with_retries(fn, RetryPolicy(max_retries=2, max_delay_s=10), sleep=record)
    assert delays == [2.5]


def test_backoff_is_bounded_full_jitter() -> None:
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0)
    for attempt in range(10):
        assert 0 <= p.backoff(attempt) <= min(8.0, 2**attempt)
    assert p.backoff(0, retry_after=100) == 8.0


async def test_stream_retries_before_first_event() -> None:
    llm = RetryingProvider(FakeLLM([RetryableLLMError("503"), FakeLLM.text("hi")]), FAST)
    events = [e async for e in llm.stream([])]
    assert events[-1].type == "done"


async def test_fallback_uses_next_provider_on_failure() -> None:
    primary = FakeLLM([NonRetryableLLMError("down")])
    secondary = FakeLLM([FakeLLM.text("from backup")])
    resp = await FallbackProvider([primary, secondary]).chat([])
    assert resp.message.content == "from backup"


async def test_fallback_raises_last_error_when_all_fail() -> None:
    p = FallbackProvider(
        [FakeLLM([NonRetryableLLMError("a")]), FakeLLM([NonRetryableLLMError("b")])]
    )
    with pytest.raises(NonRetryableLLMError, match="b"):
        await p.chat([])


async def test_fallback_stream_switches_before_output() -> None:
    p = FallbackProvider([FakeLLM([NonRetryableLLMError("a")]), FakeLLM([FakeLLM.text("ok")])])
    events = [e async for e in p.stream([])]
    assert events[0].text == "ok"


async def test_fallback_stream_does_not_splice_after_output() -> None:
    class Breaks(FakeLLM):
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type="text_delta", text="partial")
            raise NonRetryableLLMError("mid-stream")

    p = FallbackProvider([Breaks(), FakeLLM([FakeLLM.text("other")])])
    with pytest.raises(NonRetryableLLMError, match="mid-stream"):
        _ = [e async for e in p.stream([])]


def test_fallback_requires_providers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackProvider([])
