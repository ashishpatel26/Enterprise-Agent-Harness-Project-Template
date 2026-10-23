from __future__ import annotations

from collections.abc import Sequence

import pytest

from harness.llm import FakeLLM, LLMResponse, Message, RetryableLLMError


async def test_scripted_responses_are_returned_in_order() -> None:
    llm = FakeLLM([FakeLLM.tool_call("file_read", {"path": "a"}), FakeLLM.text("done")])
    first = await llm.chat([Message.user("hi")])
    assert first.finish_reason == "tool_use"
    assert first.message.tool_calls[0].name == "file_read"
    second = await llm.chat([Message.user("hi")])
    assert second.message.content == "done"
    assert llm.remaining == 0
    assert len(llm.calls) == 2


async def test_callable_step_sees_conversation() -> None:
    def echo(messages: Sequence[Message]) -> LLMResponse:
        return FakeLLM.text(f"echo:{messages[-1].content}")

    llm = FakeLLM([echo])
    resp = await llm.chat([Message.user("ping")])
    assert resp.message.content == "echo:ping"


async def test_exception_step_is_raised() -> None:
    llm = FakeLLM([RetryableLLMError("429", provider="fake")])
    with pytest.raises(RetryableLLMError):
        await llm.chat([Message.user("x")])


async def test_exhausted_script_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="exhausted"):
        await FakeLLM().chat([Message.user("x")])


async def test_repeat_last_keeps_returning_final_step() -> None:
    llm = FakeLLM([FakeLLM.text("same")], repeat_last=True)
    for _ in range(3):
        assert (await llm.chat([])).message.content == "same"


async def test_default_stream_replays_chat() -> None:
    llm = FakeLLM([FakeLLM.text("hello")])
    events = [e async for e in llm.stream([Message.user("x")])]
    assert [e.type for e in events] == ["text_delta", "done"]
    assert events[-1].response is not None
