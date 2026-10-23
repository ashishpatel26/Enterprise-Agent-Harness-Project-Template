"""Adapter wire-format conversion and error mapping (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from harness.llm import Message, NonRetryableLLMError, RetryableLLMError, ToolCall, ToolSpec
from harness.llm.anthropic_provider import (
    FALLBACK_BETA,
    AnthropicProvider,
    to_anthropic_messages,
)
from harness.llm.anthropic_provider import _map_error as map_anthropic
from harness.llm.openai_provider import _parse_arguments, to_openai_messages

CALL = ToolCall(id="t1", name="file_read", arguments={"path": "a.txt"})
CALL2 = ToolCall(id="t2", name="file_read", arguments={"path": "b.txt"})


def _convo() -> list[Message]:
    return [
        Message.system("sys"),
        Message.user("hi"),
        Message.assistant("thinking aloud", [CALL, CALL2]),
        Message.tool(CALL, "A"),
        Message.tool(CALL2, "B", is_error=True),
        Message.system("mid-run operator note"),
    ]


def test_anthropic_conversion_splits_system_and_merges_tool_results() -> None:
    system, wire = to_anthropic_messages(_convo())
    assert system == "sys"
    assert wire[0] == {"role": "user", "content": "hi"}
    assert wire[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "file_read",
        "input": {"path": "a.txt"},
    }
    results = wire[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    assert wire[3] == {"role": "system", "content": "mid-run operator note"}


def test_anthropic_replays_raw_blocks_verbatim() -> None:
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "x"}]
    msg = Message(role="assistant", content="x", raw_content=raw, raw_provider="anthropic")
    _, wire = to_anthropic_messages([Message.user("q"), msg])
    assert wire[1]["content"] == raw


def test_raw_blocks_from_other_provider_are_ignored() -> None:
    msg = Message(role="assistant", content="x", raw_content=[{"a": 1}], raw_provider="other")
    _, wire = to_anthropic_messages([Message.user("q"), msg])
    assert wire[1]["content"] == [{"type": "text", "text": "x"}]


def test_anthropic_request_shape() -> None:
    p = AnthropicProvider("claude-opus-5-5", api_key="k", effort="medium")
    tool = ToolSpec(name="t", description="d", parameters={"type": "object"})
    req = p._request([Message.system("s"), Message.user("u")], [tool], 1000)
    assert req["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default"
    assert FALLBACK_BETA in req["betas"]
    assert req["tools"][0]["input_schema"] == {"type": "object"}
    assert "temperature" not in req


def test_no_refusal_fallbacks_for_unsupported_model() -> None:
    p = AnthropicProvider("claude-haiku-5-5", api_key="k")
    req = p._request([Message.user("u")], [], 10)
    assert "fallbacks" not in req
    assert FALLBACK_BETA not in req["betas"]


class _Block(SimpleNamespace):
    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self.__dict__)


def test_anthropic_response_parsing() -> None:
    p = AnthropicProvider("claude-opus-5-5", api_key="k")
    msg = SimpleNamespace(
        content=[
            _Block(type="thinking", thinking="", signature="s"),
            _Block(type="text", text="Let me check."),
            _Block(type="tool_use", id="tu1", name="file_read", input={"path": "x"}),
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=None,
        ),
        model="claude-opus-5-5",
        stop_reason="tool_use",
    )
    resp = p._to_response(msg)
    assert resp.finish_reason == "tool_use"
    assert resp.message.tool_calls[0].arguments == {"path": "x"}
    assert resp.message.raw_provider == "anthropic"
    assert resp.message.raw_content is not None and len(resp.message.raw_content) == 3
    assert resp.usage.cache_read_tokens == 50
    assert resp.cost_usd > 0


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status, headers=headers, request=httpx2.Request("POST", "https://api.example")
    )


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (
            anthropic.RateLimitError("rl", response=_resp(429, {"retry-after": "3"}), body=None),
            RetryableLLMError,
        ),
        (anthropic.InternalServerError("boom", response=_resp(500), body=None), RetryableLLMError),
        (anthropic.BadRequestError("bad", response=_resp(400), body=None), NonRetryableLLMError),
        (
            anthropic.AuthenticationError("auth", response=_resp(401), body=None),
            NonRetryableLLMError,
        ),
        (ValueError("weird"), NonRetryableLLMError),
    ],
)
def test_anthropic_error_mapping(exc: Exception, kind: type[Exception]) -> None:
    mapped = map_anthropic(exc)
    assert isinstance(mapped, kind)
    if isinstance(mapped, RetryableLLMError) and isinstance(exc, anthropic.RateLimitError):
        assert mapped.retry_after == 3.0


def test_openai_conversion() -> None:
    wire = to_openai_messages(_convo())
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'
    assert wire[3] == {"role": "tool", "tool_call_id": "t1", "content": "A"}


def test_openai_bad_arguments_surface_to_validation() -> None:
    assert _parse_arguments('{"a": 1}') == {"a": 1}
    assert _parse_arguments("{not json") == {"__unparseable_arguments__": "{not json"}
    assert _parse_arguments("[1]") == {"__unparseable_arguments__": "[1]"}


def test_openai_sdk_is_importable() -> None:
    assert hasattr(openai, "AsyncOpenAI")
