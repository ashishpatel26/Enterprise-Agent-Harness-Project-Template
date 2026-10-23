"""Anthropic Claude adapter (Messages API via the official `anthropic` SDK).

Production defaults:
- Streams every request and assembles it with `get_final_message()` so large
  `max_tokens` never hits HTTP timeouts.
- Adaptive thinking with an explicit `prefix_mismatch_behavior: "drop_block"`: a
  history rebase (compaction) degrades gracefully instead of 400-ing.
- Assistant turns replay their raw content blocks (thinking + signatures) verbatim.
- Server-side refusal fallbacks (`fallbacks="default"`) on models that support it.
- Automatic prompt caching (top-level `cache_control`).
- SDK retries disabled: `RetryingProvider` owns retries so attempts are not multiplied.
- `temperature` is never sent (rejected by the Messages API).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, cast

import anthropic

from harness.llm.base import LLMError, LLMProvider, NonRetryableLLMError, RetryableLLMError
from harness.llm.pricing import cost_usd
from harness.llm.types import (
    FinishReason,
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"
THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
# Models that accept `fallbacks="default"` on the Claude API.
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)

_STOP_MAP: dict[str, FinishReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def to_anthropic_messages(
    messages: Sequence[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split leading system messages into the top-level system prompt and convert the rest.

    Later system messages stay in `messages` as mid-conversation system turns, which keeps
    the cached/thinking-bound prefix append-only. Consecutive tool results are merged into
    one user turn, as the API expects for parallel tool use.
    """
    system_parts: list[str] = []
    i = 0
    while i < len(messages) and messages[i].role == "system":
        system_parts.append(messages[i].content)
        i += 1

    out: list[dict[str, Any]] = []
    for m in messages[i:]:
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
                "is_error": m.is_error,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                prev = out[-1]["content"]
                if prev and all(b.get("type") == "tool_result" for b in prev):
                    prev.append(block)
                    continue
            out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            if m.raw_provider == "anthropic" and m.raw_content is not None:
                content: list[dict[str, Any]] = m.raw_content
            else:
                content = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content.extend(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                )
            out.append({"role": "assistant", "content": content})
        else:  # user, or a mid-conversation system message
            out.append({"role": m.role, "content": m.content})
    return ("\n\n".join(system_parts) or None), out


def _map_error(exc: Exception) -> LLMError:
    if isinstance(exc, anthropic.RateLimitError):
        raw = exc.response.headers.get("retry-after")
        try:
            retry_after = float(raw) if raw is not None else None
        except ValueError:
            retry_after = None
        return RetryableLLMError(str(exc), provider="anthropic", retry_after=retry_after)
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return RetryableLLMError(str(exc), provider="anthropic")
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code in (408, 409, 429) or exc.status_code >= 500:
            return RetryableLLMError(str(exc), provider="anthropic")
        return NonRetryableLLMError(str(exc), provider="anthropic")
    return NonRetryableLLMError(f"unexpected error: {exc!r}", provider="anthropic")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        effort: Effort = "high",
        refusal_fallbacks: bool = True,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(model)
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks and model in FALLBACK_MODELS
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    def _request(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], max_tokens: int
    ) -> dict[str, Any]:
        system, wire = to_anthropic_messages(messages)
        betas = [THINKING_BINDING_BETA]
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": wire,
            "thinking": {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            },
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            req["system"] = system
        if tools:
            req["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self.refusal_fallbacks:
            req["fallbacks"] = "default"
            betas.append(FALLBACK_BETA)
        req["betas"] = betas
        return req

    def _to_response(self, msg: Any) -> LLMResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []
        for block in msg.content:
            raw.append(block.model_dump(mode="json", exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
        )
        # A server-side fallback may have served the turn; bill the model that ran.
        served_by = str(msg.model or self.model)
        return LLMResponse(
            message=Message(
                role="assistant",
                content="".join(text_parts),
                tool_calls=calls,
                raw_content=raw,
                raw_provider="anthropic",
            ),
            usage=usage,
            model=served_by,
            provider=self.name,
            finish_reason=_STOP_MAP.get(str(msg.stop_reason), "end_turn"),
            cost_usd=cost_usd(served_by, usage),
        )

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        final: LLMResponse | None = None
        async for event in self.stream(messages, tools, max_tokens=max_tokens):
            if event.type == "done":
                final = event.response
        assert final is not None
        return final

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req = self._request(messages, tools, max_tokens)
        try:
            async with self._client.beta.messages.stream(**cast(Any, req)) as stream:
                async for text in stream.text_stream:
                    yield StreamEvent(type="text_delta", text=text)
                final = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise _map_error(exc) from exc
        response = self._to_response(final)
        for call in response.message.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(type="done", response=response)

    async def aclose(self) -> None:
        await self._client.close()
