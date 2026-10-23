"""OpenAI adapter (Chat Completions with function tools, via the official `openai` SDK)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

import openai

from harness.llm.base import LLMError, LLMProvider, NonRetryableLLMError, RetryableLLMError
from harness.llm.pricing import cost_usd
from harness.llm.types import FinishReason, LLMResponse, Message, ToolCall, ToolSpec, Usage

_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Malformed JSON is surfaced to the tool layer, whose schema validation reports it
    back to the model as a correctable error rather than crashing the run."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__unparseable_arguments__": raw}
    return parsed if isinstance(parsed, dict) else {"__unparseable_arguments__": raw}


def _map_error(exc: Exception) -> LLMError:
    if isinstance(exc, openai.RateLimitError):
        raw = exc.response.headers.get("retry-after")
        try:
            retry_after = float(raw) if raw is not None else None
        except ValueError:
            retry_after = None
        return RetryableLLMError(str(exc), provider="openai", retry_after=retry_after)
    if isinstance(exc, openai.APIConnectionError):
        return RetryableLLMError(str(exc), provider="openai")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in (408, 409, 429) or exc.status_code >= 500:
            return RetryableLLMError(str(exc), provider="openai")
        return NonRetryableLLMError(str(exc), provider="openai")
    return NonRetryableLLMError(f"unexpected error: {exc!r}", provider="openai")


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(model)
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        req: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            req["temperature"] = temperature
        if tools:
            req["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        try:
            resp = await self._client.chat.completions.create(**cast(Any, req))
        except openai.OpenAIError as exc:
            raise _map_error(exc) from exc

        choice = resp.choices[0]
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_arguments(tc.function.arguments),
            )
            for tc in (choice.message.tool_calls or [])
            if tc.type == "function"
        ]
        cached = 0
        if resp.usage and resp.usage.prompt_tokens_details:
            cached = resp.usage.prompt_tokens_details.cached_tokens or 0
        prompt = resp.usage.prompt_tokens if resp.usage else 0
        usage = Usage(
            input_tokens=prompt - cached,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            cache_read_tokens=cached,
        )
        return LLMResponse(
            message=Message.assistant(choice.message.content or "", calls),
            usage=usage,
            model=resp.model,
            provider=self.name,
            finish_reason=_FINISH_MAP.get(str(choice.finish_reason), "end_turn"),
            cost_usd=cost_usd(resp.model, usage),
        )

    async def aclose(self) -> None:
        await self._client.close()
