"""Deterministic, scriptable LLM for tests, evals and the offline demo.

A script is a sequence of steps; each step is either a ready `LLMResponse`, an
exception to raise, or a callable that builds a response from the conversation so far.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from harness.llm.base import LLMProvider
from harness.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

ScriptStep = LLMResponse | BaseException | Callable[[Sequence[Message]], LLMResponse]

_ids = itertools.count(1)


def _estimate_tokens(text: str) -> int:
    # ponytail: ~4 chars/token heuristic; good enough for budgets in tests.
    return max(1, len(text) // 4)


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(
        self, script: Sequence[ScriptStep] = (), *, model: str = "fake-1", repeat_last: bool = False
    ) -> None:
        super().__init__(model)
        self._script = list(script)
        self._repeat_last = repeat_last
        self.calls: list[list[Message]] = []

    # ---- builders ---------------------------------------------------------
    @staticmethod
    def text(
        content: str, *, input_tokens: int = 10, output_tokens: int | None = None
    ) -> LLMResponse:
        return LLMResponse(
            message=Message.assistant(content),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
                if output_tokens is not None
                else _estimate_tokens(content),
            ),
            model="fake-1",
            provider="fake",
            finish_reason="end_turn",
        )

    @staticmethod
    def tool_call(
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        thought: str = "",
        call_id: str | None = None,
    ) -> LLMResponse:
        call = ToolCall(id=call_id or f"call_{next(_ids)}", name=name, arguments=arguments or {})
        return LLMResponse(
            message=Message.assistant(thought, [call]),
            usage=Usage(input_tokens=10, output_tokens=10),
            model="fake-1",
            provider="fake",
            finish_reason="tool_use",
        )

    # ---- provider ---------------------------------------------------------
    @property
    def remaining(self) -> int:
        return len(self._script)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._script:
            raise RuntimeError("FakeLLM script exhausted")
        step = (
            self._script[0]
            if (self._repeat_last and len(self._script) == 1)
            else self._script.pop(0)
        )
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, LLMResponse):
            return step
        return step(messages)
