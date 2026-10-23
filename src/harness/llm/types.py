"""Provider-neutral message and response schemas.

Adapters translate between these types and each vendor's wire format, so the rest
of the harness never imports a vendor SDK.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set on role="tool" messages: which call this result answers.
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, call: ToolCall, content: str, *, is_error: bool = False) -> Message:
        return cls(
            role="tool", content=content, tool_call_id=call.id, name=call.name, is_error=is_error
        )


class ToolSpec(BaseModel):
    """What the model sees about a tool: name, description and a JSON schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    model: str
    provider: str
    finish_reason: FinishReason = "end_turn"
    cost_usd: float = 0.0


class StreamEvent(BaseModel):
    """One chunk of a streamed completion. The final event carries the full response."""

    type: Literal["text_delta", "tool_call", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    response: LLMResponse | None = None
