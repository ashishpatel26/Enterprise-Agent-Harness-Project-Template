"""HTTP request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.core.types import RunState
from harness.llm.types import Message
from harness.safety.redaction import Redactor

ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,128}$"


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1, max_length=64)
    input: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = Field(default=None, pattern=ID_PATTERN)
    wait: bool = Field(
        default=False, description="Block until the run finishes or pauses (simple clients)."
    )


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(pattern=ID_PATTERN)
    approved: bool
    comment: str = Field(default="", max_length=2_000)


class PendingApprovalView(BaseModel):
    call_id: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    requested_at: datetime


class UsageView(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int


class RunView(BaseModel):
    run_id: str
    session_id: str
    agent: str
    status: str
    stop_reason: str | None
    answer: str | None
    error: str | None
    step: int
    usage: UsageView
    cost_usd: float
    elapsed_s: float
    pending_approvals: list[PendingApprovalView]
    parent_run_id: str | None
    created_at: datetime
    updated_at: datetime
    links: dict[str, str]
    messages: list[Message] | None = None

    @classmethod
    def from_state(
        cls, s: RunState, redactor: Redactor, *, include_messages: bool = False
    ) -> RunView:
        base = f"/runs/{s.run_id}"
        return cls(
            run_id=s.run_id,
            session_id=s.session_id,
            agent=s.agent,
            status=s.status.value,
            stop_reason=s.stop_reason.value if s.stop_reason else None,
            answer=s.answer,
            error=redactor.redact(s.error) if s.error else None,
            step=s.step,
            usage=UsageView(
                input_tokens=s.usage.input_tokens,
                output_tokens=s.usage.output_tokens,
                cache_read_tokens=s.usage.cache_read_tokens,
                cache_write_tokens=s.usage.cache_write_tokens,
                total_tokens=s.usage.total_tokens,
            ),
            cost_usd=round(s.cost_usd, 6),
            elapsed_s=round(s.elapsed_s, 3),
            pending_approvals=[
                PendingApprovalView(
                    call_id=p.call.id,
                    tool=p.call.name,
                    arguments=redactor.redact_obj(p.call.arguments),
                    reason=p.reason,
                    requested_at=p.requested_at,
                )
                for p in s.pending_approvals
            ],
            parent_run_id=s.parent_run_id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            links={
                "self": base,
                "stream": f"{base}/stream",
                "approve": f"{base}/approve",
                "cancel": f"{base}/cancel",
                "trace": f"{base}/trace",
            },
            messages=[
                m.without_raw().model_copy(update={"content": redactor.redact(m.content)})
                for m in s.messages
                if m.role != "system"
            ]
            if include_messages
            else None,
        )


class AgentView(BaseModel):
    name: str
    description: str
    version: str
    model: str
    tools: list[str]
    sub_agents: list[str]


class ErrorBody(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
