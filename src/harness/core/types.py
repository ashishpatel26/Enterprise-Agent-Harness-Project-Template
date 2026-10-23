"""Run state and the contracts the agent loop depends on.

`RunState` is the single source of truth for a run. It is serializable, so the loop
can checkpoint it after every step and resume after a crash or an approval pause.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from harness.llm.types import Message, ToolCall, ToolSpec, Usage


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TOKEN_BUDGET = "token_budget"  # noqa: S105 - not a secret
    COST_BUDGET = "cost_budget"
    TIMEOUT = "timeout"
    OUTPUT_TRUNCATED = "output_truncated"
    REFUSED = "refused"
    AWAITING_APPROVAL = "awaiting_approval"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    LLM_ERROR = "llm_error"
    CANCELLED = "cancelled"
    ERROR = "error"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class RunLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=20, ge=1, le=500)
    max_total_tokens: int = Field(default=500_000, ge=1)
    max_context_tokens: int = Field(default=150_000, ge=1_000)
    timeout_s: float = Field(default=300.0, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_parallel_tools: int = Field(default=8, ge=1)


class ToolOutcome(BaseModel):
    """Result of executing (or refusing to execute) one tool call."""

    content: str
    is_error: bool = False
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PendingApproval(BaseModel):
    call: ToolCall
    reason: str
    requested_at: datetime = Field(default_factory=utcnow)


class ApprovalDecision(BaseModel):
    call_id: str
    approved: bool
    reviewer: str
    comment: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


EventType = Literal[
    "run_started",
    "think",
    "text_delta",
    "plan",
    "act",
    "observe",
    "reflect",
    "answer",
    "compacted",
    "approval_required",
    "approval_decided",
    "checkpoint",
    "run_finished",
    "error",
]


class RunEvent(BaseModel):
    type: EventType
    run_id: str
    step: int
    ts: datetime = Field(default_factory=utcnow)
    data: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    session_id: str = Field(default_factory=lambda: new_id("ses"))
    agent: str
    user_id: str = "anonymous"
    input: str
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    step: int = 0
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    status: RunStatus = RunStatus.PENDING
    stop_reason: StopReason | None = None
    answer: str | None = None
    error: str | None = None
    # Tool results already produced for the current step, keyed by call id. Lets a
    # resumed run skip calls that finished before a crash (idempotent replay).
    tool_results: dict[str, ToolOutcome] = Field(default_factory=dict)
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    decisions: dict[str, ApprovalDecision] = Field(default_factory=dict)
    parent_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def unanswered_calls(self) -> list[ToolCall]:
        """Tool calls in the last assistant turn that have no result message yet."""
        for idx in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[idx]
            if msg.role == "assistant":
                answered = {m.tool_call_id for m in self.messages[idx + 1 :] if m.role == "tool"}
                return [c for c in msg.tool_calls if c.id not in answered]
        return []


# ---- contracts implemented by other layers ------------------------------------


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    session_id: str
    user_id: str
    agent: str
    step: int


class ToolExecutor(Protocol):
    """Implemented by `harness.tools.ToolRegistry`."""

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]: ...

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome: ...


GuardVerdict = Literal["allow", "deny", "require_approval"]


class GuardDecision(BaseModel):
    verdict: GuardVerdict
    reason: str = ""


class ToolGuard(Protocol):
    """Implemented by `harness.safety.SafetyGuard`: policy checks before Act and
    output screening after Observe."""

    async def check_call(self, call: ToolCall, ctx: ToolContext) -> GuardDecision: ...

    async def screen_output(
        self, call: ToolCall, outcome: ToolOutcome, ctx: ToolContext
    ) -> ToolOutcome: ...


class StateStore(Protocol):
    """Implemented in `harness.state`."""

    async def save(self, state: RunState) -> None: ...

    async def load(self, run_id: str) -> RunState | None: ...


# Receives every run event (SSE streaming, tracing, audit, tests).
EventSink = Callable[[RunEvent], Awaitable[None]]
