"""Human approval for gated tool calls.

The loop pauses a run (status `awaiting_approval`) when policy requires approval. A
reviewer then approves or denies each pending call through the API; once no call is
pending, the run is ready to resume. Decisions are validated and audited.
"""

from __future__ import annotations

from harness.core.types import ApprovalDecision, RunState, RunStatus, StateStore
from harness.safety.audit import AuditLog


class ApprovalError(Exception):
    """Invalid approval request (unknown run/call, run not awaiting approval)."""


class ApprovalService:
    def __init__(
        self,
        store: StateStore,
        audit: AuditLog | None = None,
        *,
        allow_self_approval: bool = True,
    ) -> None:
        self.store = store
        self.audit = audit
        # False enforces four-eyes review: the user who started the run cannot approve it.
        self.allow_self_approval = allow_self_approval

    async def decide(
        self,
        run_id: str,
        call_id: str,
        *,
        approved: bool,
        reviewer: str,
        comment: str = "",
    ) -> RunState:
        """Record one decision. Returns the updated state; when `pending_approvals` is
        empty afterwards the caller should resume the run."""
        state = await self.store.load(run_id)
        if state is None:
            raise ApprovalError(f"run {run_id} not found")
        if state.status != RunStatus.AWAITING_APPROVAL:
            raise ApprovalError(f"run {run_id} is {state.status.value}, not awaiting approval")
        pending = {p.call.id: p for p in state.pending_approvals}
        if call_id not in pending:
            raise ApprovalError(f"call {call_id} is not pending approval")
        if approved and not self.allow_self_approval and reviewer == state.user_id:
            raise ApprovalError("separation of duties: requesters cannot approve their own calls")
        state.decisions[call_id] = ApprovalDecision(
            call_id=call_id, approved=approved, reviewer=reviewer, comment=comment
        )
        state.pending_approvals = [p for p in state.pending_approvals if p.call.id != call_id]
        await self.store.save(state)
        if self.audit is not None:
            call = pending[call_id].call
            await self.audit.record(
                "approval.decided",
                actor=reviewer,
                run_id=run_id,
                call_id=call_id,
                tool=call.name,
                arguments=call.arguments,
                approved=approved,
                comment=comment,
            )
        return state
