"""The agent loop (ReAct): Think -> Plan -> Act -> Observe -> Reflect -> Answer.

The loop is a resumable state machine over `RunState`. Where to continue is derived from
the messages themselves: if the last assistant turn has unanswered tool calls, the run
resumes at Act; otherwise it Thinks. Combined with a checkpoint after every step and every
tool result, a run survives crashes and approval pauses without repeating finished work.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from harness.core.types import (
    EventSink,
    EventType,
    GuardDecision,
    PendingApproval,
    RunEvent,
    RunLimits,
    RunState,
    RunStatus,
    StateStore,
    StopReason,
    ToolContext,
    ToolExecutor,
    ToolGuard,
    ToolOutcome,
    utcnow,
)
from harness.llm.base import LLMError, LLMProvider
from harness.llm.types import LLMResponse, Message, ToolCall, Usage
from harness.observability.tracing import span
from harness.prompts.trimming import compact

log = logging.getLogger(__name__)

_ALLOW = GuardDecision(verdict="allow")


class AgentLoop:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolExecutor,
        *,
        limits: RunLimits | None = None,
        max_output_tokens: int = 16000,
        guard: ToolGuard | None = None,
        store: StateStore | None = None,
        sinks: Sequence[EventSink] = (),
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.limits = limits or RunLimits()
        self.max_output_tokens = max_output_tokens
        self.guard = guard
        self.store = store
        self.sinks = list(sinks)
        self._save_lock = asyncio.Lock()

    # ---- public -----------------------------------------------------------
    async def run(self, state: RunState) -> RunState:
        """Drive the run until it finishes or pauses. Never raises for run-level failures:
        every exit sets `status` and `stop_reason`. Task cancellation is re-raised."""
        if state.status.terminal:
            return state
        if not state.messages:
            raise ValueError("RunState has no messages; build them with PromptAssembler")
        first_start = state.status == RunStatus.PENDING
        state.status = RunStatus.RUNNING
        state.stop_reason = None
        if first_start:
            await self._emit(state, "run_started", agent=state.agent, input=state.input)

        started = time.monotonic()
        remaining = self.limits.timeout_s - state.elapsed_s
        try:
            with span("agent.run", run_id=state.run_id, agent=state.agent):
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    await self._drive(state)
        except TimeoutError:
            self._finish(state, StopReason.TIMEOUT, error=f"exceeded {self.limits.timeout_s}s")
        except LLMError as exc:
            self._finish(state, StopReason.LLM_ERROR, error=f"{exc.provider}: {exc}")
        except asyncio.CancelledError:
            self._finish(state, StopReason.CANCELLED, error="cancelled")
            state.elapsed_s += time.monotonic() - started
            await self._checkpoint(state)
            await self._emit(state, "run_finished", stop_reason=state.stop_reason)
            raise
        except Exception as exc:
            log.exception("run failed", extra={"run_id": state.run_id})
            self._finish(state, StopReason.ERROR, error=f"{type(exc).__name__}: {exc}")
        state.elapsed_s += time.monotonic() - started
        await self._checkpoint(state)
        if state.status.terminal:
            await self._emit(
                state,
                "run_finished",
                stop_reason=state.stop_reason,
                status=state.status,
                steps=state.step,
                total_tokens=state.usage.total_tokens,
                cost_usd=round(state.cost_usd, 6),
            )
        return state

    # ---- state machine ----------------------------------------------------
    async def _drive(self, state: RunState) -> None:
        while True:
            pending = state.unanswered_calls()
            if pending:
                if await self._act(state, pending):
                    return  # paused for approval
                await self._reflect(state)
                if state.status.terminal:
                    return
                continue

            if stop := self._budget_exhausted(state):
                self._finish(state, stop)
                return
            if state.step >= self.limits.max_steps:
                self._finish(state, StopReason.MAX_STEPS)
                return

            await self._maybe_compact(state)
            response = await self._think(state)
            if self._plan_or_answer(state, response):
                return

    async def _think(self, state: RunState) -> LLMResponse:
        state.step += 1
        await self._emit(state, "think", model=self.llm.model)
        response: LLMResponse | None = None
        started = time.perf_counter()
        with span("agent.think", run_id=state.run_id, step=state.step, model=self.llm.model) as s:
            async for event in self.llm.stream(
                state.messages, state.tools, max_tokens=self.max_output_tokens
            ):
                if event.type == "text_delta":
                    await self._emit(state, "text_delta", text=event.text)
                elif event.type == "done":
                    response = event.response
            if response is None:
                raise LLMError("stream ended without a final response", provider=self.llm.name)
            # OpenTelemetry GenAI semantic-convention attributes.
            s.set_attribute("gen_ai.system", response.provider)
            s.set_attribute("gen_ai.response.model", response.model)
            s.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
            s.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
            s.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
        await self._emit(
            state,
            "llm_call",
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_write_tokens=response.usage.cache_write_tokens,
            cost_usd=response.cost_usd,
            tool_calls=[c.name for c in response.message.tool_calls],
        )
        state.usage = state.usage + response.usage
        state.cost_usd += response.cost_usd
        state.messages.append(response.message)
        await self._checkpoint(state)
        return response

    def _plan_or_answer(self, state: RunState, response: LLMResponse) -> bool:
        """Return True when the run is finished."""
        msg = response.message
        if msg.tool_calls:
            state.metadata["last_plan"] = [c.name for c in msg.tool_calls]
            return False
        if response.finish_reason == "refusal":
            self._finish(state, StopReason.REFUSED, answer=msg.content)
        elif response.finish_reason == "max_tokens":
            self._finish(state, StopReason.OUTPUT_TRUNCATED, answer=msg.content)
        else:
            self._finish(state, StopReason.COMPLETED, answer=msg.content)
        return True

    async def _act(self, state: RunState, calls: list[ToolCall]) -> bool:
        """Execute the unanswered calls. Returns True if the run paused for approval."""
        await self._emit(state, "plan", tools=[c.name for c in calls])
        ctx = self._ctx(state)
        runnable: list[ToolCall] = []
        needs_approval: list[PendingApproval] = []

        for call in calls:
            if call.id in state.tool_results:
                continue
            decision = state.decisions.get(call.id)
            if decision is not None:
                if decision.approved:
                    runnable.append(call)
                else:
                    note = f": {decision.comment}" if decision.comment else ""
                    state.tool_results[call.id] = ToolOutcome(
                        content=f"Denied by human reviewer {decision.reviewer}{note}",
                        is_error=True,
                        metadata={"denied": True},
                    )
                continue
            call_ctx = ctx.model_copy(update={"call_id": call.id})
            verdict = await self.guard.check_call(call, call_ctx) if self.guard else _ALLOW
            if verdict.verdict == "deny":
                state.tool_results[call.id] = ToolOutcome(
                    content=f"Blocked by policy: {verdict.reason}",
                    is_error=True,
                    metadata={"blocked": True},
                )
            elif verdict.verdict == "require_approval":
                needs_approval.append(PendingApproval(call=call, reason=verdict.reason))
            else:
                runnable.append(call)

        sem = asyncio.Semaphore(self.limits.max_parallel_tools)

        async def run_one(call: ToolCall) -> None:
            async with sem:
                await self._emit(state, "act", tool=call.name, call_id=call.id)
                call_ctx = ctx.model_copy(update={"call_id": call.id})
                with span("agent.tool", run_id=state.run_id, tool=call.name, call_id=call.id):
                    outcome = await self.tools.execute(call, call_ctx)
                    if self.guard:
                        outcome = await self.guard.screen_output(call, outcome, call_ctx)
                state.tool_results[call.id] = outcome
                await self._checkpoint(state)

        await asyncio.gather(*(run_one(c) for c in runnable))

        if needs_approval:
            state.pending_approvals = needs_approval
            state.status = RunStatus.AWAITING_APPROVAL
            state.stop_reason = StopReason.AWAITING_APPROVAL
            await self._checkpoint(state)
            for p in needs_approval:
                await self._emit(
                    state,
                    "approval_required",
                    call_id=p.call.id,
                    tool=p.call.name,
                    arguments=p.call.arguments,
                    reason=p.reason,
                )
            return True

        # Observe: append results in call order, as one batch, then clear the scratchpad.
        for call in calls:
            outcome = state.tool_results.pop(call.id)
            state.decisions.pop(call.id, None)
            # Tools that spend model tokens on the run's behalf (sub-agent delegation)
            # report it, so budgets and cost accounting cover the whole tree.
            if isinstance(spent := outcome.metadata.get("usage"), dict):
                state.usage = state.usage + Usage.model_validate(spent)
            state.cost_usd += float(outcome.metadata.get("cost_usd") or 0.0)
            state.messages.append(Message.tool(call, outcome.content, is_error=outcome.is_error))
            await self._emit(
                state,
                "observe",
                tool=call.name,
                call_id=call.id,
                is_error=outcome.is_error,
                duration_ms=outcome.duration_ms,
                preview=outcome.content[:200],
            )
        state.pending_approvals = []
        await self._checkpoint(state)
        return False

    async def _reflect(self, state: RunState) -> None:
        recent = [m for m in state.messages[-10:] if m.role == "tool"]
        await self._emit(
            state,
            "reflect",
            total_tokens=state.usage.total_tokens,
            cost_usd=round(state.cost_usd, 6),
            recent_tool_errors=sum(m.is_error for m in recent),
            steps_left=self.limits.max_steps - state.step,
        )
        if stop := self._budget_exhausted(state):
            self._finish(state, stop)

    async def _maybe_compact(self, state: RunState) -> None:
        compacted = compact(state.messages, self.limits.max_context_tokens)
        if compacted is None:
            return
        before = len(state.messages)
        state.messages = compacted
        await self._emit(state, "compacted", before=before, after=len(compacted))
        await self._checkpoint(state)

    # ---- helpers ----------------------------------------------------------
    def _budget_exhausted(self, state: RunState) -> StopReason | None:
        if state.usage.total_tokens >= self.limits.max_total_tokens:
            return StopReason.TOKEN_BUDGET
        if self.limits.max_cost_usd is not None and state.cost_usd >= self.limits.max_cost_usd:
            return StopReason.COST_BUDGET
        return None

    @staticmethod
    def _finish(
        state: RunState, reason: StopReason, *, answer: str | None = None, error: str | None = None
    ) -> None:
        state.stop_reason = reason
        if answer is not None:
            state.answer = answer
        if error is not None:
            state.error = error
        if reason == StopReason.COMPLETED:
            state.status = RunStatus.COMPLETED
        elif reason == StopReason.CANCELLED:
            state.status = RunStatus.CANCELLED
        else:
            state.status = RunStatus.FAILED

    @staticmethod
    def _ctx(state: RunState) -> ToolContext:
        return ToolContext(
            run_id=state.run_id,
            session_id=state.session_id,
            user_id=state.user_id,
            agent=state.agent,
            step=state.step,
        )

    async def _checkpoint(self, state: RunState) -> None:
        if self.store is None:
            return
        async with self._save_lock:
            state.updated_at = utcnow()
            await self.store.save(state)

    async def _emit(self, state: RunState, type_: EventType, **data: Any) -> None:
        event = RunEvent(type=type_, run_id=state.run_id, step=state.step, data=data)
        for sink in self.sinks:
            try:
                await sink(event)
            except Exception:
                log.exception("event sink failed", extra={"event": type_})
