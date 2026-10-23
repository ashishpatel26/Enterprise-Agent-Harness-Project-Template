"""`SafetyGuard`: the loop's ToolGuard. Combines rate limits, policy, output screening,
secret redaction and auditing around every tool call."""

from __future__ import annotations

from typing import Literal

from harness.core.types import GuardDecision, ToolContext, ToolOutcome
from harness.llm.types import ToolCall
from harness.safety.audit import AuditLog
from harness.safety.injection import annotate, scan, strip_invisible
from harness.safety.policy import Policy
from harness.safety.ratelimit import RateLimiter
from harness.safety.redaction import Redactor
from harness.tools.registry import ToolRegistry

InjectionAction = Literal["annotate", "block"]


class SafetyGuard:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: Policy,
        *,
        redactor: Redactor | None = None,
        rate_limiter: RateLimiter | None = None,
        tool_calls_per_minute: int = 60,
        audit: AuditLog | None = None,
        injection_action: InjectionAction = "annotate",
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.redactor = redactor or Redactor()
        self.rate_limiter = rate_limiter
        self.tool_calls_per_minute = tool_calls_per_minute
        self.audit = audit
        self.injection_action = injection_action

    async def check_call(self, call: ToolCall, ctx: ToolContext) -> GuardDecision:
        if self.rate_limiter is not None:
            rl = await self.rate_limiter.hit(
                f"tools:{ctx.user_id}", self.tool_calls_per_minute, 60.0
            )
            if not rl.allowed:
                decision = GuardDecision(
                    verdict="deny",
                    reason=f"tool rate limit exceeded; retry in {rl.retry_after_s:.0f}s",
                )
                await self._audit_decision(call, ctx, decision)
                return decision

        tool = self.registry.get(call.name)
        decision = self.policy.evaluate(
            tool=call.name,
            permission=tool.permission.value if tool else None,
            agent=ctx.agent,
            user=ctx.user_id,
            arguments=call.arguments,
        )
        await self._audit_decision(call, ctx, decision)
        return decision

    async def screen_output(
        self, call: ToolCall, outcome: ToolOutcome, ctx: ToolContext
    ) -> ToolOutcome:
        # Secrets read by a tool (a .env file, a config dump) must not flow to the model
        # provider or into transcripts.
        content = self.redactor.redact(outcome.content)
        report = scan(content)
        meta = dict(outcome.metadata)
        if report.signals:
            meta["injection_signals"] = report.signals
            meta["injection_score"] = report.score
        if report.suspicious:
            if self.injection_action == "block":
                content = (
                    "[HARNESS: tool output withheld because it matched prompt-injection "
                    f"heuristics ({', '.join(report.signals)}).]"
                )
                outcome = outcome.model_copy(update={"is_error": True})
            else:
                content = annotate(content, report)
            if self.audit is not None:
                await self.audit.record(
                    "tool.injection_suspected",
                    actor="harness",
                    run_id=ctx.run_id,
                    call_id=ctx.call_id,
                    tool=call.name,
                    signals=report.signals,
                    score=report.score,
                    mode=self.injection_action,
                )
        elif report.invisible_chars:
            content = strip_invisible(content)
        return outcome.model_copy(update={"content": content, "metadata": meta})

    async def _audit_decision(
        self, call: ToolCall, ctx: ToolContext, decision: GuardDecision
    ) -> None:
        if self.audit is None:
            return
        await self.audit.record(
            f"tool.{decision.verdict}",
            actor=ctx.user_id,
            run_id=ctx.run_id,
            agent=ctx.agent,
            call_id=ctx.call_id,
            tool=call.name,
            arguments=call.arguments,
            reason=decision.reason,
        )
