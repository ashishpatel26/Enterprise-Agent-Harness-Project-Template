"""Agno engine: run an agent's reasoning loop with Agno, under harness governance.

Select it per agent with `engine: agno` in the YAML. Agno drives the conversation; the
harness keeps control of everything that matters for safety and operations:

- every tool call goes through the harness pipeline (schema validation, timeout,
  policy, approvals, injection screening, redaction, idempotency, audit, metrics);
- by default the model is the harness `LLMProvider` (retries, fallback, cost accounting,
  budgets, FakeLLM in tests) bridged into Agno's `Model` interface;
- optionally `agno.model: "anthropic:claude-opus-5-5"` (or any `module:Class:model_id`)
  uses one of Agno's ~60 native model integrations instead;
- events, checkpoints after every tool result, stop reasons and the run record are the
  same as the native engine; Agno telemetry is disabled.

Trade-offs vs the native engine (documented in docs/architecture.md): an Agno run is not
resumable mid-loop after a crash (it restarts from the task), and approvals hold the run
in-process until a decision arrives or `approval_timeout_s` passes.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agno.agent import Agent as AgnoAgent
from agno.models.base import Model
from agno.models.message import Message as AgnoMessage
from agno.models.response import ModelResponse
from agno.tools.function import Function

from harness.core.loop import AgentLoop
from harness.core.types import (
    ApprovalDecision,
    PendingApproval,
    RunState,
    RunStatus,
    StopReason,
    ToolOutcome,
    new_id,
)
from harness.llm.base import LLMProvider
from harness.llm.pricing import cost_usd
from harness.llm.types import LLMResponse, Message, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)

NATIVE_MODELS = {
    "anthropic": "agno.models.anthropic:Claude",
    "openai": "agno.models.openai:OpenAIChat",
    "google": "agno.models.google:Gemini",
    "groq": "agno.models.groq:Groq",
    "mistral": "agno.models.mistral:MistralChat",
    "ollama": "agno.models.ollama:Ollama",
    "bedrock": "agno.models.aws:AwsBedrock",
    "azure": "agno.models.azure:AzureOpenAI",
    "litellm": "agno.models.litellm:LiteLLM",
}


# (state, call, reason) -> decision, or None to fall back to waiting for the API.
InlineApprover = Callable[[RunState, ToolCall, str], Awaitable[ApprovalDecision | None]]


class StopRunSignal(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised inside the Agno loop to stop it with a harness stop reason."""

    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def load_native_model(ref: str) -> Model:
    """'anthropic:claude-opus-5-5' or 'agno.models.groq:Groq:llama-3.3-70b-versatile'."""
    parts = ref.split(":")
    if len(parts) == 2 and parts[0] in NATIVE_MODELS:
        module, cls = NATIVE_MODELS[parts[0]].split(":")
        model_id = parts[1]
    elif len(parts) == 3:
        module, cls, model_id = parts
    else:
        raise ValueError(
            f"invalid agno model {ref!r}; use '<provider>:<model>' "
            f"({', '.join(NATIVE_MODELS)}) or '<module>:<Class>:<model>'"
        )
    model_cls = getattr(importlib.import_module(module), cls)
    model: Model = model_cls(id=model_id)
    return model


def _agno_to_harness(messages: list[AgnoMessage], raw: dict[str, tuple[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        text = m.get_content_string() if hasattr(m, "get_content_string") else str(m.content or "")
        if m.role in ("system", "developer"):
            out.append(Message.system(text))
        elif m.role == "user":
            out.append(Message.user(text))
        elif m.role == "assistant":
            calls = [
                ToolCall(
                    id=c["id"],
                    name=c["function"]["name"],
                    arguments=json.loads(c["function"].get("arguments") or "{}"),
                )
                for c in (m.tool_calls or [])
            ]
            msg = Message.assistant(text, calls)
            # Re-attach the provider's raw blocks (e.g. Claude thinking) so they replay
            # unchanged within the tool loop.
            if calls and calls[0].id in raw:
                provider, blocks = raw[calls[0].id]
                msg = msg.model_copy(update={"raw_content": blocks, "raw_provider": provider})
            out.append(msg)
        elif m.role == "tool":
            out.append(
                Message(
                    role="tool",
                    content=text,
                    tool_call_id=m.tool_call_id,
                    name=m.tool_name,
                    is_error=bool(m.tool_call_error),
                )
            )
    return out


@dataclass
class HarnessModel(Model):
    """Agno `Model` backed by a harness `LLMProvider`."""

    id: str = "harness"
    name: str = "HarnessModel"
    provider: str = "harness"
    llm: Any = None
    loop: Any = None  # AgnoLoop, for budgets and events
    max_output_tokens: int = 16000
    raw_blocks: dict[str, tuple[str, Any]] = field(default_factory=dict)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        llm: LLMProvider = self.llm
        messages = _agno_to_harness(kwargs.get("messages") or [], self.raw_blocks)
        tools = [ToolSpec(**t["function"]) for t in kwargs.get("tools") or [] if "function" in t]
        await self.loop.before_model_call()
        resp: LLMResponse = await llm.chat(messages, tools, max_tokens=self.max_output_tokens)
        await self.loop.after_model_call(resp)
        calls = resp.message.tool_calls
        if calls and resp.message.raw_content is not None:
            self.raw_blocks[calls[0].id] = (
                resp.message.raw_provider or "",
                resp.message.raw_content,
            )
        return ModelResponse(
            role="assistant",
            content=resp.message.content,
            tool_calls=[
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in calls
            ],
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield await self.ainvoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError("the harness runs Agno asynchronously")

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the harness runs Agno asynchronously")

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response  # type: ignore[no-any-return]

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response  # type: ignore[no-any-return]


class AgnoLoop(AgentLoop):
    """Same contract as `AgentLoop.run`; the reasoning loop is delegated to Agno."""

    def __init__(
        self,
        *args: Any,
        native_model: str | None = None,
        approval_timeout_s: float = 600.0,
        approver: InlineApprover | None = None,
        approval_poll_s: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.native_model = native_model
        self.approval_timeout_s = approval_timeout_s
        self.approval_poll_s = approval_poll_s
        self.approver = approver
        self._state: RunState | None = None
        self._stop: StopReason | None = None

    # ---- model hooks (budgets + events) ----------------------------------------------
    async def before_model_call(self) -> None:
        state = self._require_state()
        stop = self._budget_exhausted(state)
        if stop is None and state.step >= self.limits.max_steps:
            stop = StopReason.MAX_STEPS
        if stop is not None:
            # Recorded as well as raised: Agno may wrap exceptions from the model call.
            self._stop = stop
            raise StopRunSignal(stop)
        state.step += 1
        await self._emit(state, "think", model=self.llm.model, engine="agno")

    async def after_model_call(self, resp: LLMResponse) -> None:
        state = self._require_state()
        state.usage = state.usage + resp.usage
        state.cost_usd += resp.cost_usd
        if resp.message.content:
            # Non-streaming bridge: deliver each turn's text as one delta for SSE/CLI.
            await self._emit(state, "text_delta", text=resp.message.content)
        await self._emit(
            state,
            "llm_call",
            provider=resp.provider,
            model=resp.model,
            finish_reason=resp.finish_reason,
            latency_ms=0,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=resp.usage.cache_read_tokens,
            cache_write_tokens=resp.usage.cache_write_tokens,
            cost_usd=resp.cost_usd,
            tool_calls=[c.name for c in resp.message.tool_calls],
        )

    def _require_state(self) -> RunState:
        assert self._state is not None
        return self._state

    # ---- tools through the harness pipeline -----------------------------------------
    def _function(self, spec: ToolSpec) -> Function:
        async def entry(**arguments: Any) -> str:
            return await self._call_tool(
                ToolCall(id=new_id("call"), name=spec.name, arguments=arguments)
            )

        return Function(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            entrypoint=entry,
            skip_entrypoint_processing=True,
        )

    async def _call_tool(self, call: ToolCall) -> str:
        state = self._require_state()
        ctx = self._ctx(state).model_copy(update={"call_id": call.id})
        await self._emit(state, "plan", tools=[call.name])
        verdict = await self.guard.check_call(call, ctx) if self.guard else None
        outcome: ToolOutcome
        if verdict is not None and verdict.verdict == "deny":
            outcome = ToolOutcome(content=f"Blocked by policy: {verdict.reason}", is_error=True)
        elif verdict is not None and verdict.verdict == "require_approval":
            decision = await self._await_approval(state, call, verdict.reason)
            if decision is None or not decision.approved:
                who = f" {decision.reviewer}" if decision else ""
                note = f": {decision.comment}" if decision and decision.comment else ""
                why = "Approval timed out" if decision is None else f"Denied by human reviewer{who}"
                outcome = ToolOutcome(content=f"{why}{note}", is_error=True)
            else:
                outcome = await self._execute(call, ctx)
        else:
            outcome = await self._execute(call, ctx)

        if isinstance(spent := outcome.metadata.get("usage"), dict):
            state.usage = state.usage + Usage.model_validate(spent)
        state.cost_usd += float(outcome.metadata.get("cost_usd") or 0.0)
        await self._emit(
            state,
            "observe",
            tool=call.name,
            call_id=call.id,
            is_error=outcome.is_error,
            duration_ms=outcome.duration_ms,
            preview=outcome.content[:200],
        )
        await self._checkpoint(state)
        return f"Error: {outcome.content}" if outcome.is_error else outcome.content

    async def _execute(self, call: ToolCall, ctx: Any) -> ToolOutcome:
        state = self._require_state()
        await self._emit(state, "act", tool=call.name, call_id=call.id)
        outcome = await self.tools.execute(call, ctx)
        if self.guard:
            outcome = await self.guard.screen_output(call, outcome, ctx)
        return outcome

    async def _await_approval(
        self, state: RunState, call: ToolCall, reason: str
    ) -> ApprovalDecision | None:
        if self.store is None:
            return None
        state.pending_approvals = [PendingApproval(call=call, reason=reason)]
        state.status = RunStatus.AWAITING_APPROVAL
        await self._checkpoint(state)
        await self._emit(
            state,
            "approval_required",
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            reason=reason,
        )
        decision: ApprovalDecision | None = None
        if self.approver is not None:
            # In-process reviewer (CLI prompt, auto-approve in evals) answers directly.
            decision = await self.approver(state, call, reason)
        deadline = asyncio.get_running_loop().time() + self.approval_timeout_s
        while decision is None and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(self.approval_poll_s)
            fresh = await self.store.load(state.run_id)
            if fresh is not None and call.id in fresh.decisions:
                decision = fresh.decisions[call.id]
                break
        state.pending_approvals = []
        state.decisions.pop(call.id, None)
        state.status = RunStatus.RUNNING
        await self._checkpoint(state)
        return decision

    # ---- the run -------------------------------------------------------------------
    async def _drive(self, state: RunState) -> None:
        self._state = state
        state.metadata["engine"] = "agno"
        # Agno runs restart from the task after a crash: replay only the original input.
        base = int(state.metadata.setdefault("agno_base", len(state.messages)))
        del state.messages[base:]
        system = next((m.content for m in state.messages if m.role == "system"), "")
        history = [
            AgnoMessage(role=m.role, content=m.content)
            for m in state.messages
            if m.role in ("user", "assistant") and m.content
        ]
        if self.native_model:
            model: Model = load_native_model(self.native_model)
        else:
            model = HarnessModel(llm=self.llm, loop=self, max_output_tokens=self.max_output_tokens)
        agent = AgnoAgent(
            model=model,
            name=state.agent,
            instructions=system,
            tools=[self._function(spec) for spec in state.tools],
            tool_call_limit=self.limits.max_steps * self.limits.max_parallel_tools,
            telemetry=False,
            markdown=False,
        )
        try:
            out = await agent.arun(history, user_id=state.user_id, session_id=state.session_id)
        except StopRunSignal as stop:
            self._finish(state, stop.reason)
            return
        if self._stop is not None:
            self._finish(state, self._stop)
            return
        status = str(getattr(out.status, "value", out.status)).lower()
        if "error" in status:
            raise RuntimeError(f"agno run failed: {out.content}")

        produced = (out.messages or [])[len(history) :]
        for msg in _agno_to_harness([m for m in produced if m.role != "system"], {}):
            state.messages.append(msg)
        answer = next(
            (
                m.content
                for m in reversed(state.messages[base:])
                if m.role == "assistant" and not m.tool_calls
            ),
            str(out.content or ""),
        )
        if self.native_model and out.metrics is not None:
            usage = Usage(
                input_tokens=int(out.metrics.input_tokens or 0),
                output_tokens=int(out.metrics.output_tokens or 0),
            )
            state.usage = state.usage + usage
            state.cost_usd += cost_usd(self.native_model.split(":")[-1], usage)
            state.step += 1
        self._finish(state, StopReason.COMPLETED, answer=answer)
