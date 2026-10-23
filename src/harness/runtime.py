"""`Harness`: the composition root. Builds every layer from settings and runs agents.

The API, the CLI and the eval runner all go through this class, so a run behaves the
same everywhere: input validation, policy, approvals, checkpoints, leases, memory,
events, metrics and audit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.config import Settings, get_settings
from harness.core.agent import Agent
from harness.core.agent_spec import AgentSpec, load_agents
from harness.core.types import EventSink, RunEvent, RunState, RunStatus, StopReason
from harness.events import EventBus, InMemoryEventBus, RedisEventBus
from harness.llm.base import LLMProvider
from harness.llm.factory import build_llm
from harness.memory import MemoryManager, PgVectorMemory, build_memory
from harness.observability.logging import run_id_var
from harness.observability.metrics import RUN_DURATION, RUNS_ACTIVE, MetricsSink
from harness.prompts.assembler import PromptAssembler
from harness.safety import (
    ApprovalService,
    AuditLog,
    AuditSink,
    FileAuditLog,
    InMemoryRateLimiter,
    Policy,
    RateLimiter,
    Redactor,
    RedisAuditLog,
    RedisRateLimiter,
    SafetyGuard,
    validate_user_input,
)
from harness.sandbox import Sandbox, SandboxError, build_sandbox
from harness.state import IdempotentExecutor, StateBackends, build_state_backends, make_redis
from harness.tools import ToolRegistry
from harness.tools.builtin import register_builtin_tools

log = logging.getLogger(__name__)

LEASE_TTL_S = 30.0

# Hook for sub-systems (orchestration) that add tools needing the Harness itself.
ToolInstaller = Callable[["Harness", ToolRegistry], None]


class RunNotFoundError(LookupError):
    pass


class RunConflictError(RuntimeError):
    """The run is being driven elsewhere or is in a state that forbids the action."""


@dataclass
class Harness:
    settings: Settings
    specs: dict[str, AgentSpec]
    registry: ToolRegistry
    sandbox: Sandbox | None
    state: StateBackends
    memory: MemoryManager
    policy: Policy
    guard: SafetyGuard
    audit: AuditLog
    approvals: ApprovalService
    events: EventBus
    rate_limiter: RateLimiter
    redactor: Redactor
    llm_overrides: dict[str, LLMProvider] = field(default_factory=dict)
    worker_id: str = field(
        default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
    )
    _agents: dict[str, Agent] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[RunState]] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    # ---- construction ---------------------------------------------------------
    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        sandbox: Sandbox | None = None,
        llm_overrides: dict[str, LLMProvider] | None = None,
        extra_tools: Sequence[ToolInstaller] = (),
    ) -> Harness:
        """Build from settings. `llm_overrides` maps agent name -> provider (tests, demos)."""
        s = settings or get_settings()
        specs = load_agents(s.agents_dir) if s.agents_dir.is_dir() else {}

        if sandbox is None:
            try:
                sandbox = build_sandbox(s)
            except SandboxError:
                log.warning("sandbox unavailable; python_exec disabled", exc_info=True)
        redactor = Redactor(
            v.get_secret_value()
            for v in (
                s.anthropic_api_key,
                s.openai_api_key,
                s.database_url,
                s.redis_url,
                *s.api_keys,
            )
            if v is not None
        )
        registry = register_builtin_tools(ToolRegistry(), s, sandbox)
        registry.load_plugins()

        state = build_state_backends(s)
        memory = build_memory(s)
        policy = Policy.from_yaml(s.policy_path) if s.policy_path.is_file() else Policy()
        if s.storage_backend == "redis":
            r = make_redis(s.redis_url.get_secret_value())
            audit: AuditLog = RedisAuditLog(r, redactor=redactor)
            rate_limiter: RateLimiter = RedisRateLimiter(r)
            events: EventBus = RedisEventBus(
                make_redis(s.redis_url.get_secret_value(), timeout_s=30)
            )
        else:
            audit = FileAuditLog(s.audit_log_path, redactor)
            rate_limiter = InMemoryRateLimiter()
            events = InMemoryEventBus()
        guard = SafetyGuard(
            registry,
            policy,
            redactor=redactor,
            rate_limiter=rate_limiter,
            tool_calls_per_minute=s.tool_calls_per_minute,
            audit=audit,
            injection_action=s.injection_action,
        )
        harness = cls(
            settings=s,
            specs=specs,
            registry=registry,
            sandbox=sandbox,
            state=state,
            memory=memory,
            policy=policy,
            guard=guard,
            audit=audit,
            approvals=ApprovalService(state.runs, audit, allow_self_approval=s.allow_self_approval),
            events=events,
            rate_limiter=rate_limiter,
            redactor=redactor,
            llm_overrides=dict(llm_overrides or {}),
        )
        from harness.orchestration import install_delegation

        install_delegation(harness, registry)
        for install in extra_tools:
            install(harness, registry)
        harness.validate()
        return harness

    def validate(self) -> None:
        """Fail fast at startup on agent definitions that cannot run."""
        problems: list[str] = []
        known = set(self.registry.names())
        for spec in self.specs.values():
            for tool in spec.tools:
                if tool not in known:
                    hint = " (no sandbox configured)" if tool == "python_exec" else ""
                    problems.append(f"{spec.name}: unknown tool {tool!r}{hint}")
            for sub in spec.sub_agents:
                if sub not in self.specs:
                    problems.append(f"{spec.name}: unknown sub-agent {sub!r}")
                elif sub == spec.name:
                    problems.append(f"{spec.name}: an agent cannot delegate to itself")
            if spec.sub_agents and "delegate" not in spec.tools:
                problems.append(f"{spec.name}: has sub_agents but not the 'delegate' tool")
        if problems:
            raise ValueError("invalid agent definitions:\n- " + "\n- ".join(problems))

    async def startup(self) -> None:
        """One-time async initialization (schema, crash recovery)."""
        if isinstance(self.memory.long_term, PgVectorMemory):
            await self.memory.long_term.ensure_schema()
        await self.recover()

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        if self.sandbox is not None:
            await self.sandbox.aclose()
        for agent in self._agents.values():
            await agent.llm.aclose()

    # ---- agents ---------------------------------------------------------------
    def agent(self, name: str) -> Agent:
        if name in self._agents:
            return self._agents[name]
        spec = self.specs.get(name)
        if spec is None:
            raise KeyError(f"unknown agent {name!r}; available: {sorted(self.specs)}")
        if spec.sub_agents:
            # Tell the supervisor who it can delegate to; the prompt stays frozen per run.
            roster = "\n".join(
                f"- {sub}: {self.specs[sub].description or 'no description'}"
                for sub in spec.sub_agents
            )
            instructions = (
                f"{spec.prompt.instructions}\n\n## Sub-agents you can delegate to\n{roster}"
            )
            spec = spec.model_copy(
                update={"prompt": spec.prompt.model_copy(update={"instructions": instructions})}
            )
        llm = self.llm_overrides.get(name) or build_llm(spec.model, self.settings)
        tools = IdempotentExecutor(self.registry, self.state.idempotency)
        agent = Agent(
            spec,
            llm,
            tools,
            assembler=PromptAssembler(self.settings.prompts_dir),
            memory=self.memory,
            guard=self.guard,
            store=self.state.runs,
        )
        self._agents[name] = agent
        return agent

    def _sinks(self, agent: str) -> list[EventSink]:
        async def publish(event: RunEvent) -> None:
            await self.events.publish(event)

        return [publish, MetricsSink(agent), AuditSink(self.audit)]

    # ---- runs -----------------------------------------------------------------
    async def create_run(
        self,
        agent_name: str,
        user_input: str,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
        owner: str | None = None,
        parent_run_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunState:
        text = validate_user_input(user_input, max_chars=self.settings.max_input_chars)
        agent = self.agent(agent_name)
        state = await agent.new_run(
            text,
            user_id=user_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
            run_id=run_id,
            metadata=metadata,
        )
        state.metadata["owner"] = owner or user_id
        await self.state.runs.save(state)
        return state

    async def execute(self, run_id: str) -> RunState:
        """Drive a run in this task until it finishes or pauses. Holds a lease so no
        other worker drives the same run concurrently."""
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            if not await self.state.leases.acquire(run_id, self.worker_id, LEASE_TTL_S):
                raise RunConflictError(f"run {run_id} is being executed by another worker")
            renew = asyncio.create_task(self._renew_lease(run_id))
            token = run_id_var.set(run_id)
            RUNS_ACTIVE.inc()
            try:
                state = await self.state.runs.load(run_id)
                if state is None:
                    raise RunNotFoundError(run_id)
                agent = self.agent(state.agent)
                with RUN_DURATION.labels(state.agent).time():
                    return await agent.execute(state, self._sinks(state.agent))
            finally:
                RUNS_ACTIVE.dec()
                run_id_var.reset(token)
                renew.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await renew
                await self.state.leases.release(run_id, self.worker_id)

    async def _renew_lease(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(LEASE_TTL_S / 3)
            if not await self.state.leases.renew(run_id, self.worker_id, LEASE_TTL_S):
                log.error("lost lease while executing run", extra={"run_id": run_id})
                return

    def start(self, run_id: str) -> asyncio.Task[RunState]:
        """Execute in the background (API). The task is tracked for cancel/shutdown."""
        task = asyncio.create_task(self.execute(run_id), name=f"run:{run_id}")
        self._tasks[run_id] = task

        def _done(t: asyncio.Task[RunState]) -> None:
            self._tasks.pop(run_id, None)
            self._locks.pop(run_id, None)
            if not t.cancelled() and (exc := t.exception()) is not None:
                log.error("background run failed", extra={"run_id": run_id}, exc_info=exc)

        task.add_done_callback(_done)
        return task

    async def run(
        self,
        agent_name: str,
        user_input: str,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> RunState:
        """Create and execute to completion (or pause) in the caller's task."""
        state = await self.create_run(
            agent_name, user_input, user_id=user_id, session_id=session_id
        )
        return await self.execute(state.run_id)

    async def get(self, run_id: str) -> RunState:
        state = await self.state.runs.load(run_id)
        if state is None:
            raise RunNotFoundError(run_id)
        return state

    async def decide(
        self, run_id: str, call_id: str, *, approved: bool, reviewer: str, comment: str = ""
    ) -> RunState:
        """Record an approval decision; resume in the background once nothing is pending."""
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            state = await self.approvals.decide(
                run_id, call_id, approved=approved, reviewer=reviewer, comment=comment
            )
        await self.events.publish(
            RunEvent(
                type="approval_decided",
                run_id=run_id,
                step=state.step,
                data={"call_id": call_id, "approved": approved, "reviewer": reviewer},
            )
        )
        if not state.pending_approvals:
            state.status = RunStatus.RUNNING
            await self.state.runs.save(state)
            self.start(run_id)
        return state

    async def cancel(self, run_id: str) -> RunState:
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return await self.get(run_id)
        state = await self.get(run_id)
        if state.status.terminal:
            return state
        if state.status == RunStatus.RUNNING:
            # ponytail: cross-replica cancel needs a cancel flag the owning worker polls.
            raise RunConflictError("run is executing on another worker")
        if not await self.state.leases.acquire(run_id, self.worker_id, LEASE_TTL_S):
            raise RunConflictError("run is executing on another worker")
        try:
            state.status = RunStatus.CANCELLED
            state.stop_reason = StopReason.CANCELLED
            state.pending_approvals = []
            await self.state.runs.save(state)
            await self.events.publish(
                RunEvent(
                    type="run_finished",
                    run_id=run_id,
                    step=state.step,
                    data={"stop_reason": "cancelled", "status": "cancelled"},
                )
            )
        finally:
            await self.state.leases.release(run_id, self.worker_id)
        return state

    async def recover(self) -> list[str]:
        """Resume runs left RUNNING by a crashed worker (their lease has expired)."""
        resumed: list[str] = []
        for state in await self.state.runs.list_runs(status=RunStatus.RUNNING, limit=1000):
            if state.run_id in self._tasks:
                continue
            if await self.state.leases.acquire(state.run_id, self.worker_id, LEASE_TTL_S):
                await self.state.leases.release(state.run_id, self.worker_id)
                log.info("recovering interrupted run", extra={"run_id": state.run_id})
                self.start(state.run_id)
                resumed.append(state.run_id)
        return resumed
