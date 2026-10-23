"""Supervisor -> sub-agent delegation.

A supervisor is an ordinary agent whose spec lists `sub_agents` and includes the
`delegate` tool. Each delegation starts a child run with an isolated context (a fresh
prompt: the child never sees the supervisor's conversation) while sharing the harness's
tools, policy, memory and tenant. Children are normal runs: checkpointed, traced,
audited, visible through the API with `parent_run_id` set.

Child run ids are derived from (parent run, tool call), so a supervisor that crashes
mid-delegation resumes the same child instead of spawning a duplicate. Token usage and
cost of the child roll up into the parent's budget.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from harness.core.types import RunStatus, ToolContext, ToolOutcome
from harness.tools.registry import Permission, ToolError, ToolRegistry, make_tool

if TYPE_CHECKING:
    from harness.runtime import Harness

MAX_DEPTH = 3
MAX_TASK_CHARS = 20_000


def child_run_id(parent_run_id: str, call_id: str) -> str:
    return "run_" + hashlib.sha256(f"{parent_run_id}:{call_id}".encode()).hexdigest()[:32]


def install_delegation(harness: Harness, registry: ToolRegistry) -> None:
    async def delegate(agent: str, task: str, ctx: ToolContext) -> ToolOutcome:
        """Delegate a self-contained task to a specialist sub-agent and return its answer.

        `agent` must be one of the sub-agents listed in your instructions. `task` must be
        complete on its own: the sub-agent does not see this conversation. Independent
        tasks can be delegated in parallel by calling this tool several times at once.
        """
        caller = harness.specs.get(ctx.agent)
        if caller is None or agent not in caller.sub_agents:
            allowed = ", ".join(caller.sub_agents) if caller else ""
            raise ToolError(f"{agent!r} is not a sub-agent of {ctx.agent}; allowed: {allowed}")
        if len(task) > MAX_TASK_CHARS:
            raise ToolError(f"task exceeds {MAX_TASK_CHARS} characters")

        parent = await harness.get(ctx.run_id)
        depth = int(parent.metadata.get("depth", 0)) + 1
        if depth > MAX_DEPTH:
            raise ToolError(f"delegation depth limit ({MAX_DEPTH}) reached")

        run_id = child_run_id(ctx.run_id, ctx.call_id)
        child = await harness.state.runs.load(run_id)
        if child is None:
            child = await harness.create_run(
                agent,
                task,
                user_id=parent.user_id,
                owner=parent.metadata.get("owner"),
                parent_run_id=parent.run_id,
                run_id=run_id,
                metadata={"depth": depth},
            )
        if not child.status.terminal and child.status != RunStatus.AWAITING_APPROVAL:
            child = await harness.execute(run_id)

        usage = {
            "usage": child.usage.model_dump(),
            "cost_usd": child.cost_usd,
            "child_run_id": child.run_id,
        }
        if child.status == RunStatus.AWAITING_APPROVAL:
            tools = ", ".join(p.call.name for p in child.pending_approvals)
            return ToolOutcome(
                content=(
                    f"Sub-agent {agent} paused: its call to {tools} needs human approval "
                    f"(run {child.run_id}). Continue with other work or report this."
                ),
                is_error=True,
                metadata=usage,
            )
        if child.status != RunStatus.COMPLETED:
            reason = child.stop_reason.value if child.stop_reason else child.status.value
            partial = f" Partial answer: {child.answer}" if child.answer else ""
            return ToolOutcome(
                content=f"Sub-agent {agent} did not finish ({reason}).{partial}",
                is_error=True,
                metadata=usage,
            )
        return ToolOutcome(content=child.answer or "(empty answer)", metadata=usage)

    registry.register(
        make_tool(
            delegate,
            permission=Permission.WRITE,
            # Children enforce their own wall-clock limits; this is an outer bound.
            timeout_s=3600,
            idempotent=True,  # deterministic child ids make re-running safe
        )
    )
