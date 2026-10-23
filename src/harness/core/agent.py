"""`Agent` = spec + model + harness components. Builds runs and executes them."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from harness.core.agent_spec import AgentSpec
from harness.core.loop import AgentLoop
from harness.core.types import (
    EventSink,
    RunState,
    StateStore,
    ToolExecutor,
    ToolGuard,
)
from harness.llm.base import LLMProvider
from harness.llm.types import Message
from harness.prompts.assembler import PromptAssembler

log = logging.getLogger(__name__)


class MemoryHook(Protocol):
    """Implemented by `harness.memory.MemoryManager`."""

    async def recall(self, *, query: str, user_id: str, k: int) -> list[str]: ...

    async def session_history(self, session_id: str) -> list[Message]: ...

    async def record_run(self, state: RunState) -> None: ...


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        llm: LLMProvider,
        tools: ToolExecutor,
        *,
        assembler: PromptAssembler | None = None,
        memory: MemoryHook | None = None,
        guard: ToolGuard | None = None,
        store: StateStore | None = None,
        sinks: Sequence[EventSink] = (),
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.tools = tools
        self.assembler = assembler or PromptAssembler()
        self.memory = memory
        self.guard = guard
        self.store = store
        self.sinks = list(sinks)

    @property
    def name(self) -> str:
        return self.spec.name

    def loop(self, extra_sinks: Sequence[EventSink] = ()) -> AgentLoop:
        return AgentLoop(
            self.llm,
            self.tools,
            limits=self.spec.limits,
            max_output_tokens=self.spec.model.max_tokens,
            guard=self.guard,
            store=self.store,
            sinks=[*self.sinks, *extra_sinks],
        )

    async def new_run(
        self,
        user_input: str,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> RunState:
        """Assemble the prompt and return a PENDING run. Nothing is executed yet."""
        recalled: list[str] = []
        history: list[Message] = []
        mem = self.spec.memory
        if self.memory is not None and mem.enabled:
            if mem.top_k:
                recalled = await self.memory.recall(query=user_input, user_id=user_id, k=mem.top_k)
            if session_id and mem.include_session_history:
                history = await self.memory.session_history(session_id)

        messages = self.assembler.build(
            self.spec.prompt.template,
            agent_name=self.spec.name,
            instructions=self.spec.prompt.instructions,
            user_input=user_input,
            memory=recalled,
            history=history,
        )
        state = RunState(
            agent=self.spec.name,
            user_id=user_id,
            input=user_input,
            messages=messages,
            tools=self.tools.specs(self.spec.tools),
            parent_run_id=parent_run_id,
        )
        if session_id:
            state.session_id = session_id
        if self.store is not None:
            await self.store.save(state)
        return state

    async def execute(self, state: RunState, extra_sinks: Sequence[EventSink] = ()) -> RunState:
        result = await self.loop(extra_sinks).run(state)
        if result.status.terminal and self.memory is not None:
            try:
                await self.memory.record_run(result)
            except Exception:
                # Memory is best-effort: a write failure must not fail a finished run.
                log.exception("failed to record run in memory", extra={"run_id": result.run_id})
        return result

    async def run(
        self, user_input: str, *, user_id: str = "anonymous", session_id: str | None = None
    ) -> RunState:
        """Convenience: build and execute in one call."""
        state = await self.new_run(user_input, user_id=user_id, session_id=session_id)
        return await self.execute(state)
