"""Test doubles shared across suites."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from harness.core.types import (
    GuardDecision,
    RunEvent,
    RunState,
    ToolContext,
    ToolOutcome,
)
from harness.llm.types import ToolCall, ToolSpec

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


class StubTools:
    """Minimal ToolExecutor: name -> async fn(arguments) -> str."""

    def __init__(self, tools: dict[str, ToolFn] | None = None) -> None:
        self.tools = tools or {}
        self.calls: list[ToolCall] = []

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        selected = self.tools if names is None else {n: self.tools[n] for n in names}
        return [ToolSpec(name=n, description=n, parameters={"type": "object"}) for n in selected]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        self.calls.append(call)
        fn = self.tools.get(call.name)
        if fn is None:
            return ToolOutcome(content=f"unknown tool {call.name}", is_error=True)
        try:
            return ToolOutcome(content=await fn(call.arguments))
        except Exception as exc:
            return ToolOutcome(content=str(exc), is_error=True)


class MemoryStore:
    def __init__(self) -> None:
        self.saved: dict[str, str] = {}
        self.saves = 0

    async def save(self, state: RunState) -> None:
        self.saves += 1
        self.saved[state.run_id] = state.model_dump_json()

    async def load(self, run_id: str) -> RunState | None:
        raw = self.saved.get(run_id)
        return RunState.model_validate_json(raw) if raw else None


class EventLog:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def __call__(self, event: RunEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]


class ScriptedGuard:
    def __init__(self, verdicts: dict[str, GuardDecision]) -> None:
        self.verdicts = verdicts

    async def check_call(self, call: ToolCall, ctx: ToolContext) -> GuardDecision:
        return self.verdicts.get(call.name, GuardDecision(verdict="allow"))

    async def screen_output(
        self, call: ToolCall, outcome: ToolOutcome, ctx: ToolContext
    ) -> ToolOutcome:
        return outcome


async def echo(args: dict[str, Any]) -> str:
    return f"echo:{args}"


async def slow(args: dict[str, Any]) -> str:
    await asyncio.sleep(10)
    return "late"
