"""Tool registry: decorator registration, JSON-schema argument validation, per-tool
timeouts, output caps and permission levels.

    registry = ToolRegistry()

    @registry.tool(permission=Permission.READ, timeout_s=10)
    async def lookup(city: str, ctx: ToolContext) -> str:
        "Look up the weather for a city."
        ...

The JSON schema the model sees is generated from the signature (via Pydantic), or given
explicitly with `parameters=`. Arguments are validated against that schema before the
function runs; validation errors go back to the model as a correctable tool error.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import entry_points
from typing import Any, get_type_hints

import jsonschema
from pydantic import create_model

from harness.core.types import ToolContext, ToolOutcome
from harness.llm.types import ToolCall, ToolSpec

log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Any]]
DEFAULT_MAX_OUTPUT_CHARS = 20_000


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


class ToolError(Exception):
    """Raise inside a tool for an expected failure; the message goes to the model."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    permission: Permission = Permission.READ
    timeout_s: float = 30.0
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    # True when re-running with the same arguments has no additional side effects.
    idempotent: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @property
    def wants_ctx(self) -> bool:
        return "ctx" in inspect.signature(self.fn).parameters


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON schema from type hints and defaults (the `ctx` parameter is skipped)."""
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "ctx":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {fn.__name__}: *args/**kwargs are not supported")
        annotation = hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    model = create_model(f"{fn.__name__}_args", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema


def make_tool(
    fn: ToolFn,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    permission: Permission = Permission.READ,
    timeout_s: float = 30.0,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    idempotent: bool | None = None,
    tags: frozenset[str] = frozenset(),
) -> Tool:
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"tool {fn.__name__} must be an async function")
    desc = description or inspect.getdoc(fn)
    if not desc:
        raise ValueError(f"tool {fn.__name__} needs a description (docstring or description=)")
    schema = parameters or schema_from_signature(fn)
    jsonschema.Draft202012Validator.check_schema(schema)
    return Tool(
        name=name or fn.__name__,
        description=desc,
        parameters=schema,
        fn=fn,
        permission=permission,
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        idempotent=permission == Permission.READ if idempotent is None else idempotent,
        tags=tags,
    )


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _cap(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[output truncated: {omitted} more characters]", True


class ToolRegistry:
    """Implements the `ToolExecutor` contract used by the agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, jsonschema.Draft202012Validator] = {}

    # ---- registration -----------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        if not tool.name.replace("_", "").isalnum():
            raise ValueError(f"tool name {tool.name!r} must be alphanumeric/underscore")
        self._tools[tool.name] = tool
        self._validators[tool.name] = jsonschema.Draft202012Validator(tool.parameters)
        return tool

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        permission: Permission = Permission.READ,
        timeout_s: float = 30.0,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        idempotent: bool | None = None,
    ) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self.register(
                make_tool(
                    fn,
                    name=name,
                    description=description,
                    parameters=parameters,
                    permission=permission,
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                    idempotent=idempotent,
                )
            )
            return fn

        return decorator

    def load_plugins(self, group: str = "harness.tools") -> list[str]:
        """Register tools exposed by installed packages. An entry point may resolve to a
        `Tool` or to a callable taking the registry (for tools that need setup)."""
        loaded: list[str] = []
        for ep in entry_points(group=group):
            try:
                obj = ep.load()
                if isinstance(obj, Tool):
                    self.register(obj)
                else:
                    obj(self)
                loaded.append(ep.name)
            except Exception:
                log.exception("failed to load tool plugin %r", ep.name)
        return loaded

    # ---- lookup -----------------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        if names is None:
            return [self._tools[n].spec for n in self.names()]
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"unknown tools: {', '.join(missing)}; registered: {self.names()}")
        return [self._tools[n].spec for n in names]

    def validation_errors(self, name: str, arguments: dict[str, Any]) -> list[str]:
        validator = self._validators[name]
        return [
            f"{'/'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
            for e in sorted(validator.iter_errors(arguments), key=lambda e: list(e.path))
        ]

    # ---- execution --------------------------------------------------------
    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolOutcome(
                content=f"Unknown tool {call.name!r}. Available tools: {', '.join(self.names())}",
                is_error=True,
            )
        meta: dict[str, Any] = {"permission": tool.permission.value}
        if errors := self.validation_errors(tool.name, call.arguments):
            return ToolOutcome(
                content="Invalid arguments:\n" + "\n".join(f"- {e}" for e in errors),
                is_error=True,
                metadata={**meta, "validation_error": True},
            )

        kwargs = dict(call.arguments)
        if tool.wants_ctx:
            kwargs["ctx"] = ctx
        started = time.perf_counter()
        try:
            async with asyncio.timeout(tool.timeout_s):
                result = await tool.fn(**kwargs)
            content, is_error = _render(result), False
        except TimeoutError:
            content, is_error = f"Tool {tool.name} timed out after {tool.timeout_s}s", True
            meta["timeout"] = True
        except ToolError as exc:
            content, is_error = f"Tool error: {exc}", True
        except Exception as exc:
            # Unexpected failures: log the traceback server-side, give the model a short
            # message without internals.
            log.exception("tool crashed", extra={"tool": tool.name, "run_id": ctx.run_id})
            content, is_error = f"Tool {tool.name} failed: {type(exc).__name__}", True
        duration_ms = (time.perf_counter() - started) * 1000
        content, truncated = _cap(content, tool.max_output_chars)
        if truncated:
            meta["truncated"] = True
        return ToolOutcome(
            content=content, is_error=is_error, duration_ms=duration_ms, metadata=meta
        )
