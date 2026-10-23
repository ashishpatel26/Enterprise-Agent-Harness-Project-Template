from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from harness.core.types import ToolContext
from harness.llm import ToolCall
from harness.tools import Permission, ToolError, ToolRegistry, make_tool, schema_from_signature

CTX = ToolContext(run_id="r", session_id="s", user_id="u", agent="a", step=1)


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(id="c", name=name, arguments=dict(args))


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(permission=Permission.READ)
    async def add(a: int, b: int = 2) -> int:
        """Add two integers."""
        return a + b

    @reg.tool(permission=Permission.WRITE, timeout_s=0.05)
    async def sleepy(ctx: ToolContext) -> str:
        """Sleeps too long."""
        await asyncio.sleep(5)
        return ctx.run_id

    @reg.tool()
    async def fails(kind: Literal["expected", "crash"]) -> str:
        """Fails on purpose."""
        if kind == "expected":
            raise ToolError("bad input, try again")
        raise RuntimeError("secret internal detail")

    @reg.tool(max_output_chars=50)
    async def big() -> dict[str, str]:
        """Large output."""
        return {"data": "x" * 500}

    @reg.tool()
    async def who(ctx: ToolContext) -> str:
        """Returns the run id."""
        return ctx.run_id

    return reg


def test_schema_generated_from_signature() -> None:
    async def f(city: str, days: int = 3, unit: Literal["c", "f"] = "c") -> str:
        """doc"""
        return city

    schema = schema_from_signature(f)
    assert schema["required"] == ["city"]
    assert schema["properties"]["days"] == {"default": 3, "type": "integer"}
    assert schema["properties"]["unit"]["enum"] == ["c", "f"]
    assert schema["additionalProperties"] is False


def test_ctx_is_hidden_from_schema() -> None:
    spec = _registry().specs(["who"])[0]
    assert "ctx" not in spec.parameters.get("properties", {})


async def test_valid_call_executes() -> None:
    out = await _registry().execute(_call("add", a=1, b=5), CTX)
    assert out.content == "6"
    assert not out.is_error
    assert out.metadata["permission"] == "read"
    assert out.duration_ms >= 0


async def test_context_is_injected() -> None:
    assert (await _registry().execute(_call("who"), CTX)).content == "r"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({}, "'a' is a required property"),
        ({"a": "one"}, "a: 'one' is not of type 'integer'"),
        ({"a": 1, "zzz": 1}, "Additional properties are not allowed"),
        ({"__unparseable_arguments__": "{bad"}, "Additional properties"),
    ],
)
async def test_invalid_arguments_are_reported_not_executed(
    args: dict[str, object], fragment: str
) -> None:
    out = await _registry().execute(_call("add", **args), CTX)
    assert out.is_error
    assert out.content.startswith("Invalid arguments:")
    assert fragment in out.content
    assert out.metadata["validation_error"] is True


async def test_unknown_tool_lists_available() -> None:
    out = await _registry().execute(_call("nope"), CTX)
    assert out.is_error and "Available tools: add" in out.content


async def test_timeout_is_enforced() -> None:
    out = await _registry().execute(_call("sleepy"), CTX)
    assert out.is_error and "timed out after 0.05s" in out.content
    assert out.metadata["timeout"] is True


async def test_expected_error_message_reaches_model() -> None:
    out = await _registry().execute(_call("fails", kind="expected"), CTX)
    assert out.content == "Tool error: bad input, try again"


async def test_crash_does_not_leak_internals() -> None:
    out = await _registry().execute(_call("fails", kind="crash"), CTX)
    assert out.is_error
    assert "secret" not in out.content
    assert out.content == "Tool fails failed: RuntimeError"


async def test_output_is_capped() -> None:
    out = await _registry().execute(_call("big"), CTX)
    assert out.metadata["truncated"] is True
    assert "[output truncated:" in out.content


def test_registration_rules() -> None:
    reg = _registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(reg.get("add"))  # type: ignore[arg-type]

    def sync_fn() -> None:
        """sync"""

    with pytest.raises(TypeError, match="async"):
        make_tool(sync_fn)  # type: ignore[arg-type]

    async def nodoc() -> None:
        pass

    with pytest.raises(ValueError, match="description"):
        make_tool(nodoc)

    async def varargs(*a: int) -> None:
        """v"""

    with pytest.raises(TypeError, match=r"\*args"):
        make_tool(varargs)

    async def ok() -> str:
        """ok"""
        return ""

    with pytest.raises(ValueError, match="alphanumeric"):
        reg.register(make_tool(ok, name="bad-name"))


def test_specs_and_permissions() -> None:
    reg = _registry()
    assert reg.names() == ["add", "big", "fails", "sleepy", "who"]
    with pytest.raises(KeyError, match="unknown tools: missing"):
        reg.specs(["add", "missing"])
    sleepy = reg.get("sleepy")
    assert sleepy is not None
    assert sleepy.permission == Permission.WRITE
    assert sleepy.idempotent is False
    assert len(reg.specs()) == 5


def test_explicit_schema_is_checked() -> None:
    async def f(x: int) -> int:
        """f"""
        return x

    with pytest.raises(Exception, match="is not valid"):
        make_tool(f, parameters={"type": "not-a-type"})


def test_plugins_load_without_entry_points() -> None:
    assert ToolRegistry().load_plugins(group="harness.tools.none") == []
