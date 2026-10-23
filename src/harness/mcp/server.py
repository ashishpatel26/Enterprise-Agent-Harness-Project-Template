"""MCP server: expose harness agents to any MCP client (Claude Desktop/Code, IDEs, other
agents) via FastMCP.

Two transports:
- HTTP (streamable): mounted by the API at `/mcp`, authenticated with the same API keys
  (`X-API-Key` or `Authorization: Bearer <key>`), tenant-scoped like the REST API.
- stdio: `harness mcp-serve` for local clients; runs as tenant `mcp-local`.

Runs started over MCP get the full harness: policy, approvals, sandbox, memory, audit.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from starlette.types import ASGIApp, Receive, Scope, Send

from harness.core.types import RunState
from harness.runtime import Harness, RunNotFoundError

LOCAL_OWNER = "mcp-local"

INSTRUCTIONS = (
    "agent-harness runs governed AI agents. Call list_agents to see what is available, "
    "run_agent to start a run (it returns the answer, or pending approvals if a tool call "
    "needs human sign-off), approve_tool_call to decide, and get_run to check status."
)


def _key_id(key: str) -> str:
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _caller_key(headers: dict[str, str]) -> str | None:
    key = headers.get("x-api-key")
    auth = headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    return key or None


def _view(s: RunState) -> dict[str, Any]:
    return {
        "run_id": s.run_id,
        "agent": s.agent,
        "status": s.status.value,
        "stop_reason": s.stop_reason.value if s.stop_reason else None,
        "answer": s.answer,
        "error": s.error,
        "steps": s.step,
        "cost_usd": round(s.cost_usd, 6),
        "pending_approvals": [
            {
                "call_id": p.call.id,
                "tool": p.call.name,
                "arguments": p.call.arguments,
                "reason": p.reason,
            }
            for p in s.pending_approvals
        ],
    }


def build_mcp_server(get_harness: Callable[[], Harness]) -> FastMCP:
    """`get_harness` is resolved per call, so the server can be built before startup."""
    mcp = FastMCP("agent-harness", instructions=INSTRUCTIONS)

    def owner() -> tuple[str, bool]:
        """(tenant, can_approve) for the current caller."""
        key = _caller_key(get_http_headers(include={"authorization", "x-api-key"}))
        if key is None:
            return LOCAL_OWNER, True  # stdio: the local user owns the process
        approvers = [k.get_secret_value() for k in get_harness().settings.approver_keys]
        allowed = not approvers or any(
            hmac.compare_digest(key.encode(), a.encode()) for a in approvers
        )
        return _key_id(key), allowed

    async def owned(run_id: str) -> RunState:
        h = get_harness()
        try:
            state = await h.get(run_id)
        except RunNotFoundError:
            raise ValueError("run not found") from None
        if state.metadata.get("owner") != owner()[0]:
            raise ValueError("run not found")
        return state

    @mcp.tool
    async def list_agents() -> list[dict[str, Any]]:
        """List the agents this harness can run, with their tools and descriptions."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "tools": s.tools,
                "sub_agents": s.sub_agents,
            }
            for s in sorted(get_harness().specs.values(), key=lambda s: s.name)
        ]

    @mcp.tool
    async def run_agent(agent: str, input: str, session_id: str | None = None) -> dict[str, Any]:
        """Run an agent on a task and return its answer. If a tool call needs human
        approval, the run pauses and the pending calls are returned."""
        h = get_harness()
        tenant, _ = owner()
        state = await h.create_run(
            agent, input, user_id=tenant, session_id=session_id, owner=tenant
        )
        return _view(await h.execute(state.run_id))

    @mcp.tool
    async def get_run(run_id: str) -> dict[str, Any]:
        """Get the status, answer and pending approvals of a run."""
        return _view(await owned(run_id))

    @mcp.tool
    async def approve_tool_call(
        run_id: str, call_id: str, approved: bool, comment: str = ""
    ) -> dict[str, Any]:
        """Approve or deny a paused tool call. The run resumes in the background."""
        tenant, can_approve = owner()
        if not can_approve:
            raise ValueError("this API key cannot approve tool calls")
        await owned(run_id)
        state = await get_harness().decide(
            run_id, call_id, approved=approved, reviewer=tenant, comment=comment
        )
        return _view(state)

    return mcp


class APIKeyGuard:
    """ASGI wrapper for the HTTP MCP endpoint: rejects requests without a valid key."""

    def __init__(self, app: ASGIApp, keys: Callable[[], list[str]]) -> None:
        self.app = app
        self.keys = keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            keys = self.keys()
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            key = _caller_key(headers)
            ok = not keys or (
                key is not None and any(hmac.compare_digest(key.encode(), k.encode()) for k in keys)
            )
            if not ok:
                await _reject(send)
                return
        await self.app(scope, receive, send)


async def _reject(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    body = b'{"error":"unauthorized","detail":"missing or invalid API key"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"ApiKey")],
        }
    )
    await send({"type": "http.response.body", "body": body})
