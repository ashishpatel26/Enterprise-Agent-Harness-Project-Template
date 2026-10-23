"""MCP client: external MCP servers become governed harness tools (via FastMCP).

    # mcp_servers.yaml
    servers:
      github:
        transport: http
        url: https://api.githubcopilot.com/mcp/
        headers: {Authorization: "Bearer ${GITHUB_TOKEN}"}
        tools: [search_repositories, get_file_contents]   # optional allow-list
      files:
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "./examples"]
        permission: read                                  # override for all tools

Each remote tool is registered as `<server>__<tool>` and goes through the same pipeline
as a built-in tool: JSON-schema validation, timeout, output cap, policy, approvals,
injection screening, redaction, audit and metrics. Agents opt in per server with
`mcp: [github]` in their YAML.

Permissions come from the server's tool annotations (read-only → `read`, destructive →
`dangerous`, otherwise `write`) unless the config sets `permission`. Treat MCP servers
as untrusted code and data: unknown servers default to `write`, so policy decides.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.core.types import ToolOutcome
from harness.tools.registry import (
    DEFAULT_MAX_OUTPUT_CHARS,
    Permission,
    Tool,
    ToolError,
    ToolRegistry,
)

log = logging.getLogger(__name__)

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NAME = re.compile(r"[^A-Za-z0-9_]")


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio", "http"]
    # http
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # harness controls
    tools: list[str] | None = None
    permission: Literal["read", "write", "dangerous"] | None = None
    timeout_s: float = Field(default=30.0, gt=0, le=600)
    connect_timeout_s: float = Field(default=20.0, gt=0, le=120)
    max_output_chars: int = Field(default=DEFAULT_MAX_OUTPUT_CHARS, ge=1000)
    enabled: bool = True

    @model_validator(mode="after")
    def _check_transport(self) -> MCPServerConfig:
        if self.transport == "http" and not self.url:
            raise ValueError("http transport needs `url`")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio transport needs `command`")
        return self


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_names(self) -> MCPConfig:
        for name in self.servers:
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
                raise ValueError(f"invalid MCP server name {name!r} (lowercase, digits, _)")
        return self

    @classmethod
    def load(cls, path: Path) -> MCPConfig:
        if not path.is_file():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def expand_env(value: str) -> str:
    """Replace `${VAR}` with the environment value; a missing variable is an error so a
    secret never silently becomes an empty header."""

    def sub(m: re.Match[str]) -> str:
        if m.group(1) not in os.environ:
            raise ValueError(f"environment variable {m.group(1)} is not set")
        return os.environ[m.group(1)]

    return _ENV.sub(sub, value)


def tool_name(server: str, tool: str) -> str:
    return f"{server}__{_NAME.sub('_', tool)}"[:64]


def _permission(cfg: MCPServerConfig, annotations: Any) -> Permission:
    if cfg.permission:
        return Permission(cfg.permission)
    if annotations is not None:
        if getattr(annotations, "read_only_hint", None):
            return Permission.READ
        if getattr(annotations, "destructive_hint", None):
            return Permission.DANGEROUS
    return Permission.WRITE


def render_result(result: Any) -> tuple[str, bool]:
    """MCP CallToolResult -> (text, is_error). Non-text content is summarized."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource":
            res = getattr(block, "resource", None)
            parts.append(getattr(res, "text", None) or f"[resource {getattr(res, 'uri', '')}]")
        else:
            parts.append(f"[{kind or 'binary'} content omitted]")
    if not parts and getattr(result, "structured_content", None) is not None:
        import json

        parts.append(json.dumps(result.structured_content, default=str))
    return "\n".join(parts) or "(no content)", bool(getattr(result, "is_error", False))


def make_client(name: str, cfg: MCPServerConfig) -> Client[Any]:
    if cfg.transport == "http":
        assert cfg.url is not None
        transport: Any = StreamableHttpTransport(
            expand_env(cfg.url), headers={k: expand_env(v) for k, v in cfg.headers.items()}
        )
    else:
        assert cfg.command is not None
        transport = StdioTransport(
            cfg.command,
            [expand_env(a) for a in cfg.args],
            env={k: expand_env(v) for k, v in cfg.env.items()} or None,
            cwd=cfg.cwd,
        )
    return Client(
        transport,
        name=f"agent-harness:{name}",
        timeout=cfg.timeout_s,
        init_timeout=cfg.connect_timeout_s,
    )


class MCPHub:
    """Owns the MCP client sessions and the tools registered from them."""

    def __init__(self) -> None:
        self._stack = contextlib.AsyncExitStack()
        self.server_tools: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}

    async def connect(
        self,
        config: MCPConfig,
        registry: ToolRegistry,
        clients: dict[str, Client[Any]] | None = None,
    ) -> MCPHub:
        """Connect to every enabled server and register its tools. A server that fails
        to connect is recorded in `errors` and skipped: one bad server must not take
        the whole harness down."""
        for name, cfg in config.servers.items():
            if not cfg.enabled:
                continue
            try:
                client = (clients or {}).get(name) or make_client(name, cfg)
                await self._stack.enter_async_context(client)
                remote = await client.list_tools()
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                log.error(
                    "MCP server unavailable", extra={"server": name, "error": self.errors[name]}
                )
                continue
            self.server_tools[name] = self._register(name, cfg, client, remote, registry)
            log.info(
                "MCP server connected", extra={"server": name, "tools": self.server_tools[name]}
            )
        return self

    def _register(
        self,
        server: str,
        cfg: MCPServerConfig,
        client: Client[Any],
        remote: list[Any],
        registry: ToolRegistry,
    ) -> list[str]:
        registered: list[str] = []
        allowed = set(cfg.tools) if cfg.tools is not None else None
        for spec in remote:
            if allowed is not None and spec.name not in allowed:
                continue
            schema = dict(spec.input_schema or {"type": "object"})
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.SchemaError:
                log.warning("skipping MCP tool with invalid schema", extra={"tool": spec.name})
                continue
            name = tool_name(server, spec.name)
            registry.register(
                Tool(
                    name=name,
                    description=(spec.description or spec.name)[:1024] + f" (MCP server: {server})",
                    parameters=schema,
                    fn=self._caller(client, spec.name, cfg.timeout_s),
                    permission=_permission(cfg, spec.annotations),
                    timeout_s=cfg.timeout_s + 5,
                    max_output_chars=cfg.max_output_chars,
                    idempotent=bool(spec.annotations and spec.annotations.idempotent_hint)
                    or _permission(cfg, spec.annotations) == Permission.READ,
                    tags=frozenset({"mcp", f"mcp:{server}"}),
                )
            )
            registered.append(name)
        return registered

    @staticmethod
    def _caller(client: Client[Any], remote_name: str, timeout_s: float) -> Any:
        async def call(**arguments: Any) -> ToolOutcome:
            try:
                result = await client.call_tool(
                    remote_name, arguments, timeout=timeout_s, raise_on_error=False
                )
            except Exception as exc:
                raise ToolError(f"MCP call failed: {type(exc).__name__}: {exc}") from exc
            text, is_error = render_result(result)
            return ToolOutcome(content=text, is_error=is_error, metadata={"mcp_tool": remote_name})

        return call

    async def aclose(self) -> None:
        await self._stack.aclose()
