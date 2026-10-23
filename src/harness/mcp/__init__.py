"""Model Context Protocol: consume MCP servers as tools (`client`), and expose agents as
an MCP server (`server`, import `harness.mcp.server` explicitly)."""

from harness.mcp.client import MCPConfig, MCPHub, MCPServerConfig, tool_name

__all__ = ["MCPConfig", "MCPHub", "MCPServerConfig", "tool_name"]
