"""Built-in example tools."""

from __future__ import annotations

from harness.config import Settings
from harness.sandbox.base import Sandbox
from harness.tools.builtin.files import file_read_tool
from harness.tools.builtin.python_exec import python_exec_tool
from harness.tools.builtin.sql_query import sql_query_tool
from harness.tools.builtin.web_fetch import web_fetch_tool
from harness.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry, settings: Settings, sandbox: Sandbox | None
) -> ToolRegistry:
    root = settings.workspace_root
    registry.register(file_read_tool(root))
    registry.register(sql_query_tool(root))
    registry.register(web_fetch_tool(settings.web_fetch_allowed_domains))
    if sandbox is not None:
        registry.register(python_exec_tool(sandbox, root))
    return registry


__all__ = [
    "file_read_tool",
    "python_exec_tool",
    "register_builtin_tools",
    "sql_query_tool",
    "web_fetch_tool",
]
