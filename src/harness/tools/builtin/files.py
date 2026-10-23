"""`file_read`: read text files inside the workspace, with traversal protection."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness.tools.registry import Permission, Tool, ToolError, make_tool

MAX_FILE_BYTES = 2_000_000


def resolve_in_workspace(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`; reject absolute paths, `..` escapes and symlinks
    that point outside the workspace."""
    if not relative or Path(relative).is_absolute() or relative.startswith(("/", "\\")):
        raise ToolError("path must be relative to the workspace root")
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ToolError("path escapes the workspace")
    return target


def file_read_tool(root: Path) -> Tool:
    async def file_read(path: str, offset: int = 0, limit: int = 200) -> str:
        """Read a UTF-8 text file from the workspace.

        `path` is relative to the workspace root (e.g. "data/sales.csv"). Returns up to
        `limit` lines starting at zero-based line `offset`, prefixed with line numbers.
        """
        if offset < 0 or not 1 <= limit <= 2000:
            raise ToolError("offset must be >= 0 and limit between 1 and 2000")
        target = resolve_in_workspace(root, path)

        def read() -> str:
            if not target.is_file():
                raise ToolError(f"file not found: {path}")
            if target.stat().st_size > MAX_FILE_BYTES:
                raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
            data = target.read_bytes()
            if b"\x00" in data[:8192]:
                raise ToolError("file looks binary; only text files can be read")
            lines = data.decode("utf-8", errors="replace").splitlines()
            chunk = lines[offset : offset + limit]
            body = "\n".join(f"{offset + i + 1:>6}  {line}" for i, line in enumerate(chunk))
            more = len(lines) - (offset + len(chunk))
            footer = f"\n[{more} more lines; use offset={offset + len(chunk)}]" if more > 0 else ""
            return f"{path} ({len(lines)} lines)\n{body}{footer}"

        return await asyncio.to_thread(read)

    return make_tool(file_read, permission=Permission.READ, timeout_s=10)
