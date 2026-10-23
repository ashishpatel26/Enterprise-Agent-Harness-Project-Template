"""`python_exec`: run model-written Python in the sandbox."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness.sandbox.base import Sandbox, SandboxError, SandboxLimits
from harness.tools.builtin.files import MAX_FILE_BYTES, resolve_in_workspace
from harness.tools.registry import Permission, Tool, ToolError, make_tool

MAX_FILES = 10


def _format(stdout: str, stderr: str, exit_code: int | None) -> str:
    parts = [f"exit_code: {exit_code}"]
    parts.append(f"--- stdout ---\n{stdout.rstrip() or '(empty)'}")
    if stderr.strip():
        parts.append(f"--- stderr ---\n{stderr.rstrip()}")
    return "\n".join(parts)


def python_exec_tool(sandbox: Sandbox, root: Path, limits: SandboxLimits | None = None) -> Tool:
    limits = limits or SandboxLimits()

    async def python_exec(code: str, files: list[str] | None = None) -> str:
        """Run Python 3.12 code in an isolated sandbox and return its stdout/stderr.

        pandas and numpy are installed. There is no network access. Workspace files listed
        in `files` (paths relative to the workspace root) are copied read-only to
        `/data/<file name>`, e.g. files=["data/sales.csv"] -> "/data/sales.csv". The
        working directory is a small writable scratch space. Print the results you need.
        """
        paths = files or []
        if len(paths) > MAX_FILES:
            raise ToolError(f"at most {MAX_FILES} files can be attached")

        def load() -> dict[str, bytes]:
            out: dict[str, bytes] = {}
            for rel in paths:
                target = resolve_in_workspace(root, rel)
                if not target.is_file():
                    raise ToolError(f"file not found: {rel}")
                if target.stat().st_size > MAX_FILE_BYTES * 10:
                    raise ToolError(f"file too large: {rel}")
                if target.name in out:
                    raise ToolError(f"two attached files share the name {target.name!r}")
                out[target.name] = target.read_bytes()
            return out

        inputs = await asyncio.to_thread(load)
        try:
            result = await sandbox.run_python(code, files=inputs, limits=limits)
        except SandboxError as exc:
            raise ToolError(f"sandbox unavailable: {exc}") from exc
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if result.timed_out:
            raise ToolError(
                f"execution timed out after {limits.timeout_s}s\n"
                + _format(result.stdout, result.stderr, None)
            )
        if result.oom_killed:
            raise ToolError(f"execution exceeded the {limits.memory_mb} MB memory limit")
        text = _format(result.stdout, result.stderr, result.exit_code)
        if result.exit_code != 0:
            raise ToolError(text)
        return text

    return make_tool(
        python_exec,
        permission=Permission.DANGEROUS,
        # Container start/teardown on top of the code's own time limit.
        timeout_s=limits.timeout_s + 30,
        idempotent=True,
    )
