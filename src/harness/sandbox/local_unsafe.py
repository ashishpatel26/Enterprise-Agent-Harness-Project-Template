"""UNSAFE local sandbox: a subprocess on the host. For development without Docker only.

It applies a timeout, output caps, a scratch working directory and (on POSIX) rlimits,
but it does NOT isolate the filesystem or the network. Model-written code can read your
files and reach the internet. It refuses to start unless explicitly opted in, and the
settings validator forbids it in prod.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxLimits,
    cap_output,
    validate_files,
)


def _posix_limits(limits: SandboxLimits) -> None:  # pragma: no cover - runs in the child
    if sys.platform == "win32":
        return
    import resource

    mem = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    cpu = int(limits.timeout_s) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    fsize = limits.tmp_size_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))


class UnsafeLocalSandbox(Sandbox):
    name = "local_unsafe"

    def __init__(self, *, allow: bool, default_limits: SandboxLimits | None = None) -> None:
        if not allow:
            raise SandboxError(
                "UnsafeLocalSandbox has no isolation; set HARNESS_ALLOW_UNSAFE_SANDBOX=true "
                "to use it in development"
            )
        self.default_limits = default_limits or SandboxLimits()

    async def run_python(
        self,
        code: str,
        *,
        files: Mapping[str, bytes] | None = None,
        limits: SandboxLimits | None = None,
    ) -> ExecResult:
        limits = limits or self.default_limits
        files = dict(files or {})
        validate_files(files, limits)
        workdir = Path(tempfile.mkdtemp(prefix="harness-sbx-"))
        data = workdir / "data"
        data.mkdir()
        try:
            for name, content in files.items():
                (data / name).write_bytes(content)
            # Same layout as the Docker sandbox: inputs under /data -> ./data here.
            preamble = "import os\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\n"
            script = workdir / "main.py"
            script.write_text(preamble + code.replace("/data/", "data/"), encoding="utf-8")
            env = {"PYTHONUNBUFFERED": "1", "PATH": os.environ.get("PATH", "")}
            started = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                str(script),
                cwd=workdir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=(lambda: _posix_limits(limits)) if os.name == "posix" else None,
            )
            timed_out = False
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=limits.timeout_s)
            except TimeoutError:
                timed_out = True
                proc.kill()
                out, err = await proc.communicate()
            stdout, t1 = cap_output(out, limits.max_output_bytes)
            stderr, t2 = cap_output(err, limits.max_output_bytes)
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=None if timed_out else proc.returncode,
                timed_out=timed_out,
                truncated=t1 or t2,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
