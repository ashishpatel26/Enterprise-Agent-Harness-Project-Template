"""Sandbox interface for running untrusted, model-written code.

Backends: `DockerSandbox` (default), `UnsafeLocalSandbox` (dev only). Others (gVisor via a
Docker runtime, Firecracker, E2B, ...) implement `Sandbox` and register through the
`harness.sandboxes` entry-point group.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class SandboxLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_s: float = Field(default=30.0, gt=0, le=600)
    memory_mb: int = Field(default=512, ge=32, le=16_384)
    cpus: float = Field(default=1.0, gt=0, le=16)
    pids: int = Field(default=64, ge=8, le=4096)
    tmp_size_mb: int = Field(default=64, ge=1, le=4096)
    max_output_bytes: int = Field(default=64_000, ge=1_000, le=10_000_000)
    max_input_bytes: int = Field(default=50_000_000, ge=1_000)
    network: bool = False


class ExecResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    oom_killed: bool = False
    truncated: bool = False
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.oom_killed


class SandboxError(Exception):
    """The sandbox itself failed (daemon down, image missing), not the user's code."""


class Sandbox(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def run_python(
        self,
        code: str,
        *,
        files: Mapping[str, bytes] | None = None,
        limits: SandboxLimits | None = None,
    ) -> ExecResult:
        """Run `code` with `files` available read-only under `/data/<name>`.

        Code can write only to a size-limited scratch directory (its working directory).
        """

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


def cap_output(data: bytes, limit: int) -> tuple[str, bool]:
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace"), False
    text = data[:limit].decode("utf-8", errors="replace")
    return f"{text}\n...[output truncated at {limit} bytes]", True


def validate_files(files: Mapping[str, bytes], limits: SandboxLimits) -> None:
    total = 0
    for name, data in files.items():
        if not name or "/" in name or "\\" in name or name.startswith(".") or "\x00" in name:
            raise ValueError(f"invalid sandbox file name {name!r}: use a plain file name")
        total += len(data)
    if total > limits.max_input_bytes:
        raise ValueError(f"input files exceed {limits.max_input_bytes} bytes")
