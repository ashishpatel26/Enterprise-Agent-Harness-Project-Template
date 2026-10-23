"""Docker sandbox: one throwaway, locked-down container per execution.

Isolation applied to every container:
- no network (`network_mode=none`) unless explicitly enabled per call
- read-only root filesystem; inputs in a root-owned, read-only `/data` volume;
  the only writable path is a size-capped, noexec tmpfs at `/tmp` (the working dir)
- non-root user, all capabilities dropped, `no-new-privileges`, default seccomp profile
- memory (no swap), CPU, PID, open-file and file-size limits
- wall-clock timeout enforced by killing the container; output size capped
- optional hardened runtime (e.g. gVisor `runsc`) via `runtime=`

Inputs are copied in with `put_archive` instead of a host bind mount, so the harness
works the same when it runs inside a container that talks to the host Docker daemon.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxLimits,
    cap_output,
    validate_files,
)

if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container

log = logging.getLogger(__name__)

SANDBOX_UID = 10001
LABEL = "agent-harness.sandbox"
# Container-local, size-capped tmpfs; the only writable path inside the sandbox.
SCRATCH = "/tmp"  # noqa: S108


def _tar(files: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o444
            info.uid = info.gid = 0
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(
        self,
        image: str = "agent-harness-sandbox:latest",
        *,
        runtime: str | None = None,
        client: docker.DockerClient | None = None,
        default_limits: SandboxLimits | None = None,
    ) -> None:
        self.image = image
        self.runtime = runtime
        self.default_limits = default_limits or SandboxLimits()
        self._client = client

    def _docker(self) -> docker.DockerClient:
        if self._client is None:
            import docker

            try:
                self._client = docker.from_env(timeout=30)
                self._client.ping()
            except Exception as exc:
                self._client = None
                raise SandboxError(f"Docker daemon unavailable: {exc}") from exc
        return self._client

    def _create(self, limits: SandboxLimits) -> Container:
        import docker
        from docker.types import LogConfig, Mount, Ulimit

        kwargs: dict[str, Any] = {
            "image": self.image,
            "command": ["/data/main.py"],
            "user": f"{SANDBOX_UID}:{SANDBOX_UID}",
            "working_dir": SCRATCH,
            "environment": {"HOME": SCRATCH, "PYTHONUNBUFFERED": "1"},
            "network_mode": "bridge" if limits.network else "none",
            "read_only": True,
            "tmpfs": {SCRATCH: f"rw,noexec,nosuid,nodev,size={limits.tmp_size_mb}m"},
            "mounts": [Mount(target="/data", source=None, type="volume")],
            "mem_limit": f"{limits.memory_mb}m",
            "memswap_limit": f"{limits.memory_mb}m",
            "nano_cpus": int(limits.cpus * 1_000_000_000),
            "pids_limit": limits.pids,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "ulimits": [
                Ulimit(name="nofile", soft=256, hard=256),
                Ulimit(name="fsize", soft=64 * 1024 * 1024, hard=64 * 1024 * 1024),
            ],
            "log_config": LogConfig(type="json-file", config={"max-size": "2m"}),
            "labels": {LABEL: "1"},
            "detach": True,
        }
        if self.runtime:
            kwargs["runtime"] = self.runtime
        try:
            return self._docker().containers.create(**kwargs)
        except docker.errors.ImageNotFound as exc:
            raise SandboxError(
                f"sandbox image {self.image!r} not found; build it with `make sandbox-image`"
            ) from exc
        except docker.errors.DockerException as exc:
            raise SandboxError(f"failed to create sandbox container: {exc}") from exc

    def _run_sync(self, code: str, files: Mapping[str, bytes], limits: SandboxLimits) -> ExecResult:
        import requests.exceptions

        started = time.perf_counter()
        container = self._create(limits)
        try:
            payload = {**files, "main.py": code.encode("utf-8")}
            if not container.put_archive("/data", _tar(payload)):
                raise SandboxError("failed to copy inputs into the sandbox")
            container.start()
            timed_out = False
            try:
                status = container.wait(timeout=limits.timeout_s)
                exit_code: int | None = int(status.get("StatusCode", -1))
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                timed_out = True
                exit_code = None
                container.kill()
            container.reload()
            oom = bool(container.attrs.get("State", {}).get("OOMKilled", False))
            stdout, t1 = cap_output(
                container.logs(stdout=True, stderr=False), limits.max_output_bytes
            )
            stderr, t2 = cap_output(
                container.logs(stdout=False, stderr=True), limits.max_output_bytes
            )
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                oom_killed=oom,
                truncated=t1 or t2,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            try:
                container.remove(force=True, v=True)
            except Exception:
                log.warning("failed to remove sandbox container %s", container.id)

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
        if "main.py" in files:
            raise ValueError("'main.py' is reserved for the code being executed")
        return await asyncio.to_thread(self._run_sync, code, files, limits)

    async def aclose(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
