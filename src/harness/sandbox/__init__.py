"""Isolated execution of untrusted code."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import entry_points

from harness.config import Settings
from harness.sandbox.base import ExecResult, Sandbox, SandboxError, SandboxLimits

log = logging.getLogger(__name__)

SandboxFactory = Callable[[Settings], Sandbox]


def build_sandbox(settings: Settings) -> Sandbox:
    """Built-ins: `docker`, `local_unsafe`. Plugins: entry-point group `harness.sandboxes`."""
    backend = settings.sandbox_backend
    if backend == "docker":
        from harness.sandbox.docker_sandbox import DockerSandbox

        return DockerSandbox(settings.sandbox_image, runtime=settings.sandbox_runtime)
    if backend == "local_unsafe":
        from harness.sandbox.local_unsafe import UnsafeLocalSandbox

        log.warning("using UNSAFE local sandbox: model-written code runs on this host")
        return UnsafeLocalSandbox(allow=settings.allow_unsafe_sandbox)
    for ep in entry_points(group="harness.sandboxes"):
        if ep.name == backend:
            factory: SandboxFactory = ep.load()
            return factory(settings)
    raise SandboxError(f"unknown sandbox backend {backend!r}")


__all__ = ["ExecResult", "Sandbox", "SandboxError", "SandboxLimits", "build_sandbox"]
