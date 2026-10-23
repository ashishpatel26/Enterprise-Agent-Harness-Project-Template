"""Kubernetes sandbox: one locked-down Pod per execution.

Use this backend when the harness runs on Kubernetes. It needs no Docker socket; the
API's ServiceAccount only needs to create/read/delete Pods and ConfigMaps in a dedicated
sandbox namespace (the Helm chart sets this up).

Isolation per Pod:
- non-root (uid 10001), read-only root filesystem, all capabilities dropped,
  no privilege escalation, RuntimeDefault seccomp, no service-account token, no service
  links; optional hardened RuntimeClass (e.g. gVisor)
- inputs mounted read-only at /data from a ConfigMap; a size-capped in-memory emptyDir
  at /tmp is the only writable path
- CPU/memory/ephemeral-storage limits, `activeDeadlineSeconds`, wall-clock timeout
- network: the chart's NetworkPolicy denies all ingress and egress for sandbox Pods

Limits of this backend (documented in docs/deployment.md): inputs are capped at ~900 KB
(ConfigMap size), stdout and stderr arrive merged (Kubernetes logs), and the PID limit
is the kubelet's `podPidsLimit`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxError,
    SandboxLimits,
    cap_output,
    validate_files,
)

log = logging.getLogger(__name__)

MAX_CONFIGMAP_BYTES = 900_000
LABEL = "agent-harness.sandbox"
API_TIMEOUT_S = 10


def build_manifests(
    name: str,
    namespace: str,
    image: str,
    code: str,
    files: Mapping[str, bytes],
    limits: SandboxLimits,
    runtime_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {**files, "main.py": code.encode("utf-8")}
    labels = {
        "app.kubernetes.io/name": "agent-harness-sandbox",
        "app.kubernetes.io/managed-by": "agent-harness",
        LABEL: "1",
        f"{LABEL}/network": "allowed" if limits.network else "denied",
    }
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "binaryData": {k: base64.b64encode(v).decode() for k, v in payload.items()},
        "immutable": True,
    }
    pod: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": math.ceil(limits.timeout_s) + 5,
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 10001,
                "runAsGroup": 10001,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "sandbox",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "args": ["/data/main.py"],
                    "workingDir": "/tmp",  # noqa: S108 - pod-local emptyDir
                    "env": [
                        {"name": "HOME", "value": "/tmp"},  # noqa: S108
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                    ],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": f"{min(128, limits.memory_mb)}Mi"},
                        "limits": {
                            "cpu": str(limits.cpus),
                            "memory": f"{limits.memory_mb}Mi",
                            "ephemeral-storage": f"{limits.tmp_size_mb}Mi",
                        },
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "privileged": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [
                        {"name": "data", "mountPath": "/data", "readOnly": True},
                        {"name": "scratch", "mountPath": "/tmp"},  # noqa: S108
                    ],
                }
            ],
            "volumes": [
                {"name": "data", "configMap": {"name": name, "defaultMode": 0o444}},
                {
                    "name": "scratch",
                    "emptyDir": {"medium": "Memory", "sizeLimit": f"{limits.tmp_size_mb}Mi"},
                },
            ],
        },
    }
    if runtime_class:
        pod["spec"]["runtimeClassName"] = runtime_class
    return configmap, pod


class KubernetesSandbox(Sandbox):
    name = "kubernetes"

    def __init__(
        self,
        image: str,
        *,
        namespace: str = "agent-harness-sandbox",
        runtime_class: str | None = None,
        core_v1: Any = None,
        poll_interval_s: float = 0.5,
        default_limits: SandboxLimits | None = None,
    ) -> None:
        self.image = image
        self.namespace = namespace
        self.runtime_class = runtime_class
        self.poll_interval_s = poll_interval_s
        self.default_limits = default_limits or SandboxLimits()
        self._api = core_v1

    def _core(self) -> Any:
        if self._api is None:
            try:
                from kubernetes import client, config
            except ImportError as exc:
                raise SandboxError(
                    "the kubernetes sandbox needs the 'kubernetes' extra: "
                    "uv sync --extra kubernetes"
                ) from exc
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self._api = client.CoreV1Api()
        return self._api

    def _run_sync(self, code: str, files: Mapping[str, bytes], limits: SandboxLimits) -> ExecResult:
        api = self._core()
        name = f"harness-sbx-{uuid.uuid4().hex[:12]}"
        configmap, pod = build_manifests(
            name, self.namespace, self.image, code, files, limits, self.runtime_class
        )
        started = time.perf_counter()
        try:
            api.create_namespaced_config_map(
                self.namespace, configmap, _request_timeout=API_TIMEOUT_S
            )
            api.create_namespaced_pod(self.namespace, pod, _request_timeout=API_TIMEOUT_S)
            deadline = time.monotonic() + limits.timeout_s + 60  # + scheduling/image pull
            run_deadline: float | None = None
            timed_out = False
            status: Any = None
            while True:
                status = api.read_namespaced_pod_status(
                    name, self.namespace, _request_timeout=API_TIMEOUT_S
                ).status
                phase = status.phase
                if phase in ("Succeeded", "Failed"):
                    break
                now = time.monotonic()
                if phase == "Running" and run_deadline is None:
                    run_deadline = now + limits.timeout_s
                if now > deadline or (run_deadline is not None and now > run_deadline):
                    timed_out = True
                    break
                time.sleep(self.poll_interval_s)

            exit_code: int | None = None
            oom = False
            for cs in (status.container_statuses or []) if status else []:
                term = cs.state.terminated if cs.state else None
                if term is not None:
                    exit_code = term.exit_code
                    oom = term.reason == "OOMKilled"
            if status is not None and status.reason == "DeadlineExceeded":
                timed_out = True
            try:
                raw = api.read_namespaced_pod_log(
                    name,
                    self.namespace,
                    limit_bytes=limits.max_output_bytes + 1,
                    _request_timeout=API_TIMEOUT_S,
                )
            except Exception:
                raw = ""
            stdout, truncated = cap_output(
                (raw or "").encode("utf-8", errors="replace"), limits.max_output_bytes
            )
            return ExecResult(
                stdout=stdout,
                stderr="",  # Kubernetes merges stderr into the pod log
                exit_code=None if timed_out else exit_code,
                timed_out=timed_out,
                oom_killed=oom,
                truncated=truncated,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"kubernetes sandbox failed: {type(exc).__name__}: {exc}") from exc
        finally:
            for delete in (api.delete_namespaced_pod, api.delete_namespaced_config_map):
                try:
                    delete(
                        name, self.namespace, grace_period_seconds=0, _request_timeout=API_TIMEOUT_S
                    )
                except Exception:
                    log.warning("failed to delete sandbox resource %s", name)

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
        if sum(len(v) for v in files.values()) + len(code) > MAX_CONFIGMAP_BYTES:
            raise ValueError(f"inputs exceed {MAX_CONFIGMAP_BYTES} bytes (Kubernetes backend)")
        return await asyncio.to_thread(self._run_sync, code, files, limits)
