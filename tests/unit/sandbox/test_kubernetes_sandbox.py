"""Kubernetes sandbox against a mocked CoreV1Api (no cluster needed)."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from harness.config import Settings
from harness.sandbox import SandboxError, SandboxLimits, build_sandbox
from harness.sandbox.kubernetes_sandbox import KubernetesSandbox, build_manifests


def _status(phase: str, exit_code: int | None = None, reason: str | None = None) -> Any:
    term = SimpleNamespace(exit_code=exit_code, reason=reason) if exit_code is not None else None
    cs = SimpleNamespace(state=SimpleNamespace(terminated=term))
    return SimpleNamespace(
        status=SimpleNamespace(phase=phase, container_statuses=[cs], reason=None)
    )


def _api(*phases: Any, log: str = "hello\n") -> MagicMock:
    api = MagicMock()
    api.read_namespaced_pod_status.side_effect = list(phases)
    api.read_namespaced_pod_log.return_value = log
    return api


def test_manifests_are_locked_down() -> None:
    cm, pod = build_manifests(
        "sbx",
        "ns",
        "img:1",
        "print(1)",
        {"d.csv": b"a,b"},
        SandboxLimits(memory_mb=256, cpus=0.5, tmp_size_mb=16, timeout_s=10),
        runtime_class="gvisor",
    )
    assert base64.b64decode(cm["binaryData"]["main.py"]) == b"print(1)"
    assert cm["immutable"] is True
    spec = pod["spec"]
    assert spec["automountServiceAccountToken"] is False
    assert spec["enableServiceLinks"] is False
    assert spec["runtimeClassName"] == "gvisor"
    assert spec["activeDeadlineSeconds"] == 15
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    c = spec["containers"][0]
    assert c["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "privileged": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    assert c["resources"]["limits"]["memory"] == "256Mi"
    assert {"name": "data", "mountPath": "/data", "readOnly": True} in c["volumeMounts"]
    scratch = next(v for v in spec["volumes"] if v["name"] == "scratch")
    assert scratch["emptyDir"] == {"medium": "Memory", "sizeLimit": "16Mi"}
    assert pod["metadata"]["labels"]["agent-harness.sandbox/network"] == "denied"


async def test_successful_run_cleans_up() -> None:
    api = _api(_status("Pending"), _status("Running"), _status("Succeeded", 0))
    sbx = KubernetesSandbox("img", namespace="ns", core_v1=api, poll_interval_s=0)
    result = await sbx.run_python("print('hello')")
    assert result.ok and result.stdout == "hello\n"
    api.create_namespaced_config_map.assert_called_once()
    api.create_namespaced_pod.assert_called_once()
    api.delete_namespaced_pod.assert_called_once()
    api.delete_namespaced_config_map.assert_called_once()


async def test_failures_timeouts_and_oom() -> None:
    api = _api(_status("Failed", 1))
    result = await KubernetesSandbox("i", core_v1=api, poll_interval_s=0).run_python("x")
    assert result.exit_code == 1 and not result.ok

    api = _api(_status("Failed", 137, "OOMKilled"))
    result = await KubernetesSandbox("i", core_v1=api, poll_interval_s=0).run_python("x")
    assert result.oom_killed

    api = _api(*[_status("Running")] * 1000)
    sbx = KubernetesSandbox("i", core_v1=api, poll_interval_s=0.01)
    result = await sbx.run_python("while True: pass", limits=SandboxLimits(timeout_s=0.05))
    assert result.timed_out and result.exit_code is None
    api.delete_namespaced_pod.assert_called_once()


async def test_api_errors_become_sandbox_errors_and_still_clean_up() -> None:
    api = MagicMock()
    api.create_namespaced_pod.side_effect = RuntimeError("forbidden")
    with pytest.raises(SandboxError, match="forbidden"):
        await KubernetesSandbox("i", core_v1=api).run_python("x")
    api.delete_namespaced_config_map.assert_called_once()


async def test_input_size_cap() -> None:
    sbx = KubernetesSandbox("i", core_v1=MagicMock())
    with pytest.raises(ValueError, match="Kubernetes backend"):
        await sbx.run_python("x", files={"big.csv": b"0" * 1_000_000})
    with pytest.raises(ValueError, match="reserved"):
        await sbx.run_python("x", files={"main.py": b""})


def test_build_sandbox_kubernetes() -> None:
    s = Settings(_env_file=None, sandbox_backend="kubernetes", sandbox_runtime="gvisor")
    sbx = build_sandbox(s)
    assert isinstance(sbx, KubernetesSandbox)
    assert sbx.runtime_class == "gvisor" and sbx.namespace == "agent-harness-sandbox"
