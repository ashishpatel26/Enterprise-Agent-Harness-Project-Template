"""Sandbox unit tests that need no Docker daemon. Real isolation is proven by
tests/sandbox/test_escape.py, which runs against Docker in CI."""

from __future__ import annotations

import io
import sys
import tarfile
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from harness.config import Settings
from harness.sandbox import SandboxError, SandboxLimits, build_sandbox
from harness.sandbox.base import cap_output, validate_files
from harness.sandbox.docker_sandbox import DockerSandbox, _tar
from harness.sandbox.local_unsafe import UnsafeLocalSandbox


def test_cap_output() -> None:
    assert cap_output(b"abc", 10) == ("abc", False)
    text, truncated = cap_output(b"x" * 20, 10)
    assert truncated and text.startswith("x" * 10) and "truncated at 10 bytes" in text


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", "", "a\\b", "nul\x00"])
def test_validate_files_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid sandbox file name"):
        validate_files({name: b""}, SandboxLimits())


def test_validate_files_size_cap() -> None:
    with pytest.raises(ValueError, match="exceed"):
        validate_files({"a": b"x" * 2000}, SandboxLimits(max_input_bytes=1000))


def test_tar_entries_are_root_owned_and_read_only() -> None:
    with tarfile.open(fileobj=io.BytesIO(_tar({"a.csv": b"1,2"}))) as tar:
        info = tar.getmember("a.csv")
        assert (info.uid, info.gid, info.mode) == (0, 0, 0o444)


def _fake_client(wait: Any = None, oom: bool = False) -> tuple[MagicMock, MagicMock]:
    container = MagicMock()
    container.put_archive.return_value = True
    if isinstance(wait, Exception):
        container.wait.side_effect = wait
    else:
        container.wait.return_value = {"StatusCode": 0 if wait is None else wait}
    container.attrs = {"State": {"OOMKilled": oom}}
    container.logs.side_effect = lambda stdout, stderr: b"out" if stdout else b"err"
    client = MagicMock()
    client.containers.create.return_value = container
    return client, container


async def test_docker_sandbox_applies_hardening() -> None:
    client, container = _fake_client()
    sbx = DockerSandbox("img:1", runtime="runsc", client=client)
    limits = SandboxLimits(memory_mb=256, cpus=0.5, pids=32, tmp_size_mb=16)
    result = await sbx.run_python("print(1)", files={"d.csv": b"x"}, limits=limits)

    kw = client.containers.create.call_args.kwargs
    assert kw["network_mode"] == "none"
    assert kw["read_only"] is True
    assert kw["user"] == "10001:10001"
    assert kw["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kw["security_opt"]
    assert kw["mem_limit"] == kw["memswap_limit"] == "256m"
    assert kw["nano_cpus"] == 500_000_000
    assert kw["pids_limit"] == 32
    assert "size=16m" in kw["tmpfs"]["/tmp"] and "noexec" in kw["tmpfs"]["/tmp"]
    assert kw["runtime"] == "runsc"
    container.put_archive.assert_called_once()
    container.remove.assert_called_once_with(force=True, v=True)
    assert result.ok and result.stdout == "out" and result.stderr == "err"


async def test_docker_sandbox_network_opt_in() -> None:
    client, _ = _fake_client()
    await DockerSandbox(client=client).run_python("x", limits=SandboxLimits(network=True))
    assert client.containers.create.call_args.kwargs["network_mode"] == "bridge"


async def test_docker_sandbox_timeout_kills_container() -> None:
    client, container = _fake_client(wait=requests.exceptions.ReadTimeout())
    result = await DockerSandbox(client=client).run_python("while True: pass")
    assert result.timed_out and result.exit_code is None
    container.kill.assert_called_once()
    container.remove.assert_called_once()


async def test_docker_sandbox_reports_oom_and_cleans_up_on_failure() -> None:
    client, _ = _fake_client(wait=137, oom=True)
    result = await DockerSandbox(client=client).run_python("x")
    assert result.oom_killed and not result.ok

    client, container = _fake_client()
    container.put_archive.return_value = False
    with pytest.raises(SandboxError, match="copy inputs"):
        await DockerSandbox(client=client).run_python("x")
    container.remove.assert_called_once()


async def test_docker_sandbox_rejects_reserved_name() -> None:
    client, _ = _fake_client()
    with pytest.raises(ValueError, match="reserved"):
        await DockerSandbox(client=client).run_python("x", files={"main.py": b""})


async def test_local_sandbox_requires_opt_in() -> None:
    with pytest.raises(SandboxError, match="no isolation"):
        UnsafeLocalSandbox(allow=False)


async def test_local_sandbox_runs_code_with_files() -> None:
    sbx = UnsafeLocalSandbox(allow=True)
    code = "print(open('/data/in.txt').read().upper())"
    result = await sbx.run_python(code, files={"in.txt": b"hello"})
    assert result.ok, result.stderr
    assert result.stdout.strip() == "HELLO"


async def test_local_sandbox_timeout_and_errors() -> None:
    sbx = UnsafeLocalSandbox(allow=True)
    result = await sbx.run_python("import time; time.sleep(5)", limits=SandboxLimits(timeout_s=0.5))
    assert result.timed_out
    result = await sbx.run_python("raise SystemExit(3)")
    assert result.exit_code == 3
    result = await sbx.run_python("print('y' * 5000)", limits=SandboxLimits(max_output_bytes=1000))
    assert result.truncated


def test_build_sandbox_selection() -> None:
    s = Settings(_env_file=None, sandbox_backend="docker")
    assert build_sandbox(s).name == "docker"
    s = Settings(_env_file=None, sandbox_backend="local_unsafe", allow_unsafe_sandbox=True)
    assert build_sandbox(s).name == "local_unsafe"
    with pytest.raises(SandboxError, match="no isolation"):
        build_sandbox(Settings(_env_file=None, sandbox_backend="local_unsafe"))
    with pytest.raises(SandboxError, match="unknown sandbox backend"):
        build_sandbox(Settings(_env_file=None, sandbox_backend="firecracker"))


def test_python_is_available() -> None:
    assert sys.executable
