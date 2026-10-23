"""Sandbox escape tests against a real Docker daemon.

Each test runs hostile code and asserts the sandbox contained it. Requires the sandbox
image (`make sandbox-image`); auto-skipped when Docker is unavailable. CI runs them.
"""

from __future__ import annotations

import pytest

from harness.sandbox import SandboxLimits
from harness.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.docker


@pytest.fixture
def sbx() -> DockerSandbox:
    return DockerSandbox("agent-harness-sandbox:latest")


async def test_runs_normal_code_with_data(sbx: DockerSandbox) -> None:
    code = "import pandas as pd; df = pd.read_csv('/data/s.csv'); print(int(df.a.sum()))"
    result = await sbx.run_python(code, files={"s.csv": b"a\n1\n2\n"})
    assert result.ok, result.stderr
    assert result.stdout.strip() == "3"


async def test_network_is_blocked(sbx: DockerSandbox) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED', e)\n"
    )
    result = await sbx.run_python(code)
    assert "BLOCKED" in result.stdout
    assert "CONNECTED" not in result.stdout


async def test_dns_is_blocked(sbx: DockerSandbox) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.gethostbyname('example.com'); print('RESOLVED')\n"
        "except OSError:\n"
        "    print('NO_DNS')\n"
    )
    assert "NO_DNS" in (await sbx.run_python(code)).stdout


async def test_time_limit_is_enforced(sbx: DockerSandbox) -> None:
    result = await sbx.run_python("while True: pass", limits=SandboxLimits(timeout_s=2))
    assert result.timed_out
    assert result.duration_ms < 30_000


async def test_root_filesystem_is_read_only(sbx: DockerSandbox) -> None:
    code = (
        "for p in ['/etc/evil', '/usr/evil', '/evil', '/data/evil', '/data/main.py']:\n"
        "    try:\n"
        "        open(p, 'w').write('x'); print('WROTE', p)\n"
        "    except OSError:\n"
        "        print('DENIED', p)\n"
    )
    out = (await sbx.run_python(code)).stdout
    assert "WROTE" not in out
    assert out.count("DENIED") == 5


async def test_scratch_dir_is_writable_but_bounded(sbx: DockerSandbox) -> None:
    ok = await sbx.run_python("open('x.txt','w').write('hi'); print(open('x.txt').read())")
    assert ok.stdout.strip() == "hi"
    code = (
        "try:\n"
        "    open('big','wb').write(b'0' * (40 * 1024 * 1024)); print('FILLED')\n"
        "except OSError:\n"
        "    print('FULL')\n"
    )
    result = await sbx.run_python(code, limits=SandboxLimits(tmp_size_mb=8))
    assert "FULL" in result.stdout or not result.ok


async def test_runs_as_non_root_without_capabilities(sbx: DockerSandbox) -> None:
    code = (
        "import os\n"
        "print(os.getuid(), os.getgid())\n"
        "print(open('/proc/self/status').read().split('CapEff:')[1].split()[0])\n"
    )
    lines = (await sbx.run_python(code)).stdout.split()
    assert lines[:2] == ["10001", "10001"]
    assert int(lines[2], 16) == 0


async def test_memory_limit_is_enforced(sbx: DockerSandbox) -> None:
    result = await sbx.run_python(
        "x = bytearray(512 * 1024 * 1024); print('ALLOCATED')", limits=SandboxLimits(memory_mb=64)
    )
    assert "ALLOCATED" not in result.stdout
    assert not result.ok


async def test_fork_bomb_is_contained(sbx: DockerSandbox) -> None:
    code = (
        "import os\n"
        "n = 0\n"
        "try:\n"
        "    for _ in range(500):\n"
        "        if os.fork() == 0:\n"
        "            import time; time.sleep(5); os._exit(0)\n"
        "        n += 1\n"
        "except OSError:\n"
        "    pass\n"
        "print('FORKED', n)\n"
    )
    result = await sbx.run_python(code, limits=SandboxLimits(pids=16, timeout_s=10))
    forked = int(result.stdout.split()[-1]) if "FORKED" in result.stdout else 0
    assert forked < 16


async def test_output_is_capped(sbx: DockerSandbox) -> None:
    result = await sbx.run_python(
        "print('A' * 200_000)", limits=SandboxLimits(max_output_bytes=10_000)
    )
    assert result.truncated
    assert len(result.stdout) < 11_000


async def test_host_files_are_not_visible(sbx: DockerSandbox) -> None:
    code = "import os; print(sorted(os.listdir('/data')))"
    assert (await sbx.run_python(code, files={"only.csv": b"1"})).stdout.strip() == (
        "['main.py', 'only.csv']"
    )
