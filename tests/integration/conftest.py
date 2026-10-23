"""Fixtures for full-stack tests: a real Harness with FakeLLM models and no network."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from harness.api.app import create_app
from harness.config import Settings
from harness.llm import LLMProvider
from harness.runtime import Harness
from harness.sandbox.local_unsafe import UnsafeLocalSandbox

REPO = Path(__file__).resolve().parents[2]
API_KEY = "test-key-0123456789"

AGENTS = {
    "assistant.yaml": "name: assistant\ndescription: test agent\ntools: [file_read]\n"
    "model: {provider: fake, name: fake-1}\n",
    "operator.yaml": "name: operator\ntools: [python_exec]\n"
    "model: {provider: fake, name: fake-1}\n",
}


@pytest.fixture
def make_harness(tmp_path: Path) -> Callable[..., Harness]:
    agents = tmp_path / "agents"
    agents.mkdir()
    for name, body in AGENTS.items():
        (agents / name).write_text(body)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "hello.txt").write_text("hello from the workspace")

    def build(llms: dict[str, LLMProvider], **overrides: object) -> Harness:
        settings = Settings(
            _env_file=None,
            agents_dir=agents,
            policy_path=REPO / "policies" / "default.yaml",
            audit_log_path=tmp_path / "audit.log",
            workspace_root=tmp_path,
            **{"api_keys": API_KEY, **overrides},  # type: ignore[arg-type]
        )
        return Harness.build(settings, sandbox=UnsafeLocalSandbox(allow=True), llm_overrides=llms)

    return build


@asynccontextmanager
async def api_client(harness: Harness) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(harness))
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers={"X-API-Key": API_KEY}
        ) as client:
            yield client
    finally:
        await harness.aclose()
