"""Shared fixtures. Tests needing Docker are auto-skipped when no daemon is reachable."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from harness.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@cache
def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env(timeout=3)
        client.ping()
        client.close()
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _docker_available():
        return
    skip = pytest.mark.skip(reason="Docker daemon not available")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test gets fresh settings and no leakage from a developer's .env."""
    monkeypatch.setenv("HARNESS_ENV", "test")
    monkeypatch.setenv("HARNESS_DOTENV", "")
    monkeypatch.chdir(REPO_ROOT)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
