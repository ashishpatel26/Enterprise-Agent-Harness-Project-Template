from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.config import Settings, get_settings


def test_defaults_are_safe_for_dev() -> None:
    s = get_settings()
    assert s.env == "test"
    assert s.sandbox_backend == "docker"
    assert s.allow_unsafe_sandbox is False


def test_api_keys_split_from_comma_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_API_KEYS", "a, b ,,c")
    keys = [k.get_secret_value() for k in get_settings().api_keys]
    assert keys == ["a", "b", "c"]


def test_secrets_are_not_rendered_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    s = get_settings()
    assert "supersecret" not in repr(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-supersecret"


def test_prod_rejects_insecure_configuration() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None, env="prod", storage_backend="memory", sandbox_backend="local_unsafe"
        )
    msg = str(exc.value)
    assert "HARNESS_API_KEYS" in msg
    assert "memory" in msg
    assert "local_unsafe" in msg


def test_prod_accepts_secure_configuration() -> None:
    s = Settings(_env_file=None, env="prod", api_keys="k1", storage_backend="redis")
    assert s.env == "prod"
