from __future__ import annotations

import pytest

from harness.config import get_settings
from harness.llm import FakeLLM, Usage
from harness.llm.factory import (
    ModelConfig,
    ModelRef,
    available_providers,
    build_llm,
    build_provider,
    register_provider,
)
from harness.llm.fallback import FallbackProvider
from harness.llm.pricing import cost_usd, register_price
from harness.llm.retry import RetryingProvider


def test_cost_for_known_model() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(24.0)


def test_cache_tokens_are_discounted_and_surcharged() -> None:
    usage = Usage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 * 0.1 + 4.0 * 1.25)


def test_unknown_model_costs_zero() -> None:
    assert cost_usd("mystery-model", Usage(input_tokens=10)) == 0.0


def test_register_price() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        register_price("x", -1, 1)
    register_price("custom-model", 1.0, 2.0)
    assert cost_usd("custom-model", Usage(input_tokens=1_000_000)) == pytest.approx(1.0)


def test_build_llm_wraps_with_retries() -> None:
    llm = build_llm(ModelConfig(provider="fake", name="fake-1"), get_settings())
    assert isinstance(llm, RetryingProvider)


def test_build_llm_with_fallbacks() -> None:
    cfg = ModelConfig(provider="fake", name="a", fallbacks=[ModelRef(provider="fake", name="b")])
    llm = build_llm(cfg, get_settings())
    assert isinstance(llm, FallbackProvider)
    assert [p.model for p in llm.providers] == ["a", "b"]


def test_unknown_provider_lists_available() -> None:
    with pytest.raises(ValueError, match=r"available: .*anthropic"):
        build_provider(ModelRef(provider="nope"), get_settings())


def test_register_custom_provider() -> None:
    register_provider("custom", lambda ref, s: FakeLLM(model=ref.name))
    assert "custom" in available_providers()
    assert build_provider(ModelRef(provider="custom", name="m"), get_settings()).model == "m"


def test_real_adapters_construct_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    a = build_provider(ModelRef(provider="anthropic", name="claude-opus-5-5"), get_settings())
    o = build_provider(ModelRef(provider="openai", name="gpt-x"), get_settings())
    assert (a.name, o.name) == ("anthropic", "openai")
