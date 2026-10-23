"""Build providers from declarative config, with a plugin registry.

Third-party packages add providers without touching this repo by exposing an entry
point in the `harness.providers` group that resolves to a `ProviderFactory`:

    [project.entry-points."harness.providers"]
    bedrock = "my_pkg.bedrock:make_provider"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.config import Settings
from harness.llm.base import LLMProvider
from harness.llm.fake import FakeLLM
from harness.llm.fallback import FallbackProvider
from harness.llm.retry import RetryingProvider, RetryPolicy

log = logging.getLogger(__name__)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "anthropic"
    name: str = "claude-opus-5-5"
    effort: Effort = "high"


class ModelConfig(ModelRef):
    max_tokens: int = Field(default=16000, gt=0)
    fallbacks: list[ModelRef] = Field(default_factory=list)


ProviderFactory = Callable[[ModelRef, Settings], LLMProvider]


def _anthropic(ref: ModelRef, s: Settings) -> LLMProvider:
    from harness.llm.anthropic_provider import AnthropicProvider

    key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
    return AnthropicProvider(ref.name, api_key=key, timeout_s=s.llm_timeout_s, effort=ref.effort)


def _openai(ref: ModelRef, s: Settings) -> LLMProvider:
    from harness.llm.openai_provider import OpenAIProvider

    key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
    return OpenAIProvider(ref.name, api_key=key, timeout_s=s.llm_timeout_s)


def _fake(ref: ModelRef, s: Settings) -> LLMProvider:
    return FakeLLM(model=ref.name)


_REGISTRY: dict[str, ProviderFactory] = {"anthropic": _anthropic, "openai": _openai, "fake": _fake}
_plugins_loaded = False


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name] = factory


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="harness.providers"):
        try:
            _REGISTRY.setdefault(ep.name, ep.load())
        except Exception:
            log.exception("failed to load provider plugin %r", ep.name)


def available_providers() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)


def build_provider(ref: ModelRef, settings: Settings) -> LLMProvider:
    _load_plugins()
    try:
        factory = _REGISTRY[ref.provider]
    except KeyError:
        raise ValueError(
            f"unknown provider {ref.provider!r}; available: {', '.join(available_providers())}"
        ) from None
    return factory(ref, settings)


def build_llm(config: ModelConfig, settings: Settings) -> LLMProvider:
    """Primary + fallbacks, each wrapped with timeout/retry."""
    policy = RetryPolicy(max_retries=settings.llm_max_retries, timeout_s=settings.llm_timeout_s)
    chain = [
        RetryingProvider(build_provider(ref, settings), policy)
        for ref in [config, *config.fallbacks]
    ]
    return chain[0] if len(chain) == 1 else FallbackProvider(chain)
