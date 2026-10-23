"""Token -> USD cost accounting.

Prices are USD per 1M tokens. Only first-party Anthropic list prices ship by default;
register others (OpenAI, partner platforms, negotiated rates) with `register_price`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from harness.llm.types import Usage

log = logging.getLogger(__name__)

# Cache reads bill at 10% of input, cache writes (5-minute TTL) at 125%.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


_PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.0, 50.0),
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-5-5": Price(0.10, 0.50),
    "fake-1": Price(0.0, 0.0),
}
_warned: set[str] = set()


def register_price(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    if input_per_mtok < 0 or output_per_mtok < 0:
        raise ValueError("prices must be non-negative")
    _PRICES[model] = Price(input_per_mtok, output_per_mtok)


def get_price(model: str) -> Price | None:
    return _PRICES.get(model)


def cost_usd(model: str, usage: Usage) -> float:
    """Cost of one call. Unknown models cost 0 and log one warning per model."""
    price = _PRICES.get(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("no price registered for model %r; cost reported as 0", model)
        return 0.0
    per_in = price.input_per_mtok / 1_000_000
    per_out = price.output_per_mtok / 1_000_000
    return (
        usage.input_tokens * per_in
        + usage.cache_read_tokens * per_in * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * per_in * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * per_out
    )
