"""Token pricing. Cost must never be invisible, so an unknown model is loud."""

import json
import logging
from decimal import Decimal

from tutor.config import get_settings

logger = logging.getLogger(__name__)

_MILLION = Decimal(1_000_000)

# USD per 1M tokens, (input, output). Anthropic list prices as of 2026-08-30.
# Other providers: supply them through TUTOR_PRICING_FILE rather than guessing.
_PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "claude-opus-5": (Decimal("5"), Decimal("25")),
    "claude-opus-4-8": (Decimal("5"), Decimal("25")),
    "claude-sonnet-5": (Decimal("3"), Decimal("15")),
    "claude-sonnet-4-6": (Decimal("3"), Decimal("15")),
    "claude-haiku-4-5": (Decimal("1"), Decimal("5")),
    "claude-fable-5": (Decimal("10"), Decimal("50")),
}

_unpriced_models: set[str] = set()


def _load_overrides() -> dict[str, tuple[Decimal, Decimal]]:
    path = get_settings().pricing_file
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    return {
        model: (Decimal(price["input"]), Decimal(price["output"]))
        for model, price in raw.items()
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Cost of one call, or None when the model has no known price."""
    prices = {**_PRICES, **_load_overrides()}
    price = prices.get(model)
    if price is None:
        if model not in _unpriced_models:
            _unpriced_models.add(model)
            logger.warning(
                "No price for model %r — reporting tokens without cost. "
                "Add it to TUTOR_PRICING_FILE.",
                model,
            )
        return None
    input_price, output_price = price
    return (input_tokens * input_price + output_tokens * output_price) / _MILLION
