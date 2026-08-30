import json
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tutor.callbacks import UsageCallbackHandler
from tutor.config import get_settings
from tutor.pricing import estimate_cost


def _llm_result(model: str, input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        "hi",
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_known_model_is_priced() -> None:
    # 1M in + 1M out on Opus 5 = $5 + $25.
    assert estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == Decimal(30)


def test_unknown_model_has_no_price() -> None:
    assert estimate_cost("qwen2.5:14b", 1000, 1000) is None


def test_pricing_file_supplies_missing_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"qwen2.5:14b": {"input": "0", "output": "0"}}))
    monkeypatch.setenv("TUTOR_PRICING_FILE", str(prices))
    get_settings.cache_clear()

    assert estimate_cost("qwen2.5:14b", 1000, 1000) == Decimal(0)


async def test_handler_sums_tokens_and_cost() -> None:
    handler = UsageCallbackHandler()

    await handler.on_llm_end(_llm_result("claude-opus-5", 1_000_000, 0))
    await handler.on_llm_end(_llm_result("claude-opus-5", 0, 1_000_000))

    assert handler.input_tokens == 1_000_000
    assert handler.output_tokens == 1_000_000
    assert handler.cost_usd == Decimal(30)


async def test_cost_is_none_when_any_call_is_unpriced() -> None:
    handler = UsageCallbackHandler()

    await handler.on_llm_end(_llm_result("claude-opus-5", 100, 100))
    await handler.on_llm_end(_llm_result("qwen2.5:14b", 100, 100))

    assert handler.input_tokens == 200
    assert handler.cost_usd is None
