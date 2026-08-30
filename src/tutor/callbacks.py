"""Per-turn token and cost accounting.

One handler instance per graph invocation; the UI reads it after the run.
Usage stays out of the graph state so nodes remain pure state transitions.
"""

from decimal import Decimal
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from tutor.models import TurnUsage
from tutor.pricing import estimate_cost


class UsageCallbackHandler(AsyncCallbackHandler):
    """Collects usage from every LLM call made during one turn."""

    def __init__(self) -> None:
        self.usages: list[TurnUsage] = []

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if not isinstance(message, AIMessage):
                    continue
                usage = message.usage_metadata
                if usage is None:
                    continue
                model = str(
                    message.response_metadata.get("model_name")
                    or message.response_metadata.get("model")
                    or "unknown"
                )
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                self.usages.append(
                    TurnUsage(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=estimate_cost(model, input_tokens, output_tokens),
                    )
                )

    @property
    def input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.usages)

    @property
    def output_tokens(self) -> int:
        return sum(usage.output_tokens for usage in self.usages)

    @property
    def cost_usd(self) -> Decimal | None:
        """Total cost, or None if any call had no known price."""
        costs = [usage.cost_usd for usage in self.usages]
        if not costs or any(cost is None for cost in costs):
            return None
        return sum((cost for cost in costs if cost is not None), Decimal(0))
