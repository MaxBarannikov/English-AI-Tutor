"""Score the analyzer against labelled sentences.

    uv run python evals/run.py [--limit N]

Reports precision / recall / F0.5 per error type, plus the false-positive rate
on sentences that are already correct. False positives are the failure mode
that matters: an analyzer inventing errors makes learners stop talking.
"""

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from tutor.callbacks import UsageCallbackHandler
from tutor.config import get_settings
from tutor.models import CEFRLevel, ErrorType
from tutor.nodes.analyzer import analyze

DATASET = Path(__file__).parent / "dataset.jsonl"
BETA_SQUARED = 0.25  # F0.5 — precision counts double


class Example:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.id: str = raw["id"]
        self.text: str = raw["text"]
        self.level: CEFRLevel = raw["level"]
        self.correct: bool = raw["correct"]
        self.expected: Counter[ErrorType] = Counter(
            item["error_type"] for item in raw["expected"]
        )


def load(path: Path, limit: int | None) -> list[Example]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    examples = [Example(json.loads(line)) for line in lines]
    return examples[:limit] if limit else examples


async def predict(
    example: Example, usage: UsageCallbackHandler, gate: asyncio.Semaphore
) -> Counter[ErrorType]:
    state = {"messages": [HumanMessage(example.text)], "level": example.level}
    async with gate:
        result = await analyze(state, config={"callbacks": [usage]})
    return Counter(c.error_type for c in result["corrections"])


def f_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + BETA_SQUARED) * precision * recall / (BETA_SQUARED * precision + recall)


def score(
    examples: list[Example], predictions: list[Counter[ErrorType]]
) -> tuple[str, float]:
    """Returns the per-error-type table and the false-positive rate."""
    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()

    for example, predicted in zip(examples, predictions, strict=True):
        for error_type in set(example.expected) | set(predicted):
            hits = min(example.expected[error_type], predicted[error_type])
            true_positive[error_type] += hits
            false_positive[error_type] += predicted[error_type] - hits
            false_negative[error_type] += example.expected[error_type] - hits

    rows = ["| error type | P | R | F0.5 | support |", "|---|---|---|---|---|"]
    for error_type in sorted(
        set(true_positive) | set(false_positive) | set(false_negative)
    ):
        tp, fp, fn = (
            true_positive[error_type],
            false_positive[error_type],
            false_negative[error_type],
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            f"| {error_type} | {precision:.2f} | {recall:.2f} | "
            f"{f_score(precision, recall):.2f} | {tp + fn} |"
        )

    controls = [
        predicted
        for example, predicted in zip(examples, predictions, strict=True)
        if example.correct
    ]
    flagged = sum(1 for predicted in controls if predicted.total() > 0)
    fpr = flagged / len(controls) if controls else 0.0
    return "\n".join(rows), fpr


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel requests. Keep low for a local model or a small rate limit.",
    )
    args = parser.parse_args()

    settings = get_settings()
    examples = load(DATASET, args.limit)
    usage = UsageCallbackHandler()
    gate = asyncio.Semaphore(args.concurrency)

    predictions = await asyncio.gather(*(predict(e, usage, gate) for e in examples))

    table, fpr = score(examples, predictions)
    cost: Decimal | None = usage.cost_usd
    price = f"${cost:.4f}" if cost is not None else "unknown (unpriced model)"

    print(f"model: {settings.analyzer_model} ({settings.provider})")
    print(f"date: {datetime.now(UTC).date().isoformat()}")
    print(f"examples: {len(examples)}  cost: {price}")
    print()
    print(table)
    print()
    print(f"false-positive rate on correct sentences: {fpr:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
