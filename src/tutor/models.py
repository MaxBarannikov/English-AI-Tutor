"""Data models crossing node, prompt and UI boundaries.

Field descriptions on the structured-output models are read by the LLM — they
are part of the prompt, not documentation.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

Severity = Literal["critical", "minor", "style"]

ErrorType = Literal[
    "grammar",
    "tense",
    "aspect",
    "article",
    "preposition",
    "word_order",
    "agreement",
    "vocabulary",
    "spelling",
]


class Correction(BaseModel):
    original: str = Field(
        description="The exact fragment of the learner's message that is wrong."
    )
    corrected: str = Field(description="The same fragment, rewritten correctly.")
    error_type: ErrorType = Field(description="Which kind of mistake this is.")
    severity: Severity = Field(
        description=(
            "critical — breaks comprehension or is a rule the learner's level "
            "must already own; minor — noticeable but the meaning survives; "
            "style — the sentence is correct, a native speaker would phrase it "
            "differently."
        )
    )
    explanation: str = Field(
        description="One sentence, addressed to the learner, in simple English."
    )


class AnalysisResult(BaseModel):
    """Wrapper so the model returns an object, not a bare list."""

    corrections: list[Correction] = Field(
        description=(
            "Every mistake found in the learner's last message. Empty list when "
            "the message is correct — do not invent mistakes."
        )
    )


class SessionSummary(BaseModel):
    strengths: list[str] = Field(
        description="What the learner did well, in concrete terms."
    )
    recurring_errors: list[str] = Field(
        description="Patterns that came up more than once, phrased as rules."
    )
    style_notes: list[str] = Field(
        description="Phrasing that was correct but not natural."
    )
    level_recommendation: str = Field(
        description="One sentence on what to practise next."
    )


class TurnUsage(BaseModel):
    """Token and cost accounting for one graph invocation."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None = None
