"""Severity gating.

Over-correcting makes learners stop talking, so how much is shown is a product
decision, not something the model is asked to judge. Deterministic and testable.
"""

from tutor.models import Correction

MAX_MINOR_PER_TURN = 1


def select_corrections(corrections: list[Correction]) -> list[Correction]:
    """The corrections shown mid-dialogue: all critical, at most one minor.

    `style` never appears during the conversation — it is saved for the
    end-of-session summary.
    """
    shown = [c for c in corrections if c.severity == "critical"]
    minor = [c for c in corrections if c.severity == "minor"]
    return shown + minor[:MAX_MINOR_PER_TURN]
