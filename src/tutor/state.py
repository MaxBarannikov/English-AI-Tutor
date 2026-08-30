"""Graph state.

`responder` and `analyzer` run concurrently. They write to `reply` and
`corrections` respectively — no shared key, so no reducer is needed for them.
Any new key written by both branches must get a reducer or LangGraph raises
`InvalidUpdateError`.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from tutor.models import CEFRLevel, Correction, SessionSummary


class TutorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    level: CEFRLevel

    # Written by responder, consumed by merge.
    reply: str
    # Written by analyzer: every mistake in the last learner message.
    corrections: list[Correction]

    # Written by merge: the subset the learner actually sees this turn.
    shown_corrections: list[Correction]
    # Everything found across the session, including `style` — feeds the summary.
    correction_log: Annotated[list[Correction], operator.add]

    summary: SessionSummary | None
    end_session: bool
