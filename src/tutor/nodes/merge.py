"""Assembles the turn: reply first, then the corrections worth showing."""

from typing import Any

from langchain_core.messages import AIMessage

from tutor.severity import select_corrections
from tutor.state import TutorState


def merge_turn(state: TutorState) -> dict[str, Any]:
    corrections = state.get("corrections", [])
    return {
        "messages": [AIMessage(state["reply"])],
        "shown_corrections": select_corrections(corrections),
        "correction_log": corrections,
    }
