"""End-of-session report. The only place `style` corrections surface."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from tutor.llm import build_chat_model
from tutor.models import SessionSummary
from tutor.prompts import SUMMARY
from tutor.state import TutorState


async def summarise(
    state: TutorState, config: RunnableConfig | None = None
) -> dict[str, SessionSummary]:
    llm = build_chat_model("summary").with_structured_output(SessionSummary)
    system = SystemMessage(SUMMARY.format(level=state["level"]))
    findings = json.dumps(
        [c.model_dump() for c in state.get("correction_log", [])],
        ensure_ascii=False,
        indent=2,
    )
    request = HumanMessage(f"Corrections found during this session:\n{findings}")

    result = await llm.ainvoke([system, *state["messages"], request], config=config)
    if not isinstance(result, SessionSummary):
        raise RuntimeError(
            f"summary returned {type(result).__name__}, expected SessionSummary"
        )

    return {"summary": result}
