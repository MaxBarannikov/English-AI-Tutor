"""Inspects the learner's last message. Produces structured corrections only."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from tutor.llm import build_chat_model
from tutor.models import AnalysisResult, Correction
from tutor.prompts import ANALYZER
from tutor.state import TutorState


async def analyze(
    state: TutorState, config: RunnableConfig | None = None
) -> dict[str, list[Correction]]:
    messages = state["messages"]
    last = messages[-1] if messages else None

    # Nothing to analyse on the opening turn — decide that in Python, not by
    # asking the model to find mistakes in a message that does not exist.
    if not isinstance(last, HumanMessage) or not last.text.strip():
        return {"corrections": []}

    llm = build_chat_model("analyzer").with_structured_output(AnalysisResult)
    system = SystemMessage(ANALYZER.format(level=state["level"]))
    result = await llm.ainvoke([system, HumanMessage(last.text)], config=config)

    if not isinstance(result, AnalysisResult):
        raise RuntimeError(
            f"analyzer returned {type(result).__name__}, expected AnalysisResult"
        )

    return {"corrections": result.corrections}
