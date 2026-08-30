"""Conversation partner. Knows nothing about corrections."""

import logging

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from tutor.llm import build_chat_model
from tutor.prompts import RESPONDER
from tutor.state import TutorState

logger = logging.getLogger(__name__)


async def respond(
    state: TutorState, config: RunnableConfig | None = None
) -> dict[str, str]:
    llm = build_chat_model("responder")
    system = SystemMessage(RESPONDER.format(level=state["level"]))
    response = await llm.ainvoke([system, *state["messages"]], config=config)

    reply = response.text.strip()
    if not reply:
        raise RuntimeError("responder returned an empty reply")
    if "?" not in reply:
        logger.info("responder reply did not end with a question: %r", reply)

    return {"reply": reply}
