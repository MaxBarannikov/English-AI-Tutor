"""Chainlit UI: reply first, corrections second, cost always visible."""

import logging
from contextlib import AsyncExitStack
from decimal import Decimal
from typing import Any, cast

import chainlit as cl
from chainlit.input_widget import Select
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from tutor.callbacks import UsageCallbackHandler
from tutor.config import get_settings
from tutor.graph import open_graph
from tutor.models import CEFRLevel, Correction, SessionSummary

logger = logging.getLogger(__name__)

LEVELS: list[str] = ["A1", "A2", "B1", "B2", "C1", "C2"]
END_COMMAND = "/summary"

_stack = AsyncExitStack()
_graph: CompiledStateGraph | None = None


@cl.on_app_startup
async def open_checkpointer() -> None:
    global _graph
    _graph = await _stack.enter_async_context(open_graph())


@cl.on_app_shutdown
async def close_checkpointer() -> None:
    await _stack.aclose()


@cl.on_chat_start
async def start() -> None:
    default = get_settings().default_level
    cl.user_session.set("level", default)
    await cl.ChatSettings(
        [
            Select(
                id="level",
                label="CEFR level",
                values=LEVELS,
                initial_index=LEVELS.index(default),
            )
        ]
    ).send()
    await cl.Message(
        content=(
            "Hi! Let's talk in English — I'll reply as a conversation partner and "
            f"point out mistakes worth fixing. Send `{END_COMMAND}` when you want "
            "the end-of-session report."
        )
    ).send()


@cl.on_settings_update
async def update_level(settings: dict[str, Any]) -> None:
    cl.user_session.set("level", settings["level"])


@cl.on_message
async def handle(message: cl.Message) -> None:
    if message.content.strip() == END_COMMAND:
        await _end_session()
        return

    result = await _invoke({"messages": [HumanMessage(message.content)]})
    corrections = cast(list[Correction], result.get("shown_corrections", []))
    await cl.Message(
        content=result["reply"], elements=_correction_elements(corrections)
    ).send()


async def _end_session() -> None:
    result = await _invoke({"end_session": True})
    summary = cast(SessionSummary, result["summary"])
    await cl.Message(content=_format_summary(summary)).send()


async def _invoke(update: dict[str, Any]) -> dict[str, Any]:
    if _graph is None:
        raise RuntimeError("graph was not initialised on app startup")

    level = cast(CEFRLevel, cl.user_session.get("level"))
    usage = UsageCallbackHandler()
    config = {
        "configurable": {"thread_id": cl.context.session.thread_id},
        "callbacks": [usage],
    }

    result = await _graph.ainvoke({**update, "level": level}, config=config)
    await _report_cost(usage)
    return cast(dict[str, Any], result)


async def _report_cost(usage: UsageCallbackHandler) -> None:
    cost: Decimal | None = usage.cost_usd
    price = f"${cost:.4f}" if cost is not None else "cost unknown (unpriced model)"
    await cl.Message(
        content=(
            f"`{usage.input_tokens} in / {usage.output_tokens} out tokens — {price}`"
        ),
        author="usage",
    ).send()


def _correction_elements(corrections: list[Correction]) -> list[cl.Text]:
    if not corrections:
        return []
    lines = [
        f"**{c.original}** → **{c.corrected}**  \n"
        f"_{c.error_type}, {c.severity}_ — {c.explanation}"
        for c in corrections
    ]
    return [cl.Text(name="Corrections", content="\n\n".join(lines), display="inline")]


def _format_summary(summary: SessionSummary) -> str:
    sections = [
        ("What went well", summary.strengths),
        ("Recurring mistakes", summary.recurring_errors),
        ("More natural phrasing", summary.style_notes),
    ]
    parts = ["## Session report"]
    for title, items in sections:
        if items:
            parts.append(f"**{title}**\n" + "\n".join(f"- {item}" for item in items))
    parts.append(f"**Next**\n{summary.level_recommendation}")
    return "\n\n".join(parts)
