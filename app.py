"""Chainlit UI: reply first, corrections second, cost always visible."""

import logging
from contextlib import AsyncExitStack
from decimal import Decimal
from typing import Any, cast

import chainlit as cl
from chainlit.input_widget import Select
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from tutor.asr import transcribe
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
    # Typing wins over a pending dictation, so no draft can be sent by surprise.
    await _drop_draft()

    if message.content.strip() == END_COMMAND:
        await _end_session()
        return

    await _run_turn(message.content)


async def _run_turn(text: str) -> None:
    result = await _invoke({"messages": [HumanMessage(text)]})
    corrections = cast(list[Correction], result.get("shown_corrections", []))
    await cl.Message(
        content=result["reply"], elements=_correction_elements(corrections)
    ).send()


# --- Voice dictation -------------------------------------------------------
#
# Chainlit has no way to prefill the composer, so a recording becomes a draft
# message with Send / Discard actions: the learner reads back what was
# recognised before anything is sent to the tutor.


@cl.on_audio_start
async def audio_start() -> bool:
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def audio_chunk(chunk: cl.InputAudioChunk) -> None:
    chunks = cast(list[bytes], cl.user_session.get("audio_chunks") or [])
    chunks.append(chunk.data)
    cl.user_session.set("audio_chunks", chunks)


@cl.on_audio_end
async def audio_end() -> None:
    chunks = cast(list[bytes], cl.user_session.get("audio_chunks") or [])
    cl.user_session.set("audio_chunks", [])

    async with cl.Step(name="Transcribing", type="tool") as step:
        try:
            text = await transcribe(b"".join(chunks))
        except Exception as error:  # surfaced to the learner, never swallowed
            logger.exception("transcription failed")
            step.output = str(error)
            await cl.Message(
                content="I couldn't transcribe that recording. Try again, or type it."
            ).send()
            return
        step.output = text or "(nothing recognised)"

    if not text:
        await cl.Message(
            content="I didn't catch anything — try recording again."
        ).send()
        return

    await _drop_draft()
    draft = cl.Message(
        content=f"🎙 **Draft:** {text}\n\n_Send it, or type a corrected version._",
        actions=[
            cl.Action(name="send_draft", payload={"text": text}, label="Send"),
            cl.Action(name="discard_draft", payload={}, label="Discard"),
        ],
    )
    await draft.send()
    cl.user_session.set("draft_message_id", draft.id)


@cl.action_callback("send_draft")
async def send_draft(action: cl.Action) -> None:
    text = str(action.payload["text"])
    await _drop_draft()
    # Echo it as the learner's own turn so the transcript reads naturally.
    await cl.Message(content=text, type="user_message").send()
    await _run_turn(text)


@cl.action_callback("discard_draft")
async def discard_draft(action: cl.Action) -> None:
    await _drop_draft()


async def _drop_draft() -> None:
    """Remove the pending draft message, if any."""
    message_id = cl.user_session.get("draft_message_id")
    if message_id:
        await cl.Message(content="", id=str(message_id)).remove()
        cl.user_session.set("draft_message_id", None)


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
