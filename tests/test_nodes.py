from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import StubChatModel, correction
from tutor.models import AnalysisResult, SessionSummary
from tutor.nodes.analyzer import analyze
from tutor.nodes.merge import merge_turn
from tutor.nodes.responder import respond
from tutor.nodes.summary import summarise
from tutor.state import TutorState

Patch = Callable[..., StubChatModel]


async def test_responder_returns_only_reply(patch_model: Patch) -> None:
    patch_model("responder", AIMessage("Nice! Where did you go?"))
    state: TutorState = {"messages": [HumanMessage("I go to Rome")], "level": "B1"}

    assert await respond(state) == {"reply": "Nice! Where did you go?"}


async def test_responder_raises_on_empty_reply(patch_model: Patch) -> None:
    patch_model("responder", AIMessage("   "))
    state: TutorState = {"messages": [HumanMessage("hi")], "level": "B1"}

    with pytest.raises(RuntimeError, match="empty reply"):
        await respond(state)


async def test_analyzer_returns_corrections(patch_model: Patch) -> None:
    stub = patch_model("analyzer", AnalysisResult(corrections=[correction("critical")]))
    state: TutorState = {"messages": [HumanMessage("I has a dog")], "level": "B1"}

    result = await analyze(state)

    assert [c.severity for c in result["corrections"]] == ["critical"]
    assert len(stub.calls) == 1


async def test_analyzer_skips_model_when_there_is_nothing_to_analyse(
    patch_model: Patch,
) -> None:
    stub = patch_model("analyzer", AnalysisResult(corrections=[correction("critical")]))
    state: TutorState = {"messages": [AIMessage("How are you?")], "level": "B1"}

    assert await analyze(state) == {"corrections": []}
    assert stub.calls == []


def test_merge_applies_gating_and_keeps_full_log() -> None:
    corrections = [correction("critical"), correction("minor"), correction("style")]
    state: TutorState = {
        "messages": [HumanMessage("I has a dog")],
        "level": "B1",
        "reply": "Nice! What is its name?",
        "corrections": corrections,
    }

    update = merge_turn(state)

    assert update["messages"] == [AIMessage("Nice! What is its name?")]
    assert [c.severity for c in update["shown_corrections"]] == ["critical", "minor"]
    assert update["correction_log"] == corrections


async def test_summary_reads_the_whole_log(patch_model: Patch) -> None:
    summary = SessionSummary(
        strengths=["clear questions"],
        recurring_errors=["subject-verb agreement"],
        style_notes=["'very good' -> 'excellent'"],
        level_recommendation="Practise the present simple.",
    )
    stub = patch_model("summary", summary)
    state: TutorState = {
        "messages": [HumanMessage("I has a dog")],
        "level": "B1",
        "correction_log": [correction("style")],
    }

    assert await summarise(state) == {"summary": summary}
    assert "style" in str(stub.calls[0][-1].content)
