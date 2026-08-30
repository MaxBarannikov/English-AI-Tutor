"""One end-to-end run of the compiled graph with fakes."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import StubChatModel, correction
from tutor.graph import build_graph, route_turn
from tutor.models import AnalysisResult, SessionSummary
from tutor.state import TutorState


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, StubChatModel]:
    models = {
        "responder": StubChatModel(AIMessage("Sounds fun! Who did you go with?")),
        "analyzer": StubChatModel(
            AnalysisResult(
                corrections=[
                    correction("critical", "I has"),
                    correction("minor", "in Rome"),
                    correction("style", "very nice"),
                ]
            )
        ),
        "summary": StubChatModel(
            SessionSummary(
                strengths=["kept the conversation going"],
                recurring_errors=["subject-verb agreement"],
                style_notes=["'very nice' -> 'lovely'"],
                level_recommendation="Practise the past simple.",
            )
        ),
    }
    for module, stub in models.items():
        monkeypatch.setattr(
            f"tutor.nodes.{module}.build_chat_model", lambda role, s=stub: s
        )
    return models


def test_route_turn() -> None:
    assert route_turn({"end_session": False}) == ["responder", "analyzer"]
    assert route_turn({"end_session": True}) == ["summary"]


async def test_conversation_turn_merges_parallel_branches(
    stubs: dict[str, StubChatModel],
) -> None:
    graph = build_graph().compile()
    state: TutorState = {
        "messages": [HumanMessage("I has a very nice trip in Rome")],
        "level": "B1",
    }

    result = await graph.ainvoke(state)

    # Both branches ran and merged.
    assert result["reply"] == "Sounds fun! Who did you go with?"
    assert result["messages"][-1].text == "Sounds fun! Who did you go with?"
    # Gating applied on the way out; the full set is kept for the summary.
    assert [c.severity for c in result["shown_corrections"]] == ["critical", "minor"]
    assert len(result["correction_log"]) == 3
    assert result.get("summary") is None


async def test_ending_the_session_skips_the_conversation_branches(
    stubs: dict[str, StubChatModel],
) -> None:
    graph = build_graph().compile()
    state: TutorState = {
        "messages": [HumanMessage("I has a dog")],
        "level": "B1",
        "correction_log": [correction("style")],
        "end_session": True,
    }

    result = await graph.ainvoke(state)

    assert result["summary"].level_recommendation == "Practise the past simple."
    assert stubs["responder"].calls == []
    assert stubs["analyzer"].calls == []


async def test_correction_log_accumulates_across_turns(
    stubs: dict[str, StubChatModel],
) -> None:
    graph = build_graph().compile()
    state: TutorState = {"messages": [HumanMessage("I has a dog")], "level": "B1"}

    first = await graph.ainvoke(state)
    second = await graph.ainvoke(
        {**first, "messages": [*first["messages"], HumanMessage("I has a cat")]}
    )

    assert len(second["correction_log"]) == 6
