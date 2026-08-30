"""Graph assembly.

START ─┬─> responder ─┐
       │              ├─> merge ─> END
       └─> analyzer ──┘
       └─> summary ─> END        (when the session is ending)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tutor.config import get_settings
from tutor.nodes.analyzer import analyze
from tutor.nodes.merge import merge_turn
from tutor.nodes.responder import respond
from tutor.nodes.summary import summarise
from tutor.state import TutorState

# The graph carries no runtime context, and input/output share the state schema.
type TutorGraph = StateGraph[TutorState, None, TutorState, TutorState]
type CompiledTutorGraph = CompiledStateGraph[TutorState, None, TutorState, TutorState]


def route_turn(state: TutorState) -> list[str]:
    """Ending the session skips the conversation branches entirely."""
    if state.get("end_session", False):
        return ["summary"]
    return ["responder", "analyzer"]


def build_graph() -> TutorGraph:
    builder: TutorGraph = StateGraph(TutorState)

    builder.add_node("responder", respond)
    builder.add_node("analyzer", analyze)
    builder.add_node("merge", merge_turn)
    builder.add_node("summary", summarise)

    builder.add_conditional_edges(
        START, route_turn, ["responder", "analyzer", "summary"]
    )
    builder.add_edge("responder", "merge")
    builder.add_edge("analyzer", "merge")
    builder.add_edge("merge", END)
    builder.add_edge("summary", END)

    return builder


@asynccontextmanager
async def open_graph() -> AsyncIterator[CompiledTutorGraph]:
    """Compiled graph with a checkpointer, so sessions survive a restart."""
    db = get_settings().checkpoint_db
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        yield build_graph().compile(checkpointer=saver)
