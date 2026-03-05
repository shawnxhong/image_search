from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SearchState(TypedDict):
    query_mode: str
    query_text: str | None
    image_path: str | None


def build_search_graph():
    """Phase-0 LangGraph skeleton for future node expansion."""

    graph = StateGraph(SearchState)

    def normalize_input(state: SearchState) -> SearchState:
        return state

    def plan_tools(state: SearchState) -> SearchState:
        return state

    graph.add_node("normalize_input", normalize_input)
    graph.add_node("plan_tools", plan_tools)
    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "plan_tools")
    graph.add_edge("plan_tools", END)
    return graph.compile()
