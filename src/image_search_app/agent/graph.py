"""SearchAgent — thin wrapper that invokes the LangGraph agentic pipeline."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Generator

from image_search_app.agent.langgraph_flow import (
    assemble_response,
    build_initial_state,
    build_search_graph,
    invoke_graph_with_steps,
    preprocess_query,
)
from image_search_app.config import settings
from image_search_app.schemas import AgentStep, DualListSearchResponse

logger = logging.getLogger(__name__)


class SearchAgent:
    """Orchestrates text and image search via the LangGraph agent."""

    def __init__(self) -> None:
        self._graph = build_search_graph()

    def search_text(self, query: str, top_k: int | None = None) -> DualListSearchResponse:
        query = preprocess_query(query)
        initial_state = build_initial_state(query)
        result = self._graph.invoke(
            initial_state,
            config={"recursion_limit": settings.llm_max_agent_iterations * 2 + 2},
        )
        tool_results = result.get("tool_results", {})
        return assemble_response(tool_results)

    def search_text_stream(
        self, query: str, top_k: int | None = None,
    ) -> Generator[AgentStep | DualListSearchResponse, None, None]:
        """Run agentic search, yielding AgentStep events as the graph executes.

        The final item yielded is always a DualListSearchResponse.
        """
        step_queue: queue.Queue[AgentStep | DualListSearchResponse | None] = queue.Queue()

        def on_step(step: AgentStep) -> None:
            step_queue.put(step)

        def run_graph() -> None:
            try:
                processed_query = preprocess_query(query)
                if processed_query != query:
                    step_queue.put(AgentStep(
                        step_type="thinking",
                        message=f"Translated query: {processed_query}",
                    ))
                initial_state = build_initial_state(processed_query)
                result = invoke_graph_with_steps(
                    self._graph,
                    initial_state,
                    on_step=on_step,
                    recursion_limit=settings.llm_max_agent_iterations * 2 + 2,
                )
                tool_results = result.get("tool_results", {})
                response = assemble_response(tool_results)
                step_queue.put(AgentStep(step_type="done", message="Search complete"))
                step_queue.put(response)
            except Exception as exc:
                logger.exception("Agent graph failed")
                step_queue.put(AgentStep(step_type="error", message=str(exc)))
            finally:
                step_queue.put(None)  # Sentinel

        thread = threading.Thread(target=run_graph, daemon=True)
        thread.start()

        while True:
            item = step_queue.get()
            if item is None:
                break
            yield item

        thread.join(timeout=5)

    def search_image(self, image_path: str, query: str | None, top_k: int | None = None) -> DualListSearchResponse:
        if query:
            return self.search_text(query, top_k=top_k)
        return DualListSearchResponse(solid_results=[], soft_results=[])
