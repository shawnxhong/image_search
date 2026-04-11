"""LangGraph-based agentic search pipeline.

Implements a ReAct loop following the reference sample pattern:
  assistant node → (has tool request?) → tool node → assistant node → ...
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from image_search_app.config import settings
from image_search_app.db import ImageRecord, get_session
from image_search_app.schemas import (
    AgentStep,
    DualListSearchResponse,
    MatchExplanation,
    SearchResultItem,
)
from image_search_app.tools.llm import (
    get_llm_service,
    parse_tool_requests,
    strip_thinking,
)
from image_search_app.tools.search_tools import (
    TOOL_DEFINITIONS,
    execute_tool,
)
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a search assistant for a personal photo library. Given a user query, \
decide which search tools to call to find relevant images.

Guidelines:
- If the query mentions a person by name, use search_by_person.
- If the query mentions MULTIPLE people, call search_by_person ONCE PER PERSON \
so that each person is treated as a separate required condition for filtering.
- If the query mentions a COUNT of people (e.g. 'solo', 'two people', 'group', \
'nobody'), use search_by_person_count. Do NOT combine with search_by_person \
unless a specific name is also mentioned.
  - 'solo', 'alone', 'just one person' → count=1
  - 'couple', 'two people', 'pair' → count=2
  - 'group', 'crowd', 'many people' → min_count=4
  - 'no people', 'nobody', 'empty scene' → count=0
- If the query describes visual content (scenes, objects, actions), use \
search_by_caption with ONLY the descriptive words -- strip out person names, \
dates, and geographic location names.
- If the query mentions a time period, use search_by_time.
- search_by_location ONLY works with proper geographic names (cities, states, \
countries) derived from GPS metadata, e.g. New York, California, France. \
For scene/setting words like library, beach, park, kitchen, office, \
use search_by_caption instead -- the captions describe what is in the photo.
- You may call multiple tools in sequence for a single query.
- Do NOT call tools that are not relevant to the query.
- After you have called all necessary tools, respond to the user without \
Action/Action Input tags. Just say DONE.

Examples:
- Query: Tom
  Tools: search_by_person(name=Tom)

- Query: Trump with Elon Musk
  Tools: search_by_person(name=Trump), search_by_person(name=Elon Musk)
  Note: Two separate calls so only images containing BOTH appear in solid results

- Query: picture with only two people inside
  Tools: search_by_person_count(count=2), search_by_caption(query=indoor)

- Query: solo portrait of Alice
  Tools: search_by_person(name=Alice), search_by_person_count(count=1)

- Query: group photo in Paris
  Tools: search_by_person_count(min_count=4), search_by_location(location=Paris)

- Query: photo with no people
  Tools: search_by_person_count(count=0)

- Query: Alice at the beach in 2024
  Tools: search_by_person(name=Alice), search_by_caption(query=beach), \
search_by_time(description=2024)

- Query: photos in a library
  Tools: search_by_caption(query=library)
  Note: NOT search_by_location because library is a scene description, not a city/country

- Query: sunset in Paris last summer
  Tools: search_by_caption(query=sunset), search_by_location(location=Paris), \
search_by_time(description=last summer)

- Query: kids playing in the park
  Tools: search_by_caption(query=kids playing in the park)
  Note: NOT search_by_location because park is a scene, not a geographic name\
"""

# Type alias for the step callback
StepCallback = Callable[[AgentStep], None]

# Module-level holder for the step callback.
# LangGraph nodes can only receive state, so we use a context variable
# set before each graph invocation.
_step_callback: StepCallback | None = None


def _emit(step: AgentStep) -> None:
    """Emit a step event if a callback is registered."""
    if _step_callback is not None:
        _step_callback(step)


class AgentState(TypedDict):
    messages: list[dict]
    tool_requests: list[dict]  # List of {"name": ..., "args": ...}
    tool_results: dict[str, list[dict]]


def build_initial_state(query: str) -> dict:
    """Build the initial state for the graph."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "tool_requests": [],
        "tool_results": {},
    }


def _assistant_node(state: AgentState) -> dict:
    """Call the LLM to reason about the query and decide on tool calls."""
    _emit(AgentStep(step_type="thinking", message="Analyzing query..."))

    llm = get_llm_service()
    messages = state.get("messages", [])

    response = llm.chat(messages=messages, tools=TOOL_DEFINITIONS)
    text = strip_thinking(response)
    logger.info("LLM response: %s", text[:200])

    tool_reqs = parse_tool_requests(text)

    if tool_reqs:
        logger.info("Tool requests: %s", tool_reqs)
        for req in tool_reqs:
            _emit(AgentStep(
                step_type="tool_call",
                tool_name=req.get("name"),
                tool_args=req.get("args") if isinstance(req.get("args"), dict) else {"input": str(req.get("args", ""))},
                message=f"Calling {req.get('name')}",
            ))
        return {
            "messages": messages,
            "tool_requests": tool_reqs,
            "tool_results": state.get("tool_results", {}),
        }

    # No tool call — LLM is done
    return {
        "messages": messages + [{"role": "assistant", "content": text}],
        "tool_requests": [],
        "tool_results": state.get("tool_results", {}),
    }


def _tool_node(state: AgentState) -> dict:
    """Execute all requested tools and append results as tool messages."""
    tool_reqs = state.get("tool_requests", [])
    messages = list(state.get("messages", []))
    tool_results = dict(state.get("tool_results", {}))

    if not tool_reqs:
        return {"messages": messages, "tool_requests": [], "tool_results": tool_results}

    store = ChromaStore()
    embeddings = EmbeddingService()

    # Pre-process all requests: normalize args and assign deterministic keys.
    prepared: list[tuple[str, str, dict]] = []  # (call_key, name, args)
    for i, tool_req in enumerate(tool_reqs):
        name = tool_req.get("name", "")
        args = tool_req.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        prepared.append((f"{name}#{i}", name, args))

    # Execute all tools in parallel.
    def _run(item: tuple[str, str, dict]) -> tuple[str, str, list[dict]]:
        call_key, name, args = item
        logger.info("Executing tool: %s(%s)", name, args)
        results = execute_tool(name, args, store=store, embeddings=embeddings)
        return call_key, name, results

    with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
        completed = list(executor.map(_run, prepared))

    # Collect results in original order, emit events, build output.
    for call_key, name, results in completed:
        image_result_count = sum(1 for r in results if r.get("image_id"))
        logger.info("Tool %s returned %d results", name, image_result_count)

        _emit(AgentStep(
            step_type="tool_result",
            tool_name=name,
            result_count=image_result_count,
            message=f"{name} returned {image_result_count} results",
        ))

        messages.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(results[:20]),
        })

        # Each call gets its own key so multi-person queries stay separate constraints.
        tool_results[call_key] = results

    return {
        "messages": messages,
        "tool_requests": [],
        "tool_results": tool_results,
    }


def _route(state: AgentState) -> str:
    """Conditional edge: if there are tool requests, go to tool node; otherwise END."""
    if state.get("tool_requests"):
        return "tool"
    return END


def build_search_graph():
    """Build and compile the LangGraph search agent."""
    graph = StateGraph(AgentState)
    graph.add_node("assistant", _assistant_node)
    graph.add_node("tool", _tool_node)

    graph.add_conditional_edges("assistant", _route, {"tool": "tool", END: END})
    graph.add_edge("tool", END)
    graph.set_entry_point("assistant")

    return graph.compile()


def invoke_graph_with_steps(
    graph,
    initial_state: dict,
    on_step: StepCallback,
    recursion_limit: int,
) -> dict:
    """Invoke the graph with step emission via a module-level callback.

    This sets the module-level _step_callback so nodes can emit events,
    runs the graph, then clears the callback.
    """
    global _step_callback
    _step_callback = on_step
    try:
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )
        return result
    finally:
        _step_callback = None


# Weights for constraint coverage scoring. Higher = more discriminative tool.
_TOOL_WEIGHTS: dict[str, float] = {
    "search_by_person": 2.0,
    "search_by_caption": 1.5,
    "search_by_location": 1.5,
    "search_by_person_count": 1.5,
    "search_by_time": 1.0,
}


def assemble_response(
    tool_results: dict[str, list[dict]],
    query: str = "",
    store: ChromaStore | None = None,
    embeddings: EmbeddingService | None = None,
) -> DualListSearchResponse:
    """Merge tool results into solid/soft result lists.

    - Solid: images returned by ALL tools (intersection).
    - Soft: images returned by ANY tool but not all (remainder).
    - If only one tool was called, all its results are solid.

    Scores use a composite formula:
      0.7 * constraint_coverage + 0.3 * caption_similarity
    caption_similarity comes from search_by_caption when called, otherwise from
    embedding the original query against each image's stored caption vector.
    This ensures results are always meaningfully differentiated rather than
    falling back to a flat 0.5 placeholder.
    """
    if not tool_results:
        return DualListSearchResponse(solid_results=[], soft_results=[])

    # Collect image IDs per tool, track scores from caption search
    tool_image_sets: list[set[str]] = []
    caption_scores: dict[str, float] = {}
    all_image_ids: set[str] = set()
    tool_names_used: dict[str, set[str]] = {}  # image_id -> set of base tool names

    # Map each call key to its weight, e.g. "search_by_person#0" -> 2.0
    call_key_weights: dict[str, float] = {
        tool_name: _TOOL_WEIGHTS.get(tool_name.split("#")[0], 1.0)
        for tool_name in tool_results
    }
    total_weight = sum(call_key_weights.values())

    for tool_name, results in tool_results.items():
        base_name = tool_name.split("#")[0]
        ids_in_tool: set[str] = set()
        for r in results:
            img_id = r.get("image_id", "")
            if not img_id:
                continue
            ids_in_tool.add(img_id)
            all_image_ids.add(img_id)
            # Store full call key so multi-person queries count each match separately.
            tool_names_used.setdefault(img_id, set()).add(tool_name)
            if base_name == "search_by_caption" and "score" in r:
                caption_scores[img_id] = r["score"]
        if ids_in_tool:
            tool_image_sets.append(ids_in_tool)

    if not tool_image_sets:
        return DualListSearchResponse(solid_results=[], soft_results=[])

    # Fill in query-derived caption scores for images not already scored by
    # search_by_caption. Replaces the flat 0.5 fallback with a real semantic
    # similarity so results are meaningfully ranked even for person/time/location
    # only queries (e.g. "trump and jack ma" with no scene description).
    if query and store is not None and embeddings is not None:
        uncovered = all_image_ids - set(caption_scores.keys())
        if uncovered:
            try:
                query_embedding = embeddings.embed_text(query)
                # Large top_k; ChromaStore.query_caption clamps to collection size.
                ids, distances = store.query_caption(query_embedding, top_k=10_000)
                for img_id, dist in zip(ids, distances):
                    if img_id in uncovered:
                        caption_scores[img_id] = round(1.0 - float(dist or 0.0), 4)
            except Exception:
                logger.warning("Query-derived caption scoring failed; using 0.5 fallback", exc_info=True)

    # Intersection = images found by ALL tools
    solid_ids = tool_image_sets[0]
    for s in tool_image_sets[1:]:
        solid_ids = solid_ids & s

    soft_ids = all_image_ids - solid_ids

    # Load records from DB
    all_needed = solid_ids | soft_ids
    with get_session() as session:
        records = {
            rec.image_id: rec
            for rec in session.query(ImageRecord).filter(
                ImageRecord.image_id.in_(list(all_needed))
            )
        }

        solid_items = _build_result_items(
            solid_ids, records, caption_scores, tool_names_used,
            call_key_weights=call_key_weights, total_weight=total_weight, is_solid=True,
        )
        soft_items = _build_result_items(
            soft_ids, records, caption_scores, tool_names_used,
            call_key_weights=call_key_weights, total_weight=total_weight, is_solid=False,
        )

    solid_items.sort(key=lambda x: x.score, reverse=True)
    soft_items.sort(key=lambda x: x.score, reverse=True)

    return DualListSearchResponse(solid_results=solid_items, soft_results=soft_items)


def _build_result_items(
    image_ids: set[str],
    records: dict[str, ImageRecord],
    caption_scores: dict[str, float],
    tool_names_used: dict[str, set[str]],
    call_key_weights: dict[str, float],
    total_weight: float,
    is_solid: bool,
) -> list[SearchResultItem]:
    """Build SearchResultItem list from image IDs.

    For solid results, score = caption similarity (all constraints matched).
    For soft results, score = 0.7 * constraint_coverage + 0.3 * caption_similarity,
    where constraint_coverage sums the weights of each matched call key individually,
    so two separate search_by_person calls each contribute their full weight.
    """
    items: list[SearchResultItem] = []

    # Friendly display names for tools
    tool_display = {
        "search_by_caption": "caption match",
        "search_by_person": "person match",
        "search_by_person_count": "person count match",
        "search_by_time": "time match",
        "search_by_location": "location match",
    }

    for img_id in image_ids:
        record = records.get(img_id)
        if record is None:
            continue

        caption_score = caption_scores.get(img_id, 0.5)
        # call_keys contains full keys like "search_by_person#0", "search_by_person#1"
        call_keys = tool_names_used.get(img_id, set())

        if is_solid or total_weight == 0:
            score = caption_score
        else:
            matched_weight = sum(call_key_weights.get(k, 1.0) for k in call_keys)
            constraint_coverage = matched_weight / total_weight
            score = round(0.7 * constraint_coverage + 0.3 * caption_score, 4)

        # Strip "#N" suffix for display purposes
        base_names_matched = {k.split("#")[0] for k in call_keys}
        friendly = [tool_display.get(t, t) for t in sorted(base_names_matched)]

        if is_solid:
            reason = f"Matched all criteria: {', '.join(friendly)}"
        else:
            reason = f"Partial match: {', '.join(friendly)}"

        explanation = MatchExplanation(
            image_id=record.image_id,
            reason=reason,
            matched_constraints=sorted(base_names_matched),
        )
        items.append(SearchResultItem(
            image_id=record.image_id,
            file_path=record.file_path,
            score=score,
            caption=record.caption,
            capture_timestamp=record.capture_timestamp,
            country=record.country,
            state=record.state,
            city=record.city,
            explanation=explanation,
        ))
    return items
