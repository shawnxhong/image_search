"""LangGraph-based agentic search pipeline.

Implements a ReAct loop following the reference sample pattern:
  assistant node → (has tool request?) → tool node → assistant node → ...
"""

from __future__ import annotations

import json
import logging
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

# -- Query preprocessing: translate non-English queries for caption matching --

_TRANSLATE_PROMPT = """\
You are a query translator for an English-language photo search system. \
Translate the user's search query into English so it can match English captions.

Rules:
- Translate descriptive words, scene descriptions, time expressions, and \
location names into English.
- CRITICAL: Person names must be copied exactly as-is from the input. \
Do NOT transliterate, romanize, or convert names to pinyin. \
If the input has a Chinese name like 小丽, the output must also have 小丽, \
NOT Xiao Li or Xiaoli. This is because the database stores names in the \
original script entered by the user.
- If the query is already fully in English, return it unchanged.
- Return ONLY the translated query, nothing else. No explanation.

Examples:
- Input: Tom at the beach
  Output: Tom at the beach
- Input: 小丽在实验室
  Output: 小丽 in a lab
  WRONG: Xiao Li in a lab
- Input: 小丽和Tom在海边
  Output: 小丽 and Tom at the beach
  WRONG: Xiao Li and Tom at the beach
- Input: 去年在公园的照片
  Output: photos in the park last year
- Input: 张伟在纽约
  Output: 张伟 in New York
  WRONG: Zhang Wei in New York
- Input: 小丽在实验室的照片
  Output: 小丽 in a lab
  WRONG: Xiao Li in a lab\
"""


def _has_non_ascii(text: str) -> bool:
    """Check if text contains non-ASCII characters (likely non-English)."""
    return any(ord(c) > 127 for c in text)


def preprocess_query(query: str) -> str:
    """Translate a non-English query to English, preserving person names.

    If the query is already ASCII-only, returns it unchanged (no LLM call).
    Uses a lightweight LLM call without tool-calling mode.
    """
    if not _has_non_ascii(query):
        return query

    llm = get_llm_service()
    messages = [
        {"role": "system", "content": _TRANSLATE_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        response = llm.chat(messages=messages)
        translated = strip_thinking(response).strip()
        # Sanity check: if LLM returned empty or very long response, use original
        if not translated or len(translated) > len(query) * 3:
            logger.warning(
                "Query translation returned suspicious result, using original: %r -> %r",
                query, translated,
            )
            return query
        logger.info("Query translated: %r -> %r", query, translated)
        return translated
    except Exception:
        logger.exception("Query translation failed, using original query")
        return query


SYSTEM_PROMPT = """\
You are a search assistant for a personal photo library. Given a user query, \
decide which search tools to call to find relevant images.

Guidelines:
- If the query mentions a person by name, use search_by_person.
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

    for tool_req in tool_reqs:
        name = tool_req.get("name", "")
        args = tool_req.get("args", {})

        # Ensure args is a dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}

        logger.info("Executing tool: %s(%s)", name, args)
        results = execute_tool(name, args, store=store, embeddings=embeddings)
        # Count only actual image results (exclude hint-only dicts)
        image_result_count = sum(1 for r in results if r.get("image_id"))
        logger.info("Tool %s returned %d results", name, image_result_count)

        _emit(AgentStep(
            step_type="tool_result",
            tool_name=name,
            result_count=image_result_count,
            message=f"{name} returned {image_result_count} results",
        ))

        # Append tool result as a message for the LLM
        messages.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(results[:20]),  # Truncate to avoid huge context
        })

        # Track results for assembly
        if name in tool_results:
            tool_results[name] = tool_results[name] + results
        else:
            tool_results[name] = results

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
    graph.add_edge("tool", "assistant")
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


def assemble_response(tool_results: dict[str, list[dict]]) -> DualListSearchResponse:
    """Merge tool results into solid/soft result lists.

    - Solid: images returned by ALL tools (intersection).
    - Soft: images returned by ANY tool but not all (remainder).
    - If only one tool was called, all its results are solid.
    """
    if not tool_results:
        return DualListSearchResponse(solid_results=[], soft_results=[])

    # Collect image IDs per tool, track scores from caption search
    tool_image_sets: list[set[str]] = []
    caption_scores: dict[str, float] = {}
    all_image_ids: set[str] = set()
    tool_names_used: dict[str, set[str]] = {}  # image_id -> set of tool names

    for tool_name, results in tool_results.items():
        ids_in_tool: set[str] = set()
        for r in results:
            img_id = r.get("image_id", "")
            if not img_id:
                continue
            ids_in_tool.add(img_id)
            all_image_ids.add(img_id)
            tool_names_used.setdefault(img_id, set()).add(tool_name)
            if tool_name == "search_by_caption" and "score" in r:
                caption_scores[img_id] = r["score"]
        if ids_in_tool:
            tool_image_sets.append(ids_in_tool)

    if not tool_image_sets:
        return DualListSearchResponse(solid_results=[], soft_results=[])

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

        solid_items = _build_result_items(solid_ids, records, caption_scores, tool_names_used, is_solid=True)
        soft_items = _build_result_items(soft_ids, records, caption_scores, tool_names_used, is_solid=False)

    solid_items.sort(key=lambda x: x.score, reverse=True)
    soft_items.sort(key=lambda x: x.score, reverse=True)

    return DualListSearchResponse(solid_results=solid_items, soft_results=soft_items)


def _build_result_items(
    image_ids: set[str],
    records: dict[str, ImageRecord],
    caption_scores: dict[str, float],
    tool_names_used: dict[str, set[str]],
    is_solid: bool,
) -> list[SearchResultItem]:
    """Build SearchResultItem list from image IDs."""
    items: list[SearchResultItem] = []

    # Friendly display names for tools
    tool_display = {
        "search_by_caption": "caption match",
        "search_by_person": "person match",
        "search_by_time": "time match",
        "search_by_location": "location match",
    }

    for img_id in image_ids:
        record = records.get(img_id)
        if record is None:
            continue

        score = caption_scores.get(img_id, 0.5)
        tools_matched = tool_names_used.get(img_id, set())
        friendly = [tool_display.get(t, t) for t in sorted(tools_matched)]

        if is_solid:
            reason = f"Matched all criteria: {', '.join(friendly)}"
        else:
            reason = f"Partial match: {', '.join(friendly)}"

        explanation = MatchExplanation(
            image_id=record.image_id,
            reason=reason,
            matched_constraints=list(tools_matched),
        )
        items.append(SearchResultItem(
            image_id=record.image_id,
            file_path=record.file_path,
            score=score,
            explanation=explanation,
        ))
    return items
