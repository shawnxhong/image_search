"""Search tools that the LLM agent can call."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import func, select

from image_search_app.config import settings
from image_search_app.db import ImageRecord, PersonRecord, get_session
from image_search_app.tools.time_parser import TimeParser
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Shared instances (lazy-initialized by the agent)
_time_parser = TimeParser()


# -- Tool definitions (JSON schema for the LLM) --

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_caption",
            "description": (
                "Semantic search over image captions. Use this when the query describes "
                "visual content like scenes, objects, or actions. Pass ONLY the descriptive "
                "words — strip out person names, dates, and location references."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive text to search for in captions.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return. Default 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_person",
            "description": (
                "Find images containing a specific person by name. "
                "Use this when the query mentions a person's name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name to search for.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_time",
            "description": (
                "Find images captured within a time period. "
                "Use this when the query mentions dates, months, years, "
                "or relative time expressions like 'last year', 'yesterday'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language time expression, e.g. 'last year', 'January 2025', 'yesterday'.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_location",
            "description": (
                "Find images taken at a specific location. "
                "Use this when the query mentions a physical place, city, state, or country."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name to search for (city, state, or country).",
                    },
                },
                "required": ["location"],
            },
        },
    },
]


# -- Tool implementations --


def search_by_caption(
    query: str,
    top_k: int = 10,
    store: ChromaStore | None = None,
    embeddings: EmbeddingService | None = None,
) -> list[dict]:
    """Semantic search over image captions.

    Returns all results with score >= solid_score_threshold, sorted by
    score descending.  Falls back to top_k as an upper bound on the
    number of ChromaDB results retrieved.
    """
    if store is None:
        store = ChromaStore()
    if embeddings is None:
        embeddings = EmbeddingService()

    threshold = settings.solid_score_threshold

    query_embedding = embeddings.embed_text(query)
    ids, distances = store.query_caption(query_embedding, top_k=top_k)

    results = []
    for img_id, dist in zip(ids, distances):
        score = 1.0 - float(dist or 0.0)
        if score < threshold:
            continue
        results.append({"image_id": img_id, "score": round(score, 4)})
    return results


def search_by_person(name: str) -> list[dict]:
    """DB lookup for images containing a named person.

    Supports partial name matching: "Colin" matches "Colin Powell",
    "Powell" matches "Colin Powell".  Returns [{image_id, person_name}].
    """
    search_term = name.strip().lower()
    if not search_term:
        return []

    with get_session() as session:
        rows = session.execute(
            select(PersonRecord.image_id, PersonRecord.name).where(
                func.lower(PersonRecord.name).contains(search_term),
                PersonRecord.dismissed.is_(False),
            )
        ).all()

    results = []
    seen = set()
    for image_id, person_name in rows:
        if image_id not in seen:
            results.append({"image_id": image_id, "person_name": person_name})
            seen.add(image_id)
    return results


def search_by_time(description: str) -> list[dict]:
    """Find images within a time range. Returns [{image_id, capture_timestamp}]."""
    time_range = _time_parser.parse(description)
    if time_range is None:
        return []

    with get_session() as session:
        rows = session.execute(
            select(ImageRecord.image_id, ImageRecord.capture_timestamp).where(
                ImageRecord.capture_timestamp.isnot(None),
                ImageRecord.capture_timestamp >= time_range.start,
                ImageRecord.capture_timestamp <= time_range.end,
            )
        ).all()

    results = []
    for image_id, ts in rows:
        results.append({
            "image_id": image_id,
            "capture_timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        })
    return results


def search_by_location(location: str = "") -> list[dict]:
    """Find images taken at a specific location.

    Searches country, state, and city fields (case-insensitive substring match).
    Returns [{image_id, country, state, city}].
    """
    search_term = location.strip().lower()
    if not search_term:
        return []

    from sqlalchemy import or_

    with get_session() as session:
        rows = session.execute(
            select(
                ImageRecord.image_id,
                ImageRecord.country,
                ImageRecord.state,
                ImageRecord.city,
            ).where(
                or_(
                    func.lower(ImageRecord.country).contains(search_term),
                    func.lower(ImageRecord.state).contains(search_term),
                    func.lower(ImageRecord.city).contains(search_term),
                )
            )
        ).all()

    return [
        {"image_id": img_id, "country": country, "state": state, "city": city}
        for img_id, country, state, city in rows
    ]


# -- Dispatcher --

# Map of tool name -> callable
TOOL_DISPATCH: dict[str, callable] = {
    "search_by_caption": search_by_caption,
    "search_by_person": search_by_person,
    "search_by_time": search_by_time,
    "search_by_location": search_by_location,
}


def execute_tool(
    name: str,
    arguments: dict,
    store: ChromaStore | None = None,
    embeddings: EmbeddingService | None = None,
) -> list[dict]:
    """Execute a tool by name with the given arguments."""
    if name not in TOOL_DISPATCH:
        logger.warning("Unknown tool: %s", name)
        return []

    if name == "search_by_caption":
        return search_by_caption(
            query=arguments.get("query", ""),
            top_k=arguments.get("top_k", 10),
            store=store,
            embeddings=embeddings,
        )
    elif name == "search_by_person":
        return search_by_person(name=arguments.get("name", ""))
    elif name == "search_by_time":
        return search_by_time(description=arguments.get("description", ""))
    elif name == "search_by_location":
        return search_by_location(location=arguments.get("location", ""))

    return []
