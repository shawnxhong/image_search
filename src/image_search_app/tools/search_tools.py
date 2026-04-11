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
            "name": "search_by_person_count",
            "description": (
                "Find images by the number of people detected in them. "
                "Use this when the query mentions a count of people: "
                "'solo', 'alone', 'just one person' → count=1; "
                "'couple', 'two people', 'pair' → count=2; "
                "'group', 'crowd', 'many people' → min_count=4; "
                "'no people', 'nobody', 'empty scene' → count=0. "
                "Do NOT combine with search_by_person unless a specific name is also mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Exact number of people required. Use for specific counts like 0, 1, 2, 3.",
                    },
                    "min_count": {
                        "type": "integer",
                        "description": "Minimum number of people. Use for 'group', 'crowd', 'many people'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_caption",
            "description": (
                "Semantic search over image captions. Use this when the query describes "
                "visual content like scenes, objects, actions, or settings/places that are NOT "
                "proper geographic names. For example, 'library', 'beach', 'park', 'kitchen' "
                "are scene descriptions and should use this tool. Pass ONLY the descriptive "
                "words -- strip out person names, dates, and geographic location names."
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
                "Find images taken at a specific geographic location by matching against "
                "GPS-derived city, state, or country names. ONLY use this for proper "
                "geographic names like 'New York', 'Japan', 'California'. Do NOT use this "
                "for scene descriptions like 'library', 'beach', 'park', 'kitchen' -- those "
                "should go to search_by_caption instead."
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


def search_by_person_count(
    count: int | None = None,
    min_count: int | None = None,
) -> list[dict]:
    """Find images by number of detected (non-dismissed) people.

    Supports exact count (count=N) and minimum count (min_count=N).
    count=0 returns images with no detected people at all.
    """
    with get_session() as session:
        if count == 0:
            # Images that have no PersonRecord rows at all
            subq = select(PersonRecord.image_id).where(
                PersonRecord.dismissed.is_(False)
            ).distinct()
            rows = session.execute(
                select(ImageRecord.image_id).where(
                    ImageRecord.image_id.not_in(subq)
                )
            ).all()
            return [{"image_id": row[0]} for row in rows]

        # Count non-dismissed faces per image
        count_col = func.count(PersonRecord.face_id).label("face_count")
        stmt = (
            select(PersonRecord.image_id, count_col)
            .where(PersonRecord.dismissed.is_(False))
            .group_by(PersonRecord.image_id)
        )
        if count is not None:
            stmt = stmt.having(count_col == count)
        elif min_count is not None:
            stmt = stmt.having(count_col >= min_count)

        rows = session.execute(stmt).all()
    return [{"image_id": row[0], "person_count": row[1]} for row in rows]


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

    If no results are found, returns a hint suggesting the agent try
    search_by_caption instead (the term may be a scene description
    rather than a geographic name).
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

    results = [
        {"image_id": img_id, "country": country, "state": state, "city": city}
        for img_id, country, state, city in rows
    ]

    if not results:
        logger.info(
            "search_by_location found no results for '%s' — may be a scene description",
            location,
        )
        return [{
            "hint": (
                f"No geographic location matching '{location}' was found in any "
                f"image's GPS-derived city/state/country. If '{location}' is a "
                f"scene description (e.g. a building, venue, or setting), try "
                f"search_by_caption(query=\"{location}\") instead."
            ),
        }]

    return results


# -- Dispatcher --

# Map of tool name -> callable
TOOL_DISPATCH: dict[str, callable] = {
    "search_by_person_count": search_by_person_count,
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

    if name == "search_by_person_count":
        return search_by_person_count(
            count=arguments.get("count"),
            min_count=arguments.get("min_count"),
        )
    elif name == "search_by_caption":
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
