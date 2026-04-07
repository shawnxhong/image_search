# Backend Search Pipeline

## Overview

The search pipeline is an **agentic ReAct loop** built on LangGraph. A local LLM (Qwen3-4B via OpenVINO) reads the user query, decides which search tools to invoke, collects results, and optionally loops to call more tools. Results are then assembled into dual solid/soft lists.

```
User query
    |
    v
[LangGraph ReAct loop]
    |
    +---> LLM decides tool(s) ---> execute tool(s) ---> LLM reviews ---> ...
    |
    v
assemble_response()  -->  DualListSearchResponse { solid_results, soft_results }
```

## Entry Points

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `POST /search/text` | `SearchAgent.search_text()` | Synchronous text search |
| `POST /search/text/stream` | `SearchAgent.search_text_stream()` | SSE streaming with agent steps |
| `POST /search/image` | `SearchAgent.search_image()` | Image search (delegates to text if query provided) |

**File:** `agent/graph.py` — `SearchAgent` wraps the compiled LangGraph.

## LangGraph Pipeline

**File:** `agent/langgraph_flow.py`

### Nodes

1. **`_assistant_node`** — Calls `LLMService.chat(messages, tools=TOOL_DEFINITIONS)`. Parses tool requests from the LLM response. If tool requests found, routes to tool node; otherwise ends.

2. **`_tool_node`** — Executes each requested tool via `execute_tool()`. Appends results as `{"role": "tool"}` messages back to the LLM context. Accumulates all results in `tool_results: dict[str, list[dict]]`.

### Graph Structure

```
entry --> assistant --> [has tool_requests?]
              ^              |
              |         YES: tool --> assistant (loop)
              |         NO:  END
              +-----------------------------+
```

**Recursion limit:** `llm_max_agent_iterations * 2 + 2` (default: 8 steps).

### System Prompt

```
You are a search assistant for a personal photo library.
Guidelines:
- Person by name → search_by_person
- Visual content / scene descriptions (library, beach, park) → search_by_caption
- Time period → search_by_time
- Geographic location (cities, states, countries from GPS) → search_by_location
  (NOT for scene descriptions — use search_by_caption for those)
- May call multiple tools in sequence
- When done, say DONE (no Action/Action Input tags)
```

The prompt includes few-shot examples clarifying scene vs location usage, e.g. "photos in a library" → `search_by_caption(query=library)`, not `search_by_location`.

## Query Preprocessing

### Scene vs Location Distinction

The system prompt and tool descriptions explicitly distinguish:
- **Scene descriptions** (library, beach, park, kitchen) → `search_by_caption` (captions describe what's in the photo)
- **Geographic locations** (New York, California, France) → `search_by_location` (GPS-derived metadata)

When `search_by_location` returns zero results, it returns a hint suggesting `search_by_caption` instead, enabling the agent to self-correct.

## Search Tools

**File:** `tools/search_tools.py`

### `search_by_caption(query, top_k=10)`

Semantic search over image captions stored in ChromaDB.

1. Embeds query text → `EmbeddingService.embed_text(query)` (all-MiniLM-L6-v2, 384-dim)
2. Queries ChromaDB caption collection (cosine distance) → `ChromaStore.query_caption(embedding, top_k)`
3. Score = `1.0 - cosine_distance` (range 0–1)
4. **Filters out** results with score < `settings.solid_score_threshold` (default 0.25)
5. Returns only above-threshold results as `[{image_id, score}]`

This prevents low-relevance results from entering the assembly stage, where they could be falsely promoted to solid matches by intersection with other tools.

### `search_by_person(name)`

Database lookup for images containing a named person.

1. Queries `PersonRecord` where `LOWER(name) LIKE '%search_term%'` (substring match) and `dismissed = False`
2. Deduplicates by `image_id`
3. Returns `[{image_id, person_name}]`

Supports partial name matching: "Colin" matches "Colin Powell", "Powell" matches "Colin Powell". Case-insensitive.

### `search_by_time(description)`

Parses natural-language time expression and queries images by capture timestamp.

1. `TimeParser.parse(description)` → `TimeRange(start, end)`
2. Queries `ImageRecord` where `capture_timestamp BETWEEN start AND end`
3. Returns `[{image_id, capture_timestamp}]`

**Supported expressions (file: `tools/time_parser.py`):**
- Relative: "yesterday", "today", "last week", "this week", "last month", "this month", "last year", "this year"
- Counted: "last N days/weeks/months"
- Year: "in 2025", "2024"
- Month: "January", "Feb", "march" (defaults to current year; if month > current month, uses previous year)

### `search_by_location(location)`

Filters images by geographic location (country, state, or city).

1. Searches `ImageRecord` columns `country`, `state`, `city` using case-insensitive substring matching (`LOWER(field) LIKE '%search_term%'`)
2. Matches against any of the three fields (OR logic)
3. Returns `[{image_id, country, state, city}]`

Location data is populated during ingestion via reverse geocoding (see Ingestion section).

## Result Assembly

**Function:** `assemble_response(tool_results)` in `langgraph_flow.py`

### Algorithm

1. **Collect** image IDs per tool into sets. Track caption scores separately.
2. **Solid IDs** = intersection of all tool result sets (images returned by ALL tools called).
3. **Soft IDs** = union minus solid (images returned by SOME but not all tools).
4. **Load** `ImageRecord` from SQLite for all IDs.
5. **Build** `SearchResultItem` for each:
   - Score: caption search score if available, else **0.5 fallback**
   - Explanation: "Matched all criteria: ..." (solid) or "Partial match: ..." (soft)
6. **Sort** both lists by score descending.

### Scoring

| Source | Score value | Notes |
|--------|-----------|-------|
| Caption search | `1.0 - cosine_distance` | 0–1, based on embedding similarity |
| Person search | 0.5 (fallback) | Binary match, no gradient |
| Time search | 0.5 (fallback) | Binary match, no gradient |
| Location search | 0.5 (fallback) | Binary match, no gradient |

### Example

Query: "photos of Alice at the beach last year"

LLM calls:
- `search_by_person("Alice")` → {img1, img3, img7}
- `search_by_caption("beach")` → {img2, img3, img5, img7, img9, ...}
- `search_by_time("last year")` → {img1, img3, img5}

Assembly:
- Solid (all 3 tools): {img3} (Alice + beach caption + last year)
- Soft (partial): {img1, img2, img5, img7, img9, ...}

## Response Schema

```
DualListSearchResponse:
  solid_results: list[SearchResultItem]
  soft_results:  list[SearchResultItem]

SearchResultItem:
  image_id:    UUID
  file_path:   str
  score:       float
  explanation: MatchExplanation

MatchExplanation:
  image_id:            UUID
  reason:              str        # e.g. "Matched all criteria: caption match, person match"
  matched_constraints: list[str]  # tool names that matched
  missing_metadata:    list[str]
```

## Data Flow Dependencies

```
EmbeddingService (all-MiniLM-L6-v2, OpenVINO, GPU by default)
    |
    v
ChromaStore (cosine distance collections)
    |-- caption_embeddings  (caption text → 384-dim vectors)
    |-- image_embeddings    (placeholder)
    +-- face_identities     (512-dim face descriptors, used in ingestion only)

SQLite (ImageRecord, PersonRecord)
    |-- capture_timestamp            (used by time tool)
    |-- country, state, city         (used by location tool, populated by reverse geocoding)
    +-- PersonRecord.name            (used by person tool, partial match)

LLMService (Qwen3-4B-Instruct, OpenVINO GenAI)
    +-- Agent reasoning + tool calling
```

## Resolved Issues

The following issues have been fixed:

1. **Caption search threshold** — `search_by_caption` now filters results below `solid_score_threshold`, preventing low-relevance results from entering assembly.
2. **False positive promotion** — Because low-score caption results are filtered before assembly, they can no longer be promoted to solid matches by intersection with other tools.
3. **Location search** — Replaced the GPS stub with real country/state/city filtering. Location data is populated at ingestion time via reverse geocoding (geopy Nominatim).
4. **Partial name matching** — `search_by_person` now supports substring matching (first name, last name, or partial input).

## Remaining Limitations

### No score weighting for non-caption tools

Person, time, and location matches all get a default score of 0.5, which is not meaningful for ranking.

## Ingestion: Reverse Geocoding

**File:** `ingestion/geocode.py`, called from `ingestion/pipeline.py`

During ingestion, if EXIF data contains GPS coordinates (lat/lon), the pipeline calls `reverse_geocode(lat, lon)` to resolve them to human-readable location fields:

1. Uses **geopy Nominatim** API (no API key required, 5s timeout)
2. Extracts `country`, `state`, and `city` from the response address
3. Falls back to `town` or `village` if `city` is not present
4. Never raises — returns empty `GeoLocation` on error
5. Results are persisted to `ImageRecord.country`, `ImageRecord.state`, `ImageRecord.city`

These fields are then used by `search_by_location` for geographic filtering.

## Configuration

**File:** `config.py` (pydantic-settings, `IMG_SEARCH_` env prefix)

| Setting | Default | Description |
|---------|---------|-------------|
| `solid_score_threshold` | 0.25 | Minimum caption similarity score to include in results |
| `default_top_k` | 20 | Default top-K for API requests |
| `llm_max_agent_iterations` | 3 | Max LLM reasoning loops |
| `embedding_dim` | 384 | Text embedding dimension |
| `face_identity_threshold` | 0.5 | Cosine distance for face matching (ingestion) |
