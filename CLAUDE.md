# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic image search app for personal photo libraries. All AI inference runs locally via OpenVINO (CPU/GPU). Uses FastAPI for the API, SQLAlchemy/SQLite for metadata, ChromaDB for vector search, and a LangGraph ReAct agent (Qwen3-4B) for search orchestration. The frontend is a React 19 + TypeScript SPA built with Vite.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e .[dev]

# Run dev server
uvicorn image_search_app.api.main:app --reload

# Run all tests
pytest

# Run a single test
pytest tests/test_time_parser.py

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## Architecture

Source lives in `src/image_search_app/` with these layers:

- **api/** — FastAPI app (`main.py`). Mounts the React SPA, exposes `/ingest`, `/search/text`, `/search/text/stream` (SSE), `/search/image`, `/health`, `/models/*`, `/images/*` endpoints.
- **agent/** — `SearchAgent` (`graph.py`) wraps the compiled LangGraph StateGraph (`langgraph_flow.py`). The ReAct agent uses Qwen3-4B to decide which search tools to call, reviews results, and can loop.
- **ingestion/** — Full pipeline: EXIF extraction, reverse geocoding, VLM captioning (Qwen2.5-VL), face detection (Intel OMZ models), embedding. Supports face labeling and caption refinement with names.
- **tools/** — `search_tools.py` implements 4 search tools (caption, person, time, location). `llm.py` wraps the Qwen3 LLM. `time_parser.py` handles natural-language time ranges.
- **vector/** — ChromaDB wrappers. `ChromaStore` manages collections (caption, image, face identity). `EmbeddingService` uses all-MiniLM-L6-v2 via OpenVINO. `RetrieverService` provides semantic search.
- **face_recognition/** — OpenVINO face detection pipeline (detection → landmarks → ReID).
- **db.py** — SQLAlchemy models (`ImageRecord`, `PersonRecord`) and SQLite engine. `get_session()` returns a session (use as context manager).
- **schemas.py** — Pydantic models for API request/response types. `DualListSearchResponse` is the core search response with `solid_results` and `soft_results`.
- **config.py** — `pydantic-settings` based config. All settings use `IMG_SEARCH_` env prefix and can be set via `.env` file.

## Key Patterns

- Search results use a **dual-list pattern**: `solid_results` (passed hard filters) vs `soft_results` (high semantic relevance but failed hard filters). Each result includes a `MatchExplanation`.
- All ML models (LLM, VLM, embeddings, face detection) run locally via OpenVINO. Lazy-loaded on demand via the Model Control Panel.
- Config uses `pydantic-settings` with `IMG_SEARCH_` env prefix (e.g., `IMG_SEARCH_SQLITE_URL`).
- The search agent distinguishes **scene descriptions** (library, beach, park → `search_by_caption`) from **geographic locations** (New York, France → `search_by_location`).
- The app is English-only. All queries, captions, and person names are expected to be in English.
- Caption refinement after face labeling uses a short, direct VLM prompt to regenerate captions with person names included.
