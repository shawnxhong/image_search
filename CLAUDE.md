# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic image search app built with Python. Uses FastAPI for the API, SQLAlchemy/SQLite for metadata, ChromaDB for vector search, and LangGraph for agent orchestration. The frontend is a vanilla JS SPA served by FastAPI's static file handling.

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

- **api/** — FastAPI app (`main.py`). Mounts the web UI, exposes `/ingest`, `/search/text`, `/search/image`, `/health` endpoints.
- **agent/** — `SearchAgent` orchestrates search: parses intent, runs vector retrieval, applies hard filters, returns dual-list results (solid vs soft). `langgraph_flow.py` is the planned LangGraph StateGraph migration target.
- **ingestion/** — Pipeline for processing images: EXIF extraction, captioning, face detection. Currently stub implementations.
- **tools/** — `IntentParser` extracts structured intent from queries. `time_parser` handles natural-language time ranges. `filters` applies hard constraints against image metadata.
- **vector/** — ChromaDB wrappers. `ChromaStore` manages collections, `EmbeddingService` handles embeddings (stub), `RetrieverService` provides text and image semantic search.
- **db.py** — SQLAlchemy models (`ImageRecord`, `PersonRecord`) and SQLite engine. `get_session()` returns a session (use as context manager).
- **schemas.py** — Pydantic models for API request/response types. `DualListSearchResponse` is the core search response with `solid_results` and `soft_results`.
- **config.py** — `pydantic-settings` based config. All settings use `IMG_SEARCH_` env prefix and can be set via `.env` file.
- **web/** — Static frontend (vanilla HTML/JS/CSS), served at `/` by FastAPI.

## Key Patterns

- Search results use a **dual-list pattern**: `solid_results` (passed hard filters) vs `soft_results` (high semantic relevance but failed hard filters). Each result includes a `MatchExplanation`.
- Many ML components (captioner, embeddings, face detection) are **stubs** returning placeholder values — designed to be swapped with real implementations.
- Config uses `pydantic-settings` with `IMG_SEARCH_` env prefix (e.g., `IMG_SEARCH_SQLITE_URL`).
- The `SearchAgent` class is a plain Python orchestrator, not yet using LangGraph's StateGraph.
