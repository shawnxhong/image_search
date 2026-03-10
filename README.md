# Agentic Image Search (Starter Scaffold)

This repository contains a starter architecture for an agentic image search system built in Python.

## Proposed Repository Structure

```text
.
├── FRONTEND_IMPLEMENTATION_PLAN.md
├── MVP_PLAN.md
├── PHASE0_CONTRACTS.md
├── pyproject.toml
├── README.md
├── src/
│   └── image_search_app/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── db.py
│       ├── api/
│       │   └── main.py
│       ├── agent/
│       │   ├── graph.py
│       │   └── langgraph_flow.py
│       ├── ingestion/
│       │   ├── exif.py
│       │   ├── captioner.py
│       │   ├── faces.py
│       │   └── pipeline.py
│       ├── tools/
│       │   ├── intent_parser.py
│       │   ├── time_parser.py
│       │   └── filters.py
│       ├── vector/
│       │   ├── chroma_store.py
│       │   ├── embeddings.py
│       │   └── retrievers.py
│       └── web/
│           ├── index.html
│           └── static/
│               ├── app.js
│               ├── config.js
│               └── styles.css
└── tests/
    ├── test_api_frontend_routes.py
    ├── test_noop_search.py
    └── test_time_parser.py
```

## What is included

- SQLite schema and SQLAlchemy setup for image metadata and people tags.
- ChromaDB wrappers for caption/image collections.
- Time parser + intent parser + hard filter stubs.
- LangGraph orchestration skeleton with deterministic-tool boundaries.
- FastAPI endpoints for ingest/search plus static frontend hosting.
- Phase 0 contract record in `PHASE0_CONTRACTS.md`.
- Frontend implementation plan in `FRONTEND_IMPLEMENTATION_PLAN.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn image_search_app.api.main:app --reload
```

Then open:
- `http://127.0.0.1:8000/` for the web UI
- `http://127.0.0.1:8000/docs` for API docs

## API examples

### Ingest

```json
POST /ingest
{
  "image_path": "./images/sample.jpg"
}
```

### Text search

```json
POST /search/text
{
  "query": "find photos with tom on a beach last year",
  "top_k": 20
}
```
