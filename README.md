# Agentic Image Search (Starter Scaffold)

This repository contains a starter architecture for an agentic image search system built in Python.

## Proposed Repository Structure

```text
.
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
│       └── vector/
│           ├── chroma_store.py
│           ├── embeddings.py
│           └── retrievers.py
└── tests/
    ├── test_noop_search.py
    └── test_time_parser.py
```

## What is included

- SQLite schema and SQLAlchemy setup for image metadata and people tags.
- ChromaDB wrappers for caption/image collections.
- Time parser + intent parser + hard filter stubs.
- LangGraph orchestration skeleton with deterministic-tool boundaries.
- FastAPI skeleton endpoints for ingest/search.
- Phase 0 contract record in `PHASE0_CONTRACTS.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn image_search_app.api.main:app --reload
```

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
