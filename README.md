# Agentic Image Search (Starter Scaffold)

This repository contains a starter architecture for an agentic image search system built in Python.

## Proposed Repository Structure

```text
.
├── MVP_PLAN.md
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
│       │   └── graph.py
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
    └── test_time_parser.py
```

## What is included

- SQLite schema and SQLAlchemy setup for image metadata and people tags.
- ChromaDB wrappers for caption/image collections.
- Time parser + intent parser + hard filter stubs.
- LangGraph orchestration skeleton with deterministic-tool boundaries.
- FastAPI skeleton endpoints for ingest/search.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn image_search_app.api.main:app --reload
```
