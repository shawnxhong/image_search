# Architecture Overview

## System Summary

Agentic image search app for personal photo libraries. All AI inference runs locally via OpenVINO on CPU/GPU. A LangGraph ReAct agent (powered by Qwen3-4B) orchestrates search tools. The frontend is a React SPA served by FastAPI.

## High-Level Diagram

```
                        React SPA (Vite)
                             |
                      FastAPI (uvicorn)
                      /      |       \
              Search API   Ingest API  Model API
                 |            |           |
          LangGraph Agent   Pipeline    Load/Unload
          (Qwen3-4B LLM)      |
           /   |   \          |
     caption person time    EXIF → Geocode → Caption → Faces → Embed
     location                |
        |                  SQLite + ChromaDB
     ChromaDB + SQLite
```

## Project Structure

```
image_search/
├── src/image_search_app/           # Python backend
│   ├── api/main.py                 # FastAPI app + all endpoints
│   ├── agent/                      # LangGraph search orchestration
│   │   ├── graph.py                # SearchAgent wrapper class
│   │   └── langgraph_flow.py       # StateGraph, nodes, assembly
│   ├── ingestion/                  # Image processing pipeline
│   │   ├── pipeline.py             # IngestionPipeline orchestrator
│   │   ├── captioner.py            # Qwen2.5-VL captioning (OpenVINO)
│   │   ├── faces.py                # Face detection (OpenVINO)
│   │   ├── exif.py                 # EXIF metadata extraction
│   │   └── geocode.py              # Reverse geocoding (geopy)
│   ├── vector/                     # ChromaDB wrappers
│   │   ├── chroma_store.py         # Collection CRUD + queries
│   │   ├── embeddings.py           # all-MiniLM-L6-v2 (OpenVINO)
│   │   └── retrievers.py           # Semantic search interface
│   ├── tools/                      # Agent tools + utilities
│   │   ├── search_tools.py         # 4 search tool implementations
│   │   ├── llm.py                  # LLMService (Qwen3 wrapper)
│   │   ├── time_parser.py          # Natural language time parsing
│   │   ├── intent_parser.py        # Query intent extraction
│   │   └── filters.py              # Hard filter logic
│   ├── face_recognition/           # OpenVINO face recognition module
│   │   ├── face_recognition_ov.py  # Full pipeline wrapper
│   │   ├── face_detector.py        # Detection model
│   │   ├── face_identifier.py      # ReID model
│   │   ├── landmarks_detector.py   # Landmark model
│   │   └── faces_database.py       # Identity database
│   ├── db.py                       # SQLAlchemy models + engine
│   ├── schemas.py                  # Pydantic request/response types
│   └── config.py                   # pydantic-settings configuration
├── frontend/                       # React + TypeScript SPA
│   ├── src/
│   │   ├── App.tsx                 # Root component, tab routing
│   │   ├── api.ts                  # API client (fetch + SSE)
│   │   ├── types.ts                # TypeScript type definitions
│   │   └── components/             # UI components
│   ├── vite.config.ts              # Build + dev proxy config
│   └── package.json
├── tests/                          # pytest test suite
├── models/                         # OpenVINO model weights (gitignored)
├── docs/                           # This documentation
├── pyproject.toml                  # Python deps + project config
├── design.md                       # Product design spec
└── CLAUDE.md                       # Dev instructions
```

## Key Design Patterns

1. **Dual-List Results** — Search returns `solid_results` (matched ALL tools) and `soft_results` (matched SOME tools). Prevents false positives from single-dimension matches.

2. **ReAct Agent Loop** — LLM reads the query, decides which tools to call, reviews results, and can loop to call more. Stops when satisfied or hits iteration limit.

3. **Phase-Based Ingestion** — ML inference (slow) runs outside DB sessions to avoid holding SQLite locks. A short session at the end persists all results atomically.

4. **Lazy Model Loading** — Models are loaded on-demand via the Model Control Panel. Thread-safe load/unload with `gc.collect()` to reclaim memory.

5. **OpenVINO Everywhere** — All AI models (LLM, VLM, embeddings, face detection) use OpenVINO for optimized CPU/GPU inference. No PyTorch at runtime.

6. **Streaming Search** — Server-Sent Events stream agent steps (thinking, tool calls, results) to the frontend in real time.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript 5.9, Vite 7 |
| API | FastAPI, uvicorn |
| Agent | LangGraph StateGraph, Qwen3-4B (OpenVINO GenAI) |
| Vector DB | ChromaDB (cosine distance, persistent) |
| Relational DB | SQLite (WAL mode), SQLAlchemy 2.0 ORM |
| Captioning | Qwen2.5-VL-3B (OpenVINO GenAI, INT4) |
| Embeddings | all-MiniLM-L6-v2 (OpenVINO, 384-dim) |
| Face Detection | Intel OMZ retail models (OpenVINO) |
| Geocoding | geopy Nominatim |
| Testing | pytest (backend), Vitest (frontend) |
| Linting | ruff (Python), ESLint + typescript-eslint (TS) |
