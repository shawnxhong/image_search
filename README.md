# Agentic Image Search

Local-first image search for personal photo libraries. The backend ingests images, extracts metadata, captions scenes, detects faces, stores embeddings, and exposes a FastAPI API. Search is driven by a LangGraph ReAct loop that uses local OpenVINO models to combine caption, person, time, and location retrieval. A React frontend provides search, library browsing, ingestion, and model controls.

## What It Does

- Ingests local images into SQLite and ChromaDB.
- Extracts EXIF timestamps and GPS metadata.
- Reverse geocodes GPS data into country, state, and city.
- Generates captions with a local VLM.
- Detects faces, suggests identity matches, and lets users label or dismiss them.
- Refines captions after face labeling so named people become searchable.
- Supports text, image, and image+text search.
- Streams agent reasoning steps to the UI during text search.
- Loads and unloads local models on demand to manage memory.

## Stack

- Backend: FastAPI, SQLAlchemy, LangGraph, ChromaDB
- Frontend: React 19, TypeScript, Vite
- Models: OpenVINO / OpenVINO GenAI
- Storage: SQLite + persistent Chroma collections

## Architecture

```text
React SPA (Vite / frontend)
        |
     FastAPI
   /    |     \
Search Ingest Models
  |      |       |
Agent  Pipeline  Load/Unload
  |      |
  |   EXIF -> Geocode -> Caption -> Faces -> Embeddings
  |      |
  +---- SQLite + ChromaDB
```

Key backend areas:

- `src/image_search_app/api/`: FastAPI endpoints
- `src/image_search_app/agent/`: LangGraph search orchestration
- `src/image_search_app/ingestion/`: EXIF, geocoding, captioning, face pipeline
- `src/image_search_app/tools/`: search tools, time parsing, LLM wrapper
- `src/image_search_app/vector/`: embeddings and ChromaDB access
- `frontend/`: React SPA
- `docs/`: detailed project documentation

## Requirements

- Python 3.10+
- Node.js for frontend development/builds
- Local OpenVINO-compatible model files under `models/`
- Enough CPU/GPU/RAM to load the selected models

This project is designed for local inference. No cloud model API is required.

## Quick Start

### 1. Install backend dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Configure models and settings

Defaults come from `src/image_search_app/config.py` and can be overridden in `.env` with the `IMG_SEARCH_` prefix.

Example:

```env
IMG_SEARCH_SQLITE_URL=sqlite:///./image_search.db
IMG_SEARCH_CHROMA_PATH=./.chroma
IMG_SEARCH_LLM_DEVICE=GPU
IMG_SEARCH_VLM_DEVICE=GPU
IMG_SEARCH_TEXT_EMBEDDING_DEVICE=GPU
IMG_SEARCH_DEFAULT_TOP_K=20
IMG_SEARCH_SOLID_SCORE_THRESHOLD=0.25
```

Default model locations:

- `models/Qwen3-4B-Instruct-ov`
- `models/Qwen2.5-VL-3B-Instruct/INT4`
- `models/all-MiniLM-L6-v2-ov`
- `models/intel/...` for face detection, landmarks, and re-identification

### 4. Run the backend

```bash
uvicorn image_search_app.api.main:app --reload
```

Backend URLs:

- App/API host: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

### 5. Run the frontend in development

```bash
cd frontend
npm run dev
```

Vite proxies API calls to the backend at `http://127.0.0.1:8000`.

### 6. Build the frontend for FastAPI serving

```bash
cd frontend
npm run build
cd ..
```

After the build, FastAPI serves `frontend/dist`.

## Typical Workflow

1. Start the backend.
2. Build the frontend or run the Vite dev server.
3. Load the models you need from the UI.
4. Ingest images from local paths.
5. Label or dismiss detected faces when needed.
6. Search by text, image, or both.

Recommended model presets:

- Search: LLM + embeddings
- Ingest: VLM + embeddings + face detection

## Search Behavior

The search agent decides which tools to call based on the query:

- `search_by_caption` for scene/content descriptions
- `search_by_person` for named people
- `search_by_time` for natural-language time filters
- `search_by_location` for geographic places derived from GPS metadata

Results are returned as:

- `solid_results`: matched all invoked search dimensions
- `soft_results`: matched only part of the query

Text search also supports streaming agent steps through SSE at `POST /search/text/stream`.

## Ingestion Behavior

Ingestion runs in phases to avoid holding SQLite locks during model inference:

1. Upsert the image and mark it as processing.
2. Run EXIF extraction, reverse geocoding, caption generation, face detection, and embedding outside the DB session.
3. Persist the final image, face, and vector data.

If faces are found but not labeled, the image remains in `pending_labels`. Once all faces are named or dismissed, the caption is regenerated with those names included and the image becomes fully searchable.

## Main API Endpoints

- `POST /ingest`
- `PUT /images/{image_id}/faces`
- `PUT /images/{image_id}/faces/{face_id}/dismiss`
- `GET /library`
- `POST /search/text`
- `POST /search/text/stream`
- `POST /search/image`
- `GET /llm/status`
- `GET /llm/available`
- `POST /llm/load`
- `POST /llm/unload`
- `GET /models/status`
- `POST /models/{name}/load`
- `POST /models/{name}/unload`
- `GET /health`
- `GET /image-preview`

## Development

Backend:

```bash
pytest
ruff check .
```

Frontend:

```bash
cd frontend
npm test
npm run lint
```

## Documentation Map

Detailed docs live in `docs/`:

- `docs/architecture.md`: system layout and design patterns
- `docs/backend.md`: LangGraph search pipeline
- `docs/ingestion.md`: ingestion phases and caption refinement
- `docs/api.md`: endpoint reference
- `docs/configuration.md`: all config settings
- `docs/models.md`: model inventory and lifecycle
- `docs/frontend.md`: React UI structure
- `docs/database.md`: persistence model
- `docs/vector_store.md`: ChromaDB usage
- `docs/product.md`: product framing
- `docs/ui.md`: UI notes