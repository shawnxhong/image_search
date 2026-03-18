# API Reference

**File:** `src/image_search_app/api/main.py`

The FastAPI app serves both the REST API and the React SPA frontend.

## Search Endpoints

### `POST /search/text`

Non-streaming text search.

**Request:** `TextSearchRequest`
```json
{
  "query": "photos of Alice at the beach",
  "top_k": 20
}
```

**Response:** `DualListSearchResponse`
```json
{
  "solid_results": [
    {
      "image_id": "uuid",
      "file_path": "/path/to/image.jpg",
      "score": 0.82,
      "explanation": {
        "image_id": "uuid",
        "reason": "Matched all criteria: caption match, person match",
        "matched_constraints": ["search_by_caption", "search_by_person"],
        "missing_metadata": []
      }
    }
  ],
  "soft_results": [...]
}
```

### `POST /search/text/stream`

Streaming text search with Server-Sent Events. Returns agent steps in real time followed by the final result.

**Request:** Same as `/search/text`

**SSE Events:**
```
data: {"type": "thinking", "content": "Analyzing query..."}
data: {"type": "tool_call", "tool": "search_by_caption", "args": {"query": "beach"}}
data: {"type": "tool_result", "tool": "search_by_caption", "result": [...]}
data: {"type": "done", "content": "Search complete"}
data: {"type": "final_result", "data": { ...DualListSearchResponse... }}
```

### `POST /search/image`

Image-based search. Delegates to text search if a query is provided alongside the image.

**Request:** `ImageSearchRequest`
```json
{
  "image_path": "/path/to/query_image.jpg",
  "query": "optional text query",
  "top_k": 20
}
```

## Ingestion Endpoints

### `POST /ingest`

Ingest a single image through the full pipeline.

**Request:**
```json
{
  "image_path": "/absolute/path/to/image.jpg"
}
```

**Response:**
```json
{
  "image_id": "uuid",
  "status": "ready",
  "caption": "A sunny day at the beach",
  "faces": [
    {
      "face_id": "uuid",
      "bbox": [10, 20, 50, 60],
      "confidence": 0.95,
      "name": null,
      "dismissed": false,
      "candidates": [{"name": "Alice", "distance": 0.3}]
    }
  ]
}
```

### `PUT /images/{image_id}/faces`

Update face labels for an ingested image. Triggers caption refinement when all faces are named/dismissed.

**Request:**
```json
{
  "faces": [
    {"face_id": "uuid", "name": "Alice Johnson"}
  ]
}
```

### `PUT /images/{image_id}/faces/{face_id}/dismiss`

Mark a detected face as dismissed (not a real face or irrelevant).

### `GET /browse-images`

Opens a native file dialog (via tkinter) for selecting images. Returns selected file paths.

## Model Management Endpoints

### `GET /models/status`

Returns load status of all model services.

**Response:**
```json
{
  "llm": {"loaded": true, "model_name": "Qwen3-4B-Instruct-ov", "device": "GPU"},
  "vlm": {"loaded": false},
  "embeddings": {"loaded": true},
  "face_detection": {"loaded": false}
}
```

### `POST /models/{model_key}/load`

Load a specific model service. `model_key` is one of: `vlm`, `embeddings`, `face_detection`.

### `POST /models/{model_key}/unload`

Unload a specific model to free memory.

### `POST /llm/load`

Load the LLM (agent reasoning model). Accepts optional model name and device.

**Request:**
```json
{
  "model_name": "Qwen3-4B-Instruct-ov",
  "device": "GPU"
}
```

### `POST /llm/unload`

Unload the LLM.

### `GET /llm/status`

Get LLM load status.

### `GET /llm/models`

List available LLM models found in the models directory.

## Utility Endpoints

### `GET /health`

Health check. Returns `{"status": "ok"}`.

### `GET /image-preview`

Serve an image file for thumbnail preview.

**Query params:** `path=/absolute/path/to/image.jpg`

## Frontend Serving

- `GET /` — Serves `frontend/dist/index.html` (SPA fallback)
- `GET /assets/*` — Serves built static assets from `frontend/dist/assets/`
- Any unmatched route — Falls back to `index.html` for client-side routing

## Dev Proxy (Vite)

During development, Vite proxies these routes to `http://127.0.0.1:8000`:
- `/search`, `/ingest`, `/health`, `/image-preview`
- `/browse-images`, `/images`, `/llm`, `/models`
