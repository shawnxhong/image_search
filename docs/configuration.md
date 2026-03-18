# Configuration

**File:** `src/image_search_app/config.py`

Uses `pydantic-settings` with environment variable prefix `IMG_SEARCH_`. Settings can be set via `.env` file or environment variables.

## All Settings

### Database & Storage

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `sqlite_url` | `IMG_SEARCH_SQLITE_URL` | `sqlite:///./image_search.db` | SQLite connection URL |
| `chroma_path` | `IMG_SEARCH_CHROMA_PATH` | `./.chroma` | ChromaDB persistent storage directory |

### Search

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `default_top_k` | `IMG_SEARCH_DEFAULT_TOP_K` | `20` | Default number of results to retrieve |
| `solid_score_threshold` | `IMG_SEARCH_SOLID_SCORE_THRESHOLD` | `0.25` | Minimum caption similarity score (1 - cosine_distance) to include in results |
| `default_timezone` | `IMG_SEARCH_DEFAULT_TIMEZONE` | `UTC` | Timezone for time expression parsing |

### ChromaDB Collections

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `caption_collection` | `IMG_SEARCH_CAPTION_COLLECTION` | `caption_embeddings` | Collection name for caption vectors |
| `image_collection` | `IMG_SEARCH_IMAGE_COLLECTION` | `image_embeddings` | Collection name for image vectors |

### LLM (Agent)

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `llm_model_path` | `IMG_SEARCH_LLM_MODEL_PATH` | `models/Qwen3-4B-Instruct-ov` | Path to LLM model (relative to project root) |
| `llm_device` | `IMG_SEARCH_LLM_DEVICE` | `GPU` | Inference device (GPU / CPU) |
| `llm_max_agent_iterations` | `IMG_SEARCH_LLM_MAX_AGENT_ITERATIONS` | `3` | Maximum ReAct loop iterations |
| `llm_models_dir` | `IMG_SEARCH_LLM_MODELS_DIR` | `models` | Directory to scan for available LLM models |

### VLM (Captioner)

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `vlm_model_path` | `IMG_SEARCH_VLM_MODEL_PATH` | `models/Qwen2.5-VL-3B-Instruct/INT4` | Path to VLM model (relative to project root) |
| `vlm_device` | `IMG_SEARCH_VLM_DEVICE` | `GPU` | Inference device |

### Embeddings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `text_embedding_model_name` | `IMG_SEARCH_TEXT_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID (for auto-download) |
| `text_embedding_model_dir` | `IMG_SEARCH_TEXT_EMBEDDING_MODEL_DIR` | `models/all-MiniLM-L6-v2-ov` | Local OpenVINO model directory |
| `embedding_dim` | `IMG_SEARCH_EMBEDDING_DIM` | `384` | Embedding vector dimensionality |

### Face Detection

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `face_models_dir` | `IMG_SEARCH_FACE_MODELS_DIR` | `models` | Base directory for face detection models |
| `face_detection_model` | `IMG_SEARCH_FACE_DETECTION_MODEL` | `intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml` | Detection model path (relative to face_models_dir) |
| `face_landmarks_model` | `IMG_SEARCH_FACE_LANDMARKS_MODEL` | `intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml` | Landmarks model path |
| `face_reidentification_model` | `IMG_SEARCH_FACE_REIDENTIFICATION_MODEL` | `intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml` | ReID model path |
| `face_detection_confidence` | `IMG_SEARCH_FACE_DETECTION_CONFIDENCE` | `0.6` | Minimum face detection confidence |
| `face_identity_threshold` | `IMG_SEARCH_FACE_IDENTITY_THRESHOLD` | `0.5` | Cosine distance threshold for face matching |

## Example .env File

```env
IMG_SEARCH_SQLITE_URL=sqlite:///./image_search.db
IMG_SEARCH_LLM_DEVICE=GPU
IMG_SEARCH_VLM_DEVICE=GPU
IMG_SEARCH_SOLID_SCORE_THRESHOLD=0.25
IMG_SEARCH_DEFAULT_TOP_K=20
```

## Path Resolution

Model paths (`llm_model_path`, `vlm_model_path`, `text_embedding_model_dir`) are stored as **relative paths** in config. Each service resolves them to absolute paths at load time using the project root directory.

```python
# Example: in tools/llm.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
model_path = _PROJECT_ROOT / settings.llm_model_path
```
