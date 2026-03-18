# Vector Store (ChromaDB)

**File:** `src/image_search_app/vector/chroma_store.py` — `ChromaStore`

ChromaDB is used for semantic similarity search over embeddings. Persistent storage on disk.

## Collections

### caption_embeddings

Stores VLM-generated caption text as semantic vectors for content-based search.

| Field | Value |
|-------|-------|
| ID | `image_id` (UUID string) |
| Embedding | 384-dim float vector (all-MiniLM-L6-v2) |
| Document | Caption text (e.g. "Alice and Bob at the beach") |
| Distance | Cosine |

**Used by:** `search_by_caption` tool — queries this collection with an embedded query string.

### image_embeddings

Stores image-level embeddings.

| Field | Value |
|-------|-------|
| ID | `image_id` (UUID string) |
| Embedding | 384-dim float vector |
| Distance | Cosine |

**Used by:** Image similarity search endpoint.

### face_identities

Stores labeled face descriptors for identity matching during ingestion.

| Field | Value |
|-------|-------|
| ID | `face_id` (UUID string) |
| Embedding | 512-dim float vector (ReIDNet) |
| Metadata | `{"name": "Person Name"}` |
| Distance | Cosine |

**Used by:** Face matching during ingestion — new faces are compared against this collection to suggest identity candidates.

## Key Methods

### Caption Operations

```python
upsert_caption_embedding(image_id, embedding, caption)
# Upsert caption vector + text. Idempotent.

query_caption(embedding, top_k) -> (ids, distances)
# Semantic search. Returns image IDs + cosine distances.
```

### Image Operations

```python
upsert_image_embedding(image_id, embedding)
# Upsert image vector. Idempotent.

query_image(embedding, top_k) -> (ids, distances)
# Image similarity search.
```

### Face Operations

```python
upsert_face_identity(face_id, descriptor, name)
# Store a labeled face. Called when user names a face.

match_face(descriptor, threshold) -> (name, distance) | None
# Find closest match. Returns None if above threshold.

match_face_candidates(descriptor, top_k, threshold) -> [(name, distance), ...]
# Return top-K candidates within threshold, deduplicated by name.
```

## Scoring

ChromaDB returns **cosine distance** (0 = identical, 2 = opposite). The search pipeline converts to a **similarity score**:

```
score = 1.0 - cosine_distance
```

Score range: 0.0 (no similarity) to 1.0 (identical).

Results below `solid_score_threshold` (default 0.25) are filtered out before assembly.

## Storage

- **Path:** `./.chroma/` (configurable via `IMG_SEARCH_CHROMA_PATH`)
- **Persistence:** Automatic (ChromaDB persistent client)
- **Upserts are idempotent** — safe to re-ingest without duplicates

## Embedding Service

**File:** `src/image_search_app/vector/embeddings.py` — `EmbeddingService`

Wraps the all-MiniLM-L6-v2 model (OpenVINO) for text encoding:

1. Tokenize input text (HuggingFace tokenizer)
2. Run OpenVINO inference on CPU
3. Mean-pool over non-padding tokens
4. L2-normalize the output
5. Return 384-dim float list

**Auto-download:** If the OpenVINO model doesn't exist locally, exports from HuggingFace via `optimum-cli export openvino`.
