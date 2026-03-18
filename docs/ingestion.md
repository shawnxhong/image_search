# Ingestion Pipeline

**File:** `src/image_search_app/ingestion/pipeline.py` — `IngestionPipeline`

The ingestion pipeline processes a single image through EXIF extraction, geocoding, captioning, face detection, and embedding. It is designed to minimize SQLite lock time by running ML inference outside any DB session.

## Pipeline Diagram

```
Image file
    |
    v
Phase 0: Skip if already ingested (status = ready/pending_labels)
    |
    v
Phase 1: Quick DB upsert → get image_id, set status = "processing"
    |
    v
Phase 2: ML work (no DB session held)
    |
    ├── EXIF extraction → capture_timestamp, lat, lon
    ├── Reverse geocoding → country, state, city
    ├── VLM captioning → one-sentence description
    ├── Face detection → bounding boxes + 512-dim descriptors
    ├── Face matching → compare against known identities (candidates)
    ├── Caption embedding → 384-dim vector → ChromaDB
    └── Image embedding → 384-dim vector → ChromaDB
    |
    v
Phase 3: Short DB write → persist all results, set final status
    |
    ├── All faces labeled → status = "ready"
    └── Unlabeled faces → status = "pending_labels"
```

## Phase Details

### Phase 0: Skip Check

If the image file path already exists in the DB with status `ready` or `pending_labels`, skip re-ingestion and return the existing `image_id`.

### Phase 1: DB Upsert

Creates or updates an `ImageRecord` with `ingestion_status = "processing"`. Obtains the `image_id` for subsequent steps. This is a fast operation.

### Phase 2: ML Inference

All slow work happens here with **no DB session held**, preventing SQLite lock contention:

#### EXIF Extraction (`ingestion/exif.py`)

Reads EXIF metadata using PIL/Pillow:
- `capture_timestamp` — DateTimeOriginal or DateTimeDigitized
- `lat`, `lon` — GPS coordinates (converted from DMS to decimal degrees)
- `geo_confidence` — GPS accuracy if available

Returns an `ExifData` dataclass. Never raises on missing/corrupt EXIF.

#### Reverse Geocoding (`ingestion/geocode.py`)

If GPS coordinates are available, converts them to human-readable location:
- Uses **geopy Nominatim** (OpenStreetMap, no API key needed)
- Extracts `country`, `state`, `city` from response address
- Falls back to `town` or `village` if `city` is absent
- 5-second timeout, never raises (returns empty `GeoLocation` on error)

#### VLM Captioning (`ingestion/captioner.py`)

Generates a one-sentence description of the image:
- Uses Qwen2.5-VL-3B (INT4) via OpenVINO GenAI
- Prompt: "Describe this image in one sentence."
- Returns `CaptionResult(caption, confidence)`

#### Face Detection (`ingestion/faces.py`)

Detects faces and extracts identity descriptors:
1. **Detection** — face-detection-retail-0004 finds bounding boxes
2. **Landmarks** — landmarks-regression-retail-0009 locates facial landmarks for alignment
3. **ReID** — face-reidentification-retail-0095 extracts 512-dim face descriptors
4. **Matching** — Each descriptor is compared against ChromaDB `face_identities` collection. Top-3 candidates within configurable distance threshold (`face_identity_threshold`, default 0.5) are stored (never auto-assigned).

Returns `[FaceResult(face_id, bbox, confidence, descriptor, candidates)]`.

#### Embedding

- **Caption embedding** — `EmbeddingService.embed_text(caption)` → 384-dim vector
- **Image embedding** — `EmbeddingService.embed_image(path)` → 384-dim vector
- Both upserted to ChromaDB (idempotent)

### Phase 3: DB Persist

A single short DB session writes all results:
- Clears old `PersonRecord` entries on re-ingest
- Updates `ImageRecord` with all extracted data
- Creates new `PersonRecord` for each detected face
- Sets `ingestion_status`:
  - `"ready"` — All faces named or no faces detected
  - `"pending_labels"` — Unlabeled faces need user attention
- On error: sets status to `"failed"`

## Face Labeling & Caption Refinement

After ingestion, if faces are detected but unlabeled (`pending_labels`), the user can:

1. **Name a face** — Assigns a name, updates `PersonRecord.name` and `source = "user_tag"`
2. **Dismiss a face** — Sets `PersonRecord.dismissed = True`

When all faces in an image are named or dismissed, **caption refinement** is triggered:

### `refine_after_labeling(image_id)`

1. Loads named (non-dismissed) people, sorted left-to-right by bbox position
2. Calls `Captioner.generate_with_names(image_path, names, original_caption=original_caption)` with a short, direct prompt:
   - Single person: `"The person in this photo is named {name}. Describe this photo in one English sentence using their exact name."`
   - Multiple people: `"The people in this photo are named {names}. Describe this photo in one English sentence using their exact names."`
3. Re-embeds the new caption → updates ChromaDB
4. Updates `ImageRecord.caption` and sets status to `"ready"`

This produces more specific, searchable captions like "Hank is standing in front of a bookshelf" instead of "A man is standing in front of a bookshelf".

**Design note:** The prompt is intentionally short and direct. Complex rewrite prompts (asking the VLM to rewrite an existing caption with name substitution) are ignored by the small Qwen2.5-VL INT4 model — it re-describes the image instead. The short prompt reliably includes names in the output for both English and non-ASCII names.

## Face Identity Storage

When a user names a face, the face descriptor is stored in ChromaDB's `face_identities` collection:
- **Key:** face_id
- **Embedding:** 512-dim face descriptor
- **Metadata:** `{"name": "Person Name"}`

Future ingestions match new faces against this collection to suggest candidates.

## Error Handling

- Phase 2 failures set `ingestion_status = "failed"` and re-raise
- ChromaDB upserts are idempotent — safe even if the DB write fails
- Geocoding never raises — returns empty `GeoLocation`
- EXIF extraction never raises — returns `None` fields
