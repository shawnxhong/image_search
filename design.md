# Agentic Image Search — Product, Backend, and Frontend Summary

## 1) Product Behavior

The app is designed to retrieve images from a personal corpus using agent-assisted query understanding across text and image inputs.

### Supported query modes
- **Text query** (e.g., “find photos with tom on a beach last year”)
- **Image query** (similar-image retrieval)
- **Image + text query** (image similarity plus additional constraints)

### Retrieval output contract
Every search returns **two ordered result lists**:
1. **Solid matches**
   - Candidates must pass explicit hard filters first (when constraints are present), then are ranked by semantic relevance.
2. **Soft matches**
   - Pure semantic candidates that do not appear in the solid list.
   - Intended to preserve highly relevant items when hard-filter metadata is missing.

---

## 2) Backend Design Summary

The backend is a FastAPI service with modular components for ingestion, storage, vector retrieval, deterministic filtering, and orchestration.

### Core architecture
- **API layer** (`api/main.py`)
  - Serves search/ingest endpoints and static frontend assets.
- **Metadata store** (SQLite via SQLAlchemy)
  - Source of truth for image metadata and person tags.
- **Vector store** (Chroma)
  - One collection for caption embeddings.
  - One collection for image embeddings.
- **Ingestion pipeline**
  - Extract EXIF data, generate caption, detect faces, embed caption/image, and index vectors.
- **Deterministic tools**
  - Intent parsing, time parsing, and hard filters for people/time/location.
- **Search orchestration**
  - Produces dual-list response (`solid_results`, `soft_results`) with explanations.

### Data model (high level)
Per image metadata includes:
- `image_id`, `file_path`
- EXIF fields: `capture_timestamp`, `lat`, `lon`
- `caption`
- People list entries (`name`, `face_id`, `bbox`, `confidence`, `source`)
- Confidence fields (`caption_confidence`, `face_confidence`, `geo_confidence`)
- `ingestion_status`
### Ingestion flow
1. parse file path
2. generate uuid as image_id 
3. extract timestamp from EXIF fields
4. extract GPS info from EXIF fields
5. generate caption using VL model, embed the caption and save to the chroma collection
6. generate face bbox using CV model. prompt the user to mark unknown faces with names.
7. generate confidence fields
8. generate image embedding and save to the chroma collection
9. mark ingestion_status with True/False depending on if the ingestion flow is successfully executed. 

### Search flow
1. Parse query intent and detect mode.
2. Run semantic retrieval in appropriate Chroma collection(s).
3. Apply hard filters for explicit constraints.
4. Build solid list from hard-filter pass set.
5. Build soft list from semantic set minus solid.
6. Return explanations and scores.

---

## 3) UI Design Summary

The UI has two top-level tabs: **Search** and **Ingest**.

### Tab: Search

#### Main screen structure
1. **Search controls panel**
   - Query mode selector (text / image / image+text)
   - Text query input
   - Image path input
   - Configurable UI controls:
     - thumbnail size (small/medium/large)
     - max results per list
2. **Results section: Solid Matches**
3. **Results section: Soft Matches**

#### Rendering behavior
- Solid section always appears before soft section.
- Soft list is deduplicated against solid by `image_id`.
- Card UI includes:
  - preview thumbnail
  - score
  - file path
  - natural-language explanation
- Empty-state messaging is shown per section.

### Tab: Ingest

#### Purpose
Allows users to submit a batch of image paths for ingestion and review/label detected faces.

#### Batch input
- A text area where the user enters one image path per line.
- A "Start Ingestion" button submits all paths.

#### Per-image ingestion card
Each submitted image is displayed as a card showing:
- Preview thumbnail of the image.
- File path.
- Ingestion status badge with lifecycle: `uploading` → `processing` → `ready` / `failed`.
- Status updates appear incrementally — each image is ingested independently and its card updates as soon as its result arrives (no waiting for the full batch).

#### Face labeling
- After ingestion completes for an image (`ready` status), if faces were detected, the card expands to show a **Faces** section.
- Each detected face shows:
  - A cropped thumbnail of the face region (derived from bbox).
  - A text input for the person's name (pre-filled if a name was already assigned).
  - Face confidence score.
- A "Save Names" button per card sends the updated names to the backend.
- Face labeling is **non-blocking** — ingestion completes regardless of whether the user labels faces. Users can label faces at their own pace after ingestion finishes.

#### Face labeling API contract
- `PUT /images/{image_id}/faces` accepts a list of `{face_id, name}` pairs.
- Backend updates the corresponding `PersonRecord` entries and sets `source` to `"user_tag"`.

### Frontend configuration
Runtime config is centralized in `frontend/src/config.ts` for:
- default thumbnail size
- default max results per list
- thumbnail pixel mapping by size


---

## 4) End-to-End Contract Alignment

The current product/backend/frontend contract alignment is:
- API returns dual-list search responses.
- Frontend enforces the requested presentation and non-overlap behavior.
- Backend architecture cleanly separates deterministic constraints from semantic retrieval.

This provides a practical Phase-0/early-Phase foundation that can be extended with production-grade models, richer intent parsing, and stronger ingestion/retrieval reliability in subsequent phases.