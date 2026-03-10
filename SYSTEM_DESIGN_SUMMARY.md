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

### Final UX behavior decisions
- Solid and soft lists are **strictly non-overlapping**.
- Results are shown as **two vertical sections in order**: solid first, soft second.
- Explanations are shown as **natural-language sentences**.
- Thumbnail size and max results per list are configurable.

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

### Search flow
1. Parse query intent and detect mode.
2. Run semantic retrieval in appropriate Chroma collection(s).
3. Apply hard filters for explicit constraints.
4. Build solid list from hard-filter pass set.
5. Build soft list from semantic set minus solid.
6. Return explanations and scores.

---

## 3) Frontend Design Summary

The frontend is a lightweight SPA served by FastAPI static routes.

### Main screen structure
1. **Search controls panel**
   - Query mode selector (text / image / image+text)
   - Text query input
   - Image path input
   - Configurable UI controls:
     - thumbnail size (small/medium/large)
     - max results per list
2. **Results section: Solid Matches**
3. **Results section: Soft Matches**

### Rendering behavior
- Solid section always appears before soft section.
- Soft list is deduplicated against solid by `image_id`.
- Card UI includes:
  - preview thumbnail
  - score
  - file path
  - natural-language explanation
- Empty-state messaging is shown per section.

### Frontend configuration
Runtime config is centralized in `web/static/config.js` for:
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
