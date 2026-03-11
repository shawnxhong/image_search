# Agentic Image Search MVP Plan

## 1) Validation of Updated Requirements

Your updated requirements are coherent and implementable for an MVP with clear behavior boundaries.

### Product Behavior (Confirmed)
- Return **two result lists** for each query:
  1. **Solid matches**: pass hard filters first (people/time/location when explicitly required), then rank by semantic relevance.
  2. **Soft matches**: pure semantic relevance list, excluding already-returned solid matches when needed.
- Return **match explanations** for each result so users can understand why each image was included.
- Allow for missing metadata (e.g., missing EXIF timestamp/GPS) so strong semantic candidates can still appear in the soft list.

### Data Model (Confirmed)
Store in SQLite (source of truth):
- `image_id`
- `file_path`
- `capture_timestamp` (from EXIF)
- `lat`, `lon` (from EXIF)
- `caption`
- `caption_embedding` (optional in SQLite; see note below)
- `image_embedding` (optional in SQLite; see note below)
- `people` list entries with `{name, face_id, bbox, confidence, source}`
- `caption_confidence`
- `face_confidence`
- `geo_confidence`
- `ingestion_status`

### Embeddings in SQLite? (Recommendation)
With Chroma as the retrieval index, storing full vectors again in SQLite is **not required** for MVP. Prefer:
- Store vectors in Chroma collections only.
- Store only linkage and integrity metadata in SQLite (e.g., `image_id`, embedding model/version, vector status, last indexed timestamp).

This avoids duplication, reduces SQLite size, and keeps indexing concerns in Chroma. You can add vector snapshots to SQLite later only if you need backup/reindex portability.

### Retrieval Strategy (Confirmed)
- Text query: parse constraints -> hard filter route + semantic route -> produce two lists.
- Image query (with/without text): image-vector semantic route + optional text route + hard filters from parsed text -> produce two lists.

### Location/Time Handling (Confirmed)
- Natural-language time parser tool converts user time phrases into strict `[start, end]` ranges for filtering.
- Location hard filtering uses EXIF GPS only.
- “Beach/sunset/crowd” relevance comes from query-caption semantic matching (not separate scene fields).

## 2) Practical MVP Phases (Detailed)

## Phase 0 — Project Skeleton & Contracts (1–2 days)
Goal: establish stable interfaces before building models and agent logic.

### Deliverables
- Python project scaffold with modules for:
  - ingestion pipeline
  - metadata storage
  - Chroma indexing/search
  - tool interfaces
  - LangGraph orchestration
- Typed request/response schemas for:
  - upload/indexing
  - text search
  - image search
  - result explanation payload
- Configuration system (`.env` + settings module) for model paths, Chroma paths, DB path.

### Key Decisions to Freeze
- Canonical `image_id` strategy (UUID v4).
- Time normalization format (UTC ISO-8601).
- Bounding box coordinate format (`[x_min, y_min, x_max, y_max]` in pixel space).
- People field cardinality and nullability.

### Exit Criteria
- A no-op search request can flow end-to-end through interfaces and return structured empty results.

## Phase 1 — Metadata DB + Ingestion Backbone (2–4 days)
Goal: reliably ingest files and persist normalized metadata into SQLite.

### Build
1. **SQLite schema + migration layer**
   - `images` table for core metadata and confidence fields.
   - `people` table keyed by `image_id` with per-face records.
   - `index_status` fields for caption/image embedding readiness and ingestion state transitions.

2. **Ingestion state machine**
   - `received -> exif_extracted -> captioned -> faces_detected -> embedded -> indexed -> ready`
   - failure states with retry metadata.

3. **EXIF extraction module**
   - robust parsing with graceful fallback when EXIF absent/corrupt.
   - `capture_timestamp`, `lat`, `lon`, `geo_confidence` population.

4. **VL caption generation module**
   - produce caption + confidence.

5. **Face workflow hook**
   - detect faces + assign `face_id` + store bbox/confidence.
   - support `source=auto` initially; user tagging updates name + `source=user_tag`.

### Exit Criteria
- Given a folder of images, DB rows are created with correct ingestion statuses and no crash on missing metadata.

## Phase 2 — Chroma Integration & Retrieval Foundations (2–4 days)
Goal: productionize dual-vector retrieval and keep SQLite as authority.

### Build
1. **Chroma collections**
   - `caption_embeddings`
   - `image_embeddings`

2. **Indexer services**
   - Upsert vectors keyed by `image_id`.
   - Persist index bookkeeping in SQLite (`indexed_at`, `model_version`, `index_ok`).

3. **Retriever services**
   - text->caption vector search top-K.
   - image->image vector search top-K.
   - optional text expansion for image query with extra prompt.

4. **Consistency checks**
   - periodic reconciliation: SQLite ready images vs Chroma indexed IDs.

### Exit Criteria
- Deterministic query to either collection returns stable IDs and scores; missing vectors are reported cleanly.

## Phase 3 — Hard Filters + Time Parser Tool (2–3 days)
Goal: deterministic filtering layer for solid matches.

### Build
1. **Query intent parser**
   - extract structured constraints from user input:
     - people names (e.g., Tom)
     - time expression (e.g., last year)
     - location constraints (GPS-presence or geofence if specified later)

2. **Natural-language time tool**
   - convert to hard UTC range with explicit assumptions (timezone, locale).
   - examples: “last year”, “in 2021”, “between Jan and Mar last year”.

3. **Hard filter engine (SQLite side)**
   - people filter: exact/normalized name matching.
   - time filter: timestamp in parsed range.
   - location filter: EXIF GPS presence/range checks.

4. **Filter explainability payload**
   - include pass/fail reasons per candidate.

### Exit Criteria
- For a fixed candidate set, filters are deterministic and reproducible with clear per-image reasons.

## Phase 4 — Two-List Ranking Policy (Solid + Soft) (2–3 days)
Goal: implement your exact product behavior.

### Build
1. **Candidate generation**
   - semantic candidates from relevant vector search path(s).

2. **Solid list construction**
   - apply hard filters first.
   - rank survivors by semantic score (and tie-break with confidence if needed).

3. **Soft list construction**
   - rank by pure semantic relevance.
   - include images excluded from solid due to missing/failed hard-filter metadata.
   - optional dedupe with solid list depending on API contract.

4. **Response format**
   - `solid_results[]`, `soft_results[]`, each with:
     - score
     - matched constraints
     - missing metadata notes
     - human-readable explanation.

### Exit Criteria
- Same query consistently returns two lists with non-overlapping semantics and transparent rationale.

## Phase 5 — LangGraph Agent Orchestration (3–5 days)
Goal: have agent choose tools and pathways without sacrificing determinism.

### Graph Nodes
1. **Input normalization node**
2. **Intent/planning node** (decides text/image/hybrid path and required tools)
3. **Semantic retrieval node(s)** (caption/image vector tools)
4. **Hard-filter node** (people/time/location)
5. **Assembler node** (build solid + soft)
6. **Explanation node**
7. **Output validator node** (schema + empty/fallback logic)

### Guardrails
- LLM only plans/explains; filtering/ranking rules remain deterministic code.
- Time tool and filters run as mandatory when constraints are explicit.

### Exit Criteria
- End-to-end graph execution works for text-only, image-only, and image+text queries.

## Phase 6 — Face Tagging UX Loop (2–4 days)
Goal: reduce manual overhead and improve people-filter precision.

### Build
- face clustering support (optional in MVP if time permits).
- user tag update endpoints.
- backfill names to matching `face_id` clusters.
- confidence-aware people matching (exact match + optional alias map).

### Exit Criteria
- User can tag unknown faces and immediately improve future person-constrained search quality.

## Phase 7 — Evaluation, Tuning, and Reliability (ongoing)
Goal: measurable quality and robust operation.

### Build
1. **Evaluation dataset**
   - balanced text/image/hybrid queries with expected outputs.

2. **Metrics**
   - solid-list constraint satisfaction rate
   - precision@K / recall@K
   - soft-list usefulness (manual or click proxy)
   - latency per stage/node

3. **Operational robustness**
   - retries for model calls
   - dead-letter queue for failed ingestion
   - reconciliation job for DB/index drift

4. **Policy tuning**
   - K values per retrieval path
   - tie-break and confidence thresholds
   - explanation wording for user trust

### Exit Criteria
- Quantified baseline and a reproducible path to improve retrieval quality.

## 3) Implementation Notes for Your Specific Questions

### A) Should hard filters always run first?
Yes for the **solid list** whenever constraints are explicit. Keep semantic retrieval as upstream candidate generation to avoid scanning all images.

### B) How to handle missing metadata?
- Missing metadata should typically exclude images from solid when that metadata is required.
- Those images remain eligible for soft list if semantically strong.

### C) Do we need country/city fields?
No for MVP based on your preference. EXIF `lat/lon` in DB is enough; resolve place names on-demand during query-time tools if needed.

## 4) MVP Scope Cut (if you want fastest path)
If you want a lean first release:
- Include Phases 0–5 only.
- Keep Phase 6 (face clustering) as post-MVP.
- Start Phase 7 with a small manual benchmark.

This still delivers your core behavior: dual-list returns, hard-filter correctness, semantic ranking, and agentic tool selection.


1. Ingestion pipeline (EXIF, captioning, face detection, embeddings) — the foundation; nothing works without data
2. Vector/retrieval layer — real embedding + ChromaDB search
3. Search tools (intent parser, time parser, hard filters) — query understanding
4. Search agent orchestration — wiring it all together
5. Frontend polish — UI refinements if needed

todo:
- 插入后的插入列表刷新
- face detection换杨粟的