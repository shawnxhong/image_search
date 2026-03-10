# Frontend Implementation Plan (Confirmed UX)

## Goals
- Build a single-page frontend for the search workflow.
- Render results in **two vertical sections** in this strict order:
  1. Solid Matches
  2. Soft Matches
- Ensure soft list is interpreted as semantic results minus solid results.
- Keep explanation UI as natural-language sentence only.
- Make default thumbnail size and max results per list configurable.

## Information Architecture
1. Header
   - Product title
   - Lightweight description
2. Search controls card
   - Query mode selector: text / image / image+text
   - Text query input
   - Image path input
   - Config controls:
     - Thumbnail size (small/medium/large)
     - Max results per list (default 20)
   - Search action button
3. Results area
   - Solid Matches section (shown first)
   - Soft Matches section (shown second)
   - Empty states and count badges

## Frontend Contracts
- `POST /search/text` with `{query, top_k}`
- `POST /search/image` with `{image_path, query?, top_k}`
- Response shape: `{solid_results: [], soft_results: []}`

## Implementation Tasks
1. Add static assets:
   - `web/index.html`
   - `web/static/styles.css`
   - `web/static/config.js`
   - `web/static/app.js`
2. Add FastAPI routes for frontend hosting:
   - `/` serves the SPA HTML
   - `/static/*` for static assets
   - `/image-preview?path=...` for thumbnail rendering from local files
3. UI behavior:
   - Disable irrelevant inputs based on selected mode
   - Call correct search endpoint
   - Render solid first, then soft
   - Render natural sentence explanation only
4. Config behavior:
   - Use defaults from `config.js`
   - Allow runtime overrides in UI controls

## Validation
- Load page in browser and verify section order.
- Perform text search and verify both sections render.
- Verify runtime config changes thumbnail size and per-list cap.
