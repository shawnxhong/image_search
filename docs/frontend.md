# Frontend

React 19 + TypeScript 5.9 SPA, built with Vite 7. Tested with Vitest + React Testing Library.

## Directory Structure

```
frontend/
├── src/
│   ├── App.tsx                         # Root component, tab routing, model checks
│   ├── main.tsx                        # React DOM root
│   ├── api.ts                          # API client (fetch + SSE streaming)
│   ├── types.ts                        # Shared TypeScript types
│   ├── config.ts                       # Frontend config (thumbnail sizes)
│   ├── index.css                       # Global styles
│   ├── App.module.css
│   ├── components/
│   │   ├── TabNav.tsx                  # Search / Library / Ingest tab switcher
│   │   ├── SearchPanel.tsx             # Query input, mode selection, top-K
│   │   ├── AgentLog.tsx                # Real-time agent step display
│   │   ├── ResultsSection.tsx          # Solid/soft results grid
│   │   ├── ImageCard.tsx               # Reusable image card (search + library variants)
│   │   ├── LibraryPanel.tsx            # Browse all images, paginated
│   │   ├── IngestPanel.tsx             # Ingestion workflow
│   │   ├── IngestCard.tsx              # Single image card with face labeling
│   │   ├── LLMPanel.tsx                # Model load/unload controls
│   │   ├── ImageLightbox.tsx           # Full-size image overlay
│   │   └── *.module.css                # Component-scoped styles
│   └── test-setup.ts                   # Vitest setup (testing-library matchers)
├── vite.config.ts                      # Build config + dev proxy
├── package.json
├── tsconfig.json
└── dist/                               # Build output (served by FastAPI)
```

## Components

### App.tsx

Root component managing application state:
- **Tab routing** — Switches between Search, Library, and Ingest views via `TabNav`
- **Search state** — `solidResults`, `softResults`, `agentSteps`, `isLoading`
- **Model checks** — `checkSearchModels()` / `checkIngestModels()` validate model readiness before actions
- **Model status** — Tracks via `useRef<ModelStatusSnapshot>` updated by `LLMPanel.onStatusChange`

### TabNav.tsx

Three-tab navigation: **Search**, **Library**, and **Ingest**. Controls which panel is visible.

### LLMPanel.tsx

Model lifecycle control panel with:
- **Individual controls** — Load/unload buttons for each model (LLM, VLM, Embeddings, Face Detection)
- **LLM model selector** — Dropdown of available models + device (GPU/CPU)
- **Status indicators** — Loaded/unloaded state for each model
- **Preset buttons** — "Search Mode" (LLM + Embeddings) and "Ingest Mode" (VLM + Embeddings + Face Detection)
- **Status reporting** — Calls `onStatusChange(snapshot)` when any model status changes

Exports `ModelStatusSnapshot` interface for parent components.

### SearchPanel.tsx

Search input and controls:
- **Query modes** — Text, Image, Image+Text
- **Text input** — Free-text query field
- **Image path input** — For image-based search
- **Top-K control** — Number of results to retrieve
- **Thumbnail size** — Small / Medium / Large toggle
- **Model warning** — Banner shown if required models not loaded, with dismiss option

### AgentLog.tsx

Displays streaming agent steps during search:
- **Step types** — Thinking, tool_call, tool_result, done, error
- **Color-coded** — Different visual indicators per step type
- **Auto-scroll** — Follows new entries

### ResultsSection.tsx

Search results display:
- **Two sections** — "Best Matches" (solid) and "Other Results" (soft)
- **Grid layout** — Configurable thumbnail sizes
- **Image cards** — Click to preview full size

### ImageCard.tsx

Reusable image card component with two rendering variants via discriminated union props:

- **`variant: 'search'`** — Shows score, file path, and match explanation. Used by `ResultsSection`.
- **`variant: 'library'`** — Shows date, location (city/state/country), caption, and file path. Used by `LibraryPanel`.

Both variants share the same thumbnail rendering (`/image-preview?path=...`, lazy-loaded, configurable size). Clicking any thumbnail opens the `ImageLightbox` for full-size viewing. Designed for future extension with additional variants.

### ImageLightbox.tsx

Full-size image overlay modal:
- **Backdrop** — semi-transparent dark overlay, click to close
- **Image** — full-size rendering at up to 95vw/95vh
- **Close button** — red circular button with white X, positioned at the image's top-right corner
- **Escape key** — closes the lightbox
- **Body scroll lock** — prevents background scrolling while open

Used across all three tabs (Search, Library, Ingest) via `onImageClick` prop passed through components.

### LibraryPanel.tsx

Browse all ingested images:
- **Auto-loads** first page (50 images) on mount
- **Grid layout** — responsive grid with configurable thumbnail size
- **Cursor pagination** — "Load More" button appends the next page
- **Image cards** — library variant showing date, location, caption
- **No model requirement** — pure database view, works without any models loaded

### IngestPanel.tsx

Ingestion workflow:
- **File browser** — "Browse Images" button opens native OS dialog via `/browse-images`
- **Image list** — Shows selected images with ingestion status
- **Progress tracking** — Per-image status (pending → processing → ready/pending_labels)
- **Model warning** — Blocks ingestion if VLM/Embeddings/Face Detection not loaded

### IngestCard.tsx

Individual image ingestion card:
- **Image preview** — Thumbnail of the ingested image
- **Caption display** — Shows VLM-generated caption
- **Metadata** — Timestamp, location (country/state/city)
- **Face labeling** — For each detected face:
  - Face crop preview with bounding box
  - Candidate suggestions (from face matching)
  - Name input field
  - Dismiss button
- **Status** — Updates to "ready" after all faces handled

## API Client (api.ts)

All backend communication:

| Function | Endpoint | Description |
|----------|----------|-------------|
| `fetchLibrary(limit, cursor)` | `GET /library` | Paginated image listing |
| `search(request)` | `POST /search/text` | Non-streaming search |
| `searchTextStream(query, topK, onStep)` | `POST /search/text/stream` | SSE streaming search |
| `ingestImage(path)` | `POST /ingest` | Ingest single image |
| `updateFaces(imageId, faces)` | `PUT /images/{id}/faces` | Update face labels |
| `dismissFace(imageId, faceId)` | `PUT /images/{id}/faces/{faceId}/dismiss` | Dismiss a face |
| `fetchLLMStatus()` | `GET /llm/status` | LLM load state |
| `loadLLM(model, device)` | `POST /llm/load` | Load LLM |
| `unloadLLM()` | `POST /llm/unload` | Unload LLM |
| `fetchAllModelsStatus()` | `GET /models/status` | All model states |
| `loadModel(key)` | `POST /models/{key}/load` | Load model |
| `unloadModel(key)` | `POST /models/{key}/unload` | Unload model |
| `browseImages()` | `GET /browse-images` | Native file dialog |
| `fetchAvailableLLMs()` | `GET /llm/models` | List LLM models |

## Type Definitions (types.ts)

Key types:

```typescript
type QueryMode = 'text' | 'image' | 'image+text'
type IngestionStatus = 'pending' | 'processing' | 'ready' | 'pending_labels' | 'failed'
type AgentStepType = 'thinking' | 'tool_call' | 'tool_result' | 'done' | 'error'

interface SearchResultItem {
  image_id: string
  file_path: string
  score: number
  explanation: MatchExplanation
}

interface LibraryImageItem {
  image_id: string
  file_path: string
  caption: string | null
  capture_timestamp: string | null
  country: string | null
  state: string | null
  city: string | null
  ingestion_status: string
}

interface DetectedFace {
  face_id: string
  bbox: number[]
  confidence: number
  name: string | null
  dismissed: boolean
  candidates: FaceCandidate[]
}

interface AllModelsStatus {
  llm: LLMStatus
  vlm: ModelServiceStatus
  embeddings: ModelServiceStatus
  face_detection: ModelServiceStatus
}
```

## Styling

- **CSS Modules** — Component-scoped styles (`*.module.css`)
- **No CSS framework** — Custom styles throughout
- **Responsive grid** — Results layout adapts to thumbnail size setting

## Build & Dev

```bash
# Install dependencies
cd frontend && npm install

# Dev server (with API proxy to backend)
npm run dev

# Production build (output to dist/)
npm run build

# Run tests
npm test

# Lint
npm run lint
```

## Vite Dev Proxy

`vite.config.ts` proxies all API routes to the backend at `http://127.0.0.1:8000`:

```typescript
proxy: {
  '/library': 'http://127.0.0.1:8000',
  '/search': 'http://127.0.0.1:8000',
  '/ingest': 'http://127.0.0.1:8000',
  '/health': 'http://127.0.0.1:8000',
  '/image-preview': 'http://127.0.0.1:8000',
  '/browse-images': 'http://127.0.0.1:8000',
  '/images': 'http://127.0.0.1:8000',
  '/llm': 'http://127.0.0.1:8000',
  '/models': 'http://127.0.0.1:8000',
}
```
