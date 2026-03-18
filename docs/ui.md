# UI Reference

React 19 SPA with three tabs: **Search**, **Library**, and **Ingest**. The Model Control Panel is always visible above the tabs.

## Page Layout

```
+------------------------------------------+
|  Header: "Agentic Image Search"          |
+------------------------------------------+
|  Model Control Panel                     |
|  [Search Mode] [Ingest Mode]             |
|  LLM:            [model ▾] [GPU ▾] Load  |
|  VL Captioner:                      Load  |
|  Embeddings:                        Load  |
|  Face Detection:                    Load  |
+------------------------------------------+
| [ Search ] [ Library ] [ Ingest ] ← tabs |
+------------------------------------------+
|                                          |
|  (active tab content below)              |
|                                          |
+------------------------------------------+
```

## Model Control Panel

Always visible regardless of active tab. Manages the lifecycle of 4 AI model groups.

### Model Rows

Each model row shows:
- **Status indicator** — green circle (loaded), gray circle (unloaded), spinning (loading/unloading)
- **Model name**
- **Status text** — "Loaded", "Not loaded", "Loading...", or "Unloading..."
- **Load / Unload buttons** — disabled when busy or already in target state

The **LLM row** has additional controls:
- **Model selector dropdown** — lists available LLM models found in the models directory
- **Device selector** — GPU or CPU
- When loaded, shows the active model name and device (e.g. "Qwen3-4B-Instruct-ov on GPU")

### Preset Buttons

Two buttons for batch model management:
- **Search Mode** — loads LLM + Embeddings; unloads VLM + Face Detection
- **Ingest Mode** — loads VLM + Embeddings + Face Detection; unloads LLM

The button matching the active tab is visually highlighted. All load/unload operations run in parallel.

### Context Hints

A hint line below the title shows tab-aware guidance:
- On Search tab: "Search needs LLM + Embeddings. Load: LLM. Can release: VL Captioner, Face Detection."
- On Ingest tab: "Ingestion needs VL Captioner + Embeddings + Face Detection. Load: VL Captioner. Can release: LLM."

Hints disappear when all required models are loaded and no unnecessary models are consuming memory.

## Search Tab

### Search Controls

```
+---------------------------------------------------+
|  Search                                           |
|                                                   |
|  Query mode: [Text ▾]    Max results: [20]        |
|                                                   |
|  Text query:                                      |
|  [find photos with tom on a beach last year    ]  |
|                                                   |
|  Image path (for image mode):                     |
|  [/path/to/query.jpg                           ]  |
|                                                   |
|  Thumbnail size: [Medium ▾]    [Run Search]       |
+---------------------------------------------------+
```

- **Query mode** — Text / Image / Image+Text. Text input is disabled in Image mode; image input is disabled in Text mode.
- **Max results per list** — caps both solid and soft result lists (1–100)
- **Thumbnail size** — Small / Medium / Large, controls image card dimensions
- **Run Search** — triggers the search. If required models aren't loaded, shows a warning banner with instructions instead of searching. The warning has a dismiss button.
- **Enter key** — submits the search from either text field

### Agent Activity Log

```
+---------------------------------------------------+
|  Agent Activity (4)                    [▼ Hide]   |
|                                                   |
|  ⚙ Analyzing query...                            |
|  ➤ Calling search_by_person                       |
|    {"name": "tom"}                                |
|  ✔ search_by_person returned 3 results            |
|  ➤ Calling search_by_caption                      |
|    {"query": "beach"}                             |
|  ✔ search_by_caption returned 5 results           |
|  ✅ Search complete                                |
+---------------------------------------------------+
```

- Shows real-time streaming steps from the LangGraph agent
- **Step types**: thinking (gear icon), tool_call (arrow + args), tool_result (checkmark + count), done (green check), error (red X)
- **Collapsible** — click header to toggle. Auto-expands when a new search starts.
- **Auto-scrolls** to follow new entries

### Results Sections

Two result grids appear after a search:

```
+---------------------------------------------------+
|  Solid Matches                                    |
|  +--------+  +--------+  +--------+              |
|  | thumb  |  | thumb  |  | thumb  |              |
|  | 82%    |  | 75%    |  | 68%    |              |
|  | path   |  | path   |  | path   |              |
|  +--------+  +--------+  +--------+              |
+---------------------------------------------------+
|  Soft Matches                                     |
|  +--------+  +--------+                           |
|  | thumb  |  | thumb  |                           |
|  | 61%    |  | 55%    |                           |
|  | path   |  | path   |                           |
|  +--------+  +--------+                           |
+---------------------------------------------------+
```

- **Solid Matches** — "Best Matches" section, always shown first
- **Soft Matches** — "Other Results" section, deduplicated against solid by image_id
- **Empty state** — "No solid matches found." / "No additional soft matches."
- Both sections only appear after the first search is performed

### Image Card (Search Variant)

Each result card shows:
- **Thumbnail** — loaded via `/image-preview?path=...`, lazy-loaded, size controlled by thumbnail setting
- **Score** — semantic similarity score
- **File path**
- **Match explanation** — reason text (e.g. "Matched all criteria: caption match, person match")

## Library Tab

Displays all ingested images sorted by date (most recent first). No models need to be loaded — this is a pure database view.

```
+---------------------------------------------------+
|  Library  42          Thumbnail size: [Medium ▾]  |
|                                                   |
|  +--------+  +--------+  +--------+  +--------+  |
|  | thumb  |  | thumb  |  | thumb  |  | thumb  |  |
|  | Jan 1  |  | Jun 15 |  | Jun 15 |  | Mar 10 |  |
|  | NYC,NY |  | LA, CA |  |        |  | Paris  |  |
|  | caption|  | caption|  | caption|  | caption|  |
|  | path   |  | path   |  | path   |  | path   |  |
|  +--------+  +--------+  +--------+  +--------+  |
|                                                   |
|        [Load More (4 of 42)]                      |
+---------------------------------------------------+
```

- **Header** — shows total image count and thumbnail size selector
- **Grid** — responsive grid of image cards (library variant)
- **Load More** — cursor-based pagination, loads 50 images at a time. Hidden when all images are loaded.
- **Empty state** — "No images in the library yet. Use the Ingest tab to add images."
- **Auto-loads** first page on tab mount

### Image Card (Library Variant)

Each library card shows:
- **Thumbnail** — same lazy-loaded preview as search cards
- **Date** — formatted capture date (e.g. "Jun 15, 2025"), or "No date"
- **Location** — city, state, country (e.g. "New York City, New York, United States"). Only shown when location data is available.
- **Caption** — VLM-generated description, or "No caption"
- **File path**

### Image Card Component

`ImageCard` is a **reusable component** with two rendering variants controlled by the `variant` prop:

| Variant | Used In | Shows |
|---------|---------|-------|
| `search` | Search results (solid/soft) | Score, explanation, file path |
| `library` | Library grid | Date, location, caption, file path |

Both variants share the same thumbnail rendering, card layout, and lazy loading behavior. The component is designed for future extension with additional variants.

## Ingest Tab

### File Selection

```
+---------------------------------------------------+
|  Ingest Images                                    |
|  Browse to select images, then start ingestion.   |
|                                                   |
|  [Browse Images]  [Clear All]                     |
|                                                   |
|  /photos/beach.jpg                           [×]  |
|  /photos/park.jpg                            [×]  |
|  /photos/dinner.jpg                          [×]  |
|                                                   |
|  [Start Ingestion (3 images)]                     |
+---------------------------------------------------+
```

- **Browse Images** — opens a native OS file dialog (via tkinter on the backend). Can be clicked multiple times to add more files. Deduplicates against already-selected paths.
- **File list** — each path has a remove button (×). "Clear All" removes everything.
- **Start Ingestion** — disabled while running or if no files selected. Shows model warning if required models aren't loaded.
- All images are ingested in parallel — each card updates independently as its result arrives.

### Ingestion Results

```
+---------------------------------------------------+
|  Ingestion Results  3/5                           |
|                                                   |
|  +-----------------------------------------------+
|  | [thumb]  /photos/beach.jpg        [Ready]     |
|  |                                                |
|  |  Caption: Alice standing on a sunny beach      |
|  |  Timestamp: 6/15/2025, 12:00:00 AM            |
|  |  GPS: 40.71280°N, 74.00600°W                  |
|  |                                                |
|  |  Faces detected (1)                            |
|  |  [face]  Alice Johnson         95% conf        |
|  +-----------------------------------------------+
|                                                   |
|  +-----------------------------------------------+
|  | [spinner]  /photos/park.jpg    [Processing...] |
|  +-----------------------------------------------+
|                                                   |
|  +-----------------------------------------------+
|  | [thumb]  /photos/dinner.jpg   [Needs Labels]  |
|  |                                                |
|  |  Caption: Two people having dinner             |
|  |  Timestamp: 12/25/2024, 12:00:00 AM           |
|  |  GPS: —                                        |
|  |                                                |
|  |  Faces detected (2) · 1 dismissed              |
|  |                                                |
|  |  [face]  [Colin P. 85%] [Alice J. 72%]        |
|  |          [Other...]                             |
|  |                                                |
|  |  [face]  [Name this person___]                 |
|  |          [✕ Not a face]                        |
|  |                                                |
|  |  [Save Names]                                  |
|  +-----------------------------------------------+
```

### Ingest Card States

| Status | Badge | Visual |
|--------|-------|--------|
| `pending` | Pending | Placeholder (no thumbnail) |
| `processing` | Processing... | Spinner animation |
| `ready` | Ready | Green badge, full metadata |
| `pending_labels` | Needs Labels | Yellow badge, face labeling section |
| `failed` | Failed | Red badge, error message |

### Card Content

When processing completes (ready or pending_labels), the card shows:

**Metadata section:**
- **Caption** — VLM-generated description
- **Timestamp** — formatted from EXIF DateTimeOriginal
- **GPS** — decimal degrees with N/S/E/W notation, or "—" if unavailable

**Faces section** (if faces detected):
- Header shows count of active faces and dismissed count
- Each face row contains:
  - **Face crop** — 48×48px CSS-cropped thumbnail from the original image using bbox coordinates
  - **Confidence** — detection confidence as percentage

### Face Labeling

For unlabeled faces, the card provides identity assignment:

**With candidates** (face matched known identities):
- **Candidate buttons** — clickable buttons showing each candidate name + similarity percentage. Clicking selects that name.
- **"Other..." button** — switches to free text input for a new name
- After picking a candidate, shows the selected name with a "change" link to re-pick

**Without candidates** (new face, no matches):
- **Text input** — "Name this person" placeholder
- If candidates exist, a "back to suggestions" link returns to the candidate buttons

**Dismiss button:**
- **"✕ Not a face"** — marks the face as dismissed (false positive)
- Shows "..." while the dismiss request is in-flight

**Save button:**
- **"Save Names"** — sends all entered names to the backend
- Only visible when there are unlabeled faces
- Shows "Saving..." then "Saved!" on success
- After saving, the backend may return an updated caption (refined with person names) and a new status — both are reflected in the card immediately

### Caption Refinement

After all faces in an image are named or dismissed, the backend automatically re-generates the caption with person names included. The card updates in place:
- Caption text changes (e.g. "Two people at the beach" → "Alice and Bob at the beach")
- Status changes from "Needs Labels" to "Ready"

This happens transparently — the user sees the updated caption after clicking "Save Names" or dismissing the last unlabeled face.
