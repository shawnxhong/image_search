# Database Schema

**File:** `src/image_search_app/db.py`

SQLite database with WAL journal mode for concurrent read/write access. SQLAlchemy 2.0 ORM with `DeclarativeBase`.

## Tables

### `images` (ImageRecord)

Stores metadata for each ingested image.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `image_id` | String (PK) | No | UUID, auto-generated |
| `file_path` | String (UNIQUE) | No | Absolute path to image file |
| `capture_timestamp` | DateTime(tz) | Yes | From EXIF DateTimeOriginal |
| `lat` | Float | Yes | GPS latitude from EXIF |
| `lon` | Float | Yes | GPS longitude from EXIF |
| `caption` | String | Yes | VLM-generated description |
| `caption_confidence` | Float | Yes | VLM confidence score |
| `face_confidence` | Float | Yes | Max face detection confidence |
| `geo_confidence` | Float | Yes | GPS accuracy from EXIF |
| `country` | String | Yes | From reverse geocoding |
| `state` | String | Yes | From reverse geocoding |
| `city` | String | Yes | From reverse geocoding |
| `ingestion_status` | String | No | Lifecycle state (see below) |
| `caption_indexed_at` | DateTime(tz) | Yes | When caption was embedded |
| `image_indexed_at` | DateTime(tz) | Yes | When image was embedded |
| `embedding_model_version` | String | Yes | e.g. "all-MiniLM-L6-v2" |

**Relationship:** `people` — one-to-many with `PersonRecord`, cascade delete.

### `people` (PersonRecord)

Stores detected faces and their labels for each image.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | Integer (PK) | No | Auto-increment |
| `image_id` | String (FK) | No | References `images.image_id` |
| `name` | String | Yes | Person name (null = unlabeled) |
| `face_id` | String | No | Unique face identifier (UUID) |
| `bbox` | String | No | Bounding box: "x_min,y_min,x_max,y_max" |
| `confidence` | Float | No | Face detection confidence |
| `source` | String | No | "auto" / "auto_matched" / "user_tag" |
| `dismissed` | Boolean | No | True if user dismissed this face |
| `descriptor` | Text | Yes | JSON-encoded 512-dim face embedding |
| `candidates` | Text | Yes | JSON: `[{"name": "...", "distance": 0.xx}]` |

**Relationship:** `image` — many-to-one back-reference to `ImageRecord`.

## Ingestion Status Lifecycle

```
received → processing → ready
                      → pending_labels → (user labels faces) → refining_caption → ready
                      → failed
```

| Status | Meaning |
|--------|---------|
| `received` | Initial state, not yet processed |
| `processing` | ML inference in progress |
| `ready` | Fully ingested, searchable |
| `pending_labels` | Faces detected but not all labeled |
| `refining_caption` | Caption being regenerated with person names |
| `failed` | Ingestion error occurred |

## SQLite Configuration

- **Journal mode:** WAL (Write-Ahead Logging) for concurrent reads during writes
- **Busy timeout:** 30 seconds (both at engine and PRAGMA level)
- **Default path:** `sqlite:///./image_search.db` (configurable via `IMG_SEARCH_SQLITE_URL`)

## Helper Functions

- `create_all()` — Create all tables from ORM metadata
- `get_session()` — Return a new `Session` (use as context manager)
- `upsert_image(file_path)` — Insert or return existing image record
- `list_images()` — Return all image records
