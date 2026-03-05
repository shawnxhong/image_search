# Phase 0 Contracts and Frozen Decisions

This file records completed Phase 0 contract decisions from `MVP_PLAN.md`.

## Frozen decisions

- **Image ID**: canonical identifier is UUID v4 generated server-side.
- **Time normalization**: API/schema times are UTC ISO-8601 datetimes.
- **BBox format**: `[x_min, y_min, x_max, y_max]` in pixel coordinates.
- **People cardinality/nullability**:
  - `people` is a list.
  - each item requires `face_id`, `bbox`, `confidence`, `source`.
  - `name` is nullable and can be filled by later user tagging.

## Phase 0 contract coverage

- Typed request/response contracts:
  - upload/indexing request + response
  - text search request
  - image search request
  - dual-list search response with explanation payload
- Config-backed runtime paths and model names.
- No-op search contract: empty index/database returns structured empty response:

```json
{
  "solid_results": [],
  "soft_results": []
}
```
