"""Tests for the library endpoint and paginated image listing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from image_search_app.db import Base, ImageRecord, get_session, list_images_paginated


# ---- Fixtures ----

@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    """Use a temporary SQLite database for each test."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setattr("image_search_app.config.settings.sqlite_url", db_url)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)

    monkeypatch.setattr("image_search_app.db.engine", engine)

    def patched_get_session():
        return Session(engine)

    monkeypatch.setattr("image_search_app.db.get_session", patched_get_session)
    return engine


IMG_UUIDS = [f"00000000-0000-0000-0000-00000000000{i}" for i in range(1, 8)]


@pytest.fixture
def seed_library(_use_tmp_db):
    """Seed DB with 7 images, some with timestamps, some without."""
    engine = _use_tmp_db
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.add(ImageRecord(
            image_id=IMG_UUIDS[0], file_path="/photos/a.jpg",
            capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
            ingestion_status="ready",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[1], file_path="/photos/b.jpg",
            capture_timestamp=datetime(2025, 3, 10, tzinfo=timezone.utc),
            ingestion_status="ready",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[2], file_path="/photos/c.jpg",
            capture_timestamp=datetime(2024, 12, 25, tzinfo=timezone.utc),
            ingestion_status="ready",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[3], file_path="/photos/d.jpg",
            capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),  # same date as a.jpg
            ingestion_status="ready",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[4], file_path="/photos/e.jpg",
            capture_timestamp=None,  # no timestamp
            ingestion_status="ready",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[5], file_path="/photos/f.jpg",
            capture_timestamp=None,
            ingestion_status="pending_labels",
        ))
        session.add(ImageRecord(
            image_id=IMG_UUIDS[6], file_path="/photos/g.jpg",
            capture_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ingestion_status="ready",
        ))
        session.commit()


# ---- list_images_paginated tests ----

class TestListImagesPaginated:

    def test_returns_all_sorted_by_date_desc(self, seed_library):
        images, total = list_images_paginated(limit=50)
        assert total == 7
        assert len(images) == 7

        # Timestamped images should come first, sorted DESC
        timestamped = [img for img in images if img.capture_timestamp is not None]
        timestamps = [img.capture_timestamp for img in timestamped]
        assert timestamps == sorted(timestamps, reverse=True)

        # No-timestamp images should come last
        no_ts = [img for img in images if img.capture_timestamp is None]
        assert len(no_ts) == 2
        assert images[-1].capture_timestamp is None
        assert images[-2].capture_timestamp is None

    def test_returns_total_count(self, seed_library):
        _, total = list_images_paginated(limit=2)
        assert total == 7

    def test_limit_works(self, seed_library):
        images, total = list_images_paginated(limit=3)
        assert len(images) == 3
        assert total == 7

    def test_cursor_pagination(self, seed_library):
        # Get first page
        page1, _ = list_images_paginated(limit=3)
        assert len(page1) == 3

        # Get second page using cursor
        cursor = page1[-1].image_id
        page2, total = list_images_paginated(limit=3, cursor=cursor)
        assert total == 7

        # No overlap between pages
        page1_ids = {img.image_id for img in page1}
        page2_ids = {img.image_id for img in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_cursor_pagination_exhausts(self, seed_library):
        # Get all pages
        all_ids = []
        cursor = None
        for _ in range(10):  # safety limit
            images, total = list_images_paginated(limit=3, cursor=cursor)
            if not images:
                break
            all_ids.extend(img.image_id for img in images)
            cursor = images[-1].image_id
        assert len(all_ids) == 7
        assert len(set(all_ids)) == 7  # no duplicates

    def test_empty_db(self, _use_tmp_db):
        images, total = list_images_paginated(limit=50)
        assert total == 0
        assert len(images) == 0

    def test_invalid_cursor_starts_from_beginning(self, seed_library):
        images, total = list_images_paginated(limit=50, cursor="nonexistent-id")
        assert total == 7
        assert len(images) == 7

    def test_cursor_at_null_timestamp_boundary(self, seed_library):
        """Cursor on the last timestamped image should return no-timestamp images next."""
        # Get enough to include the last timestamped item
        all_images, _ = list_images_paginated(limit=50)
        # Find the boundary: last image with timestamp
        last_ts_idx = max(
            i for i, img in enumerate(all_images) if img.capture_timestamp is not None
        )
        cursor = all_images[last_ts_idx].image_id
        remaining, _ = list_images_paginated(limit=50, cursor=cursor)
        # Should get the null-timestamp images
        for img in remaining:
            assert img.capture_timestamp is None
