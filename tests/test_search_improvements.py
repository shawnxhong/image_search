"""Tests for search pipeline improvements:
- Caption score threshold filtering
- Partial name matching
- Location-based search with geocoding
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from image_search_app.db import Base, ImageRecord, PersonRecord, create_all, get_session


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
    original_get_session = get_session.__wrapped__ if hasattr(get_session, "__wrapped__") else None

    def patched_get_session():
        return Session(engine)

    monkeypatch.setattr("image_search_app.db.get_session", patched_get_session)
    monkeypatch.setattr("image_search_app.tools.search_tools.get_session", patched_get_session)

    return engine


# Stable UUIDs for test images
IMG1_UUID = "00000000-0000-0000-0000-000000000001"
IMG2_UUID = "00000000-0000-0000-0000-000000000002"
IMG3_UUID = "00000000-0000-0000-0000-000000000003"


@pytest.fixture
def seed_images(_use_tmp_db):
    """Seed the DB with test images and people."""
    engine = _use_tmp_db
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        img1 = ImageRecord(
            image_id=IMG1_UUID, file_path="/photos/beach.jpg",
            capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
            lat=40.7128, lon=-74.0060,
            country="United States", state="New York", city="New York City",
            caption="A sunny day at the beach",
        )
        img2 = ImageRecord(
            image_id=IMG2_UUID, file_path="/photos/park.jpg",
            capture_timestamp=datetime(2025, 3, 10, tzinfo=timezone.utc),
            lat=34.0522, lon=-118.2437,
            country="United States", state="California", city="Los Angeles",
            caption="Two people walking in the park",
        )
        img3 = ImageRecord(
            image_id=IMG3_UUID, file_path="/photos/dinner.jpg",
            capture_timestamp=datetime(2024, 12, 25, tzinfo=timezone.utc),
            caption="Family dinner at Christmas",
        )
        session.add_all([img1, img2, img3])

        # People
        session.add(PersonRecord(
            image_id=IMG1_UUID, name="Colin Powell", face_id="f1",
            bbox="10,10,50,50", confidence=0.9, source="user_tag",
        ))
        session.add(PersonRecord(
            image_id=IMG2_UUID, name="Colin Powell", face_id="f2",
            bbox="20,20,60,60", confidence=0.85, source="user_tag",
        ))
        session.add(PersonRecord(
            image_id=IMG2_UUID, name="Alice Johnson", face_id="f3",
            bbox="70,20,110,60", confidence=0.9, source="user_tag",
        ))
        session.add(PersonRecord(
            image_id=IMG3_UUID, name="Bob Smith", face_id="f4",
            bbox="30,30,70,70", confidence=0.8, source="user_tag",
            dismissed=True,  # dismissed face
        ))

        session.commit()


# ---- Caption Threshold Tests ----

class TestCaptionThreshold:
    """Caption search should filter results below the score threshold."""

    def test_filters_below_threshold(self, seed_images, monkeypatch):
        """Results with score < threshold should be excluded."""
        monkeypatch.setattr("image_search_app.config.settings.solid_score_threshold", 0.5)

        from image_search_app.tools.search_tools import search_by_caption

        mock_store = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = [0.0] * 384

        # Simulate ChromaDB returning 3 results with varying distances
        # score = 1.0 - distance
        mock_store.query_caption.return_value = (
            [IMG1_UUID, IMG2_UUID, IMG3_UUID],
            [0.3, 0.6, 0.9],  # scores: 0.7, 0.4, 0.1
        )

        results = search_by_caption("beach", store=mock_store, embeddings=mock_embeddings)

        # Only img-001 (score=0.7) should pass the 0.5 threshold
        assert len(results) == 1
        assert results[0]["image_id"] == IMG1_UUID
        assert results[0]["score"] == 0.7

    def test_all_pass_when_above_threshold(self, seed_images, monkeypatch):
        """All results above threshold should be included."""
        monkeypatch.setattr("image_search_app.config.settings.solid_score_threshold", 0.25)

        from image_search_app.tools.search_tools import search_by_caption

        mock_store = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = [0.0] * 384
        mock_store.query_caption.return_value = (
            [IMG1_UUID, IMG2_UUID],
            [0.2, 0.4],  # scores: 0.8, 0.6
        )

        results = search_by_caption("beach", store=mock_store, embeddings=mock_embeddings)
        assert len(results) == 2

    def test_empty_when_all_below_threshold(self, seed_images, monkeypatch):
        """Should return empty list when all results are below threshold."""
        monkeypatch.setattr("image_search_app.config.settings.solid_score_threshold", 0.8)

        from image_search_app.tools.search_tools import search_by_caption

        mock_store = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = [0.0] * 384
        mock_store.query_caption.return_value = (
            [IMG1_UUID, IMG2_UUID],
            [0.5, 0.7],  # scores: 0.5, 0.3
        )

        results = search_by_caption("beach", store=mock_store, embeddings=mock_embeddings)
        assert len(results) == 0


# ---- Partial Name Matching Tests ----

class TestPartialNameMatching:
    """Person search should support partial name matching."""

    def test_full_name_match(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Colin Powell")
        assert len(results) == 2
        ids = {r["image_id"] for r in results}
        assert ids == {IMG1_UUID, IMG2_UUID}

    def test_first_name_match(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Colin")
        assert len(results) == 2
        ids = {r["image_id"] for r in results}
        assert ids == {IMG1_UUID, IMG2_UUID}

    def test_last_name_match(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Powell")
        assert len(results) == 2

    def test_case_insensitive(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("colin powell")
        assert len(results) == 2

    def test_partial_different_person(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Alice")
        assert len(results) == 1
        assert results[0]["image_id"] == IMG2_UUID
        assert results[0]["person_name"] == "Alice Johnson"

    def test_dismissed_faces_excluded(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Bob")
        assert len(results) == 0  # Bob is dismissed

    def test_empty_name_returns_nothing(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("")
        assert len(results) == 0

    def test_no_match_returns_empty(self, seed_images):
        from image_search_app.tools.search_tools import search_by_person
        results = search_by_person("Nonexistent Person")
        assert len(results) == 0


# ---- Location Search Tests ----

class TestLocationSearch:
    """Location search should filter by country/state/city."""

    def test_search_by_city(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("New York")
        assert len(results) == 1
        assert results[0]["image_id"] == IMG1_UUID

    def test_search_by_state(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("California")
        assert len(results) == 1
        assert results[0]["image_id"] == IMG2_UUID

    def test_search_by_country(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("United States")
        assert len(results) == 2
        ids = {r["image_id"] for r in results}
        assert ids == {IMG1_UUID, IMG2_UUID}

    def test_case_insensitive(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("new york")
        assert len(results) == 1

    def test_partial_match(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("Angeles")
        assert len(results) == 1
        assert results[0]["image_id"] == IMG2_UUID

    def test_no_match(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("Tokyo")
        # No actual image results — only a hint suggesting caption search
        image_results = [r for r in results if r.get("image_id")]
        assert len(image_results) == 0
        assert len(results) == 1
        assert "hint" in results[0]

    def test_empty_location_returns_nothing(self, seed_images):
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("")
        assert len(results) == 0

    def test_image_without_location_excluded(self, seed_images):
        """img-003 has no location data, should never appear."""
        from image_search_app.tools.search_tools import search_by_location
        results = search_by_location("United States")
        ids = {r["image_id"] for r in results}
        assert IMG3_UUID not in ids


# ---- Geocoding Tests ----

class TestReverseGeocode:
    """Reverse geocoding utility tests."""

    def test_returns_geolocation_on_success(self):
        from image_search_app.ingestion.geocode import reverse_geocode

        mock_location = MagicMock()
        mock_location.raw = {
            "address": {
                "country": "United States",
                "state": "New York",
                "city": "New York City",
            }
        }

        with patch("geopy.geocoders.Nominatim") as MockNom:
            MockNom.return_value.reverse.return_value = mock_location
            result = reverse_geocode(40.7128, -74.0060)

        assert result.country == "United States"
        assert result.state == "New York"
        assert result.city == "New York City"

    def test_returns_empty_on_no_result(self):
        from image_search_app.ingestion.geocode import reverse_geocode

        with patch("geopy.geocoders.Nominatim") as MockNom:
            MockNom.return_value.reverse.return_value = None
            result = reverse_geocode(0, 0)

        assert result.country is None
        assert result.state is None
        assert result.city is None

    def test_returns_empty_on_exception(self):
        from image_search_app.ingestion.geocode import reverse_geocode

        with patch("geopy.geocoders.Nominatim") as MockNom:
            MockNom.return_value.reverse.side_effect = Exception("Network error")
            result = reverse_geocode(40.7128, -74.0060)

        assert result.country is None

    def test_falls_back_to_town_or_village(self):
        from image_search_app.ingestion.geocode import reverse_geocode

        mock_location = MagicMock()
        mock_location.raw = {
            "address": {
                "country": "France",
                "state": "Provence",
                "village": "Gordes",
            }
        }

        with patch("geopy.geocoders.Nominatim") as MockNom:
            MockNom.return_value.reverse.return_value = mock_location
            result = reverse_geocode(43.9, 5.2)

        assert result.city == "Gordes"


# ---- Assembly with Threshold Tests ----

class TestAssemblyWithThreshold:
    """Verify that threshold filtering in caption search prevents false positive promotion."""

    def test_low_score_caption_not_promoted_to_solid(self, seed_images, monkeypatch):
        """A low-score caption match should not become solid just because
        it also matches a time/person filter."""
        monkeypatch.setattr("image_search_app.config.settings.solid_score_threshold", 0.5)

        from image_search_app.agent.langgraph_flow import assemble_response

        # Simulate: caption search found img-001 above threshold,
        # person search found img-001 and img-002
        tool_results = {
            "search_by_caption": [
                {"image_id": IMG1_UUID, "score": 0.7},
                # img-002 was filtered out by threshold (score was 0.3)
            ],
            "search_by_person": [
                {"image_id": IMG1_UUID, "person_name": "Colin Powell"},
                {"image_id": IMG2_UUID, "person_name": "Colin Powell"},
            ],
        }

        response = assemble_response(tool_results)

        solid_ids = {str(r.image_id) for r in response.solid_results}
        soft_ids = {str(r.image_id) for r in response.soft_results}

        # img-001 should be solid (both tools matched)
        assert IMG1_UUID in solid_ids, f"Expected {IMG1_UUID} in solid, got {solid_ids}"
        # img-002 should be soft (only person matched, caption was filtered)
        assert IMG2_UUID in soft_ids, f"Expected {IMG2_UUID} in soft, got {soft_ids}"
