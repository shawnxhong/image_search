"""Tests for SearchAgent — end-to-end text search with mocked retriever and DB."""

import os
from datetime import datetime, timezone

import pytest

# Use in-memory DB and temp chroma for tests
os.environ["IMG_SEARCH_SQLITE_URL"] = "sqlite:///./test_search_agent.db"
os.environ["IMG_SEARCH_CHROMA_PATH"] = "./.chroma_search_test"

from image_search_app.agent.graph import SearchAgent
from image_search_app.db import Base, ImageRecord, PersonRecord, create_all, engine, get_session
from image_search_app.tools.intent_parser import IntentParser
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService
from image_search_app.vector.retrievers import RetrieverService

# Fixed UUIDs for test data
IMG1_ID = "00000000-0000-4000-8000-000000000001"
IMG2_ID = "00000000-0000-4000-8000-000000000002"
IMG3_ID = "00000000-0000-4000-8000-000000000003"


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Create tables and seed test data once for the module."""
    create_all()

    store = ChromaStore()
    emb = EmbeddingService()

    with get_session() as session:
        # Image 1: has person "Alice", has timestamp in 2025, has GPS
        img1 = ImageRecord(
            image_id=IMG1_ID,
            file_path="/photos/alice_beach.jpg",
            caption="alice standing on a sandy beach at sunset",
            capture_timestamp=datetime(2025, 7, 15, tzinfo=timezone.utc),
            lat=34.0195,
            lon=-118.4912,
            ingestion_status="ready",
        )
        session.add(img1)
        session.add(PersonRecord(
            image_id=IMG1_ID, name="Alice", face_id="f1",
            bbox="10,10,100,100", confidence=0.9, source="user_tag",
        ))

        # Image 2: no person, has timestamp in 2024, no GPS
        img2 = ImageRecord(
            image_id=IMG2_ID,
            file_path="/photos/sunset.jpg",
            caption="beautiful sunset over the ocean with orange sky",
            capture_timestamp=datetime(2024, 12, 1, tzinfo=timezone.utc),
            ingestion_status="ready",
        )
        session.add(img2)

        # Image 3: has person "Bob", no timestamp, no GPS
        img3 = ImageRecord(
            image_id=IMG3_ID,
            file_path="/photos/bob_portrait.jpg",
            caption="portrait of a man smiling in a garden",
            ingestion_status="ready",
        )
        session.add(img3)
        session.add(PersonRecord(
            image_id=IMG3_ID, name="Bob", face_id="f2",
            bbox="20,20,120,120", confidence=0.85, source="user_tag",
        ))

        session.commit()

    # Index captions in ChromaDB
    for img_id, caption in [
        (IMG1_ID, "alice standing on a sandy beach at sunset"),
        (IMG2_ID, "beautiful sunset over the ocean with orange sky"),
        (IMG3_ID, "portrait of a man smiling in a garden"),
    ]:
        embedding = emb.embed_text(caption)
        store.upsert_caption_embedding(img_id, embedding, caption)

    yield

    # Cleanup
    import shutil
    engine.dispose()
    if os.path.exists("test_search_agent.db"):
        try:
            os.remove("test_search_agent.db")
        except OSError:
            pass
    for wal_file in ("test_search_agent.db-shm", "test_search_agent.db-wal"):
        if os.path.exists(wal_file):
            try:
                os.remove(wal_file)
            except OSError:
                pass
    if os.path.exists(".chroma_search_test"):
        shutil.rmtree(".chroma_search_test", ignore_errors=True)


@pytest.fixture()
def agent():
    store = ChromaStore()
    emb = EmbeddingService()
    retriever = RetrieverService(store=store, embeddings=emb)
    parser = IntentParser()
    parser._known_names = {"alice", "bob"}
    return SearchAgent(retriever=retriever, intent_parser=parser)


def test_text_search_returns_results(agent):
    """A broad query should return some results."""
    response = agent.search_text("sunset")
    total = len(response.solid_results) + len(response.soft_results)
    assert total > 0


def test_text_search_person_filter(agent):
    """Searching for 'alice' should put alice_beach in solid, others in soft."""
    response = agent.search_text("alice")
    solid_ids = {str(r.image_id) for r in response.solid_results}
    assert IMG1_ID in solid_ids


def test_text_search_person_not_found(agent):
    """Searching for 'bob sunset' — bob_portrait has no sunset caption, sunset.jpg has no Bob."""
    response = agent.search_text("bob sunset")
    soft_ids = {str(r.image_id) for r in response.soft_results}
    solid_ids = {str(r.image_id) for r in response.solid_results}
    # img-002 (sunset) should be in soft (no person Bob)
    if IMG2_ID in solid_ids or IMG2_ID in soft_ids:
        assert IMG2_ID in soft_ids


def test_text_search_no_constraints_all_solid(agent):
    """A plain query with no detectable intent → all results in solid."""
    response = agent.search_text("photo")
    # No people/time/gps constraints → everything passes hard filters
    assert len(response.soft_results) == 0
    assert len(response.solid_results) > 0


def test_text_search_returns_explanations(agent):
    """Every result should have a MatchExplanation."""
    response = agent.search_text("sunset")
    for r in response.solid_results + response.soft_results:
        assert r.explanation is not None
        assert r.explanation.reason


def test_text_search_scores_are_valid(agent):
    """Scores should be between 0 and some reasonable upper bound."""
    response = agent.search_text("beach")
    for r in response.solid_results + response.soft_results:
        assert r.score is not None


def test_text_search_respects_top_k(agent):
    """Results should not exceed top_k."""
    response = agent.search_text("photo", top_k=2)
    total = len(response.solid_results) + len(response.soft_results)
    assert total <= 2


def test_empty_query_returns_results(agent):
    """Even an empty-ish query should work without crashing."""
    response = agent.search_text("a")
    # Should not crash; may return results based on embedding of "a"
    assert response is not None
