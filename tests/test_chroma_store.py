"""Tests for ChromaDB store (ingestion step 5+8 storage)."""

from pathlib import Path

from image_search_app.vector.chroma_store import ChromaStore

DIM = 384  # matches all-MiniLM-L6-v2


def _make_store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(
        path=str(tmp_path / "chroma"),
        caption_collection="test_captions",
        image_collection="test_images",
    )


def test_upsert_and_query_caption(tmp_path: Path):
    store = _make_store(tmp_path)
    embedding = [0.1] * DIM
    store.upsert_caption_embedding("img-001", embedding, "a dog on the beach")

    ids, distances = store.query_caption(embedding, top_k=5)
    assert "img-001" in ids
    assert len(distances) == len(ids)


def test_upsert_and_query_image(tmp_path: Path):
    store = _make_store(tmp_path)
    embedding = [0.2] * DIM
    store.upsert_image_embedding("img-002", embedding)

    ids, distances = store.query_image(embedding, top_k=5)
    assert "img-002" in ids


def test_query_empty_collection(tmp_path: Path):
    store = _make_store(tmp_path)
    ids, distances = store.query_caption([0.1] * DIM, top_k=5)
    assert ids == []
    assert distances == []


def test_upsert_is_idempotent(tmp_path: Path):
    store = _make_store(tmp_path)
    embedding = [0.3] * DIM
    store.upsert_caption_embedding("img-003", embedding, "first caption")
    store.upsert_caption_embedding("img-003", embedding, "updated caption")

    assert store.caption_collection.count() == 1


def test_multiple_images_retrieval_order(tmp_path: Path):
    store = _make_store(tmp_path)
    # Insert two embeddings: one close to query, one far
    close = [1.0] + [0.0] * (DIM - 1)
    far = [0.0] * (DIM - 1) + [1.0]
    store.upsert_caption_embedding("close", close, "close match")
    store.upsert_caption_embedding("far", far, "far match")

    query = [1.0] + [0.0] * (DIM - 1)  # same as 'close'
    ids, distances = store.query_caption(query, top_k=2)
    assert ids[0] == "close"
