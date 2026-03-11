"""Tests for embedding service (ingestion steps 5+8)."""

import math
from pathlib import Path

from image_search_app.vector.embeddings import EmbeddingService


def test_embed_text_returns_list_of_floats():
    svc = EmbeddingService()
    result = svc.embed_text("a photo of a dog on the beach")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


def test_embed_text_dimension():
    """all-MiniLM-L6-v2 should produce 384-dim vectors."""
    svc = EmbeddingService()
    result = svc.embed_text("hello world")
    assert len(result) == 384


def test_embed_image_returns_list_of_floats(sample_image_no_exif: Path):
    svc = EmbeddingService()
    result = svc.embed_image(str(sample_image_no_exif))
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


def test_embed_image_dimension(sample_image_no_exif: Path):
    svc = EmbeddingService()
    result = svc.embed_image(str(sample_image_no_exif))
    assert len(result) == 384


def test_text_and_image_same_dimension(sample_image_no_exif: Path):
    """Text and image embeddings must be the same dimensionality."""
    svc = EmbeddingService()
    text_emb = svc.embed_text("a blue square")
    img_emb = svc.embed_image(str(sample_image_no_exif))
    assert len(text_emb) == len(img_emb)


def test_embeddings_are_not_zero():
    svc = EmbeddingService()
    result = svc.embed_text("a cat sitting on a mat")
    norm = math.sqrt(sum(v * v for v in result))
    assert norm > 0.0


def test_different_texts_produce_different_embeddings():
    svc = EmbeddingService()
    e1 = svc.embed_text("a sunny beach")
    e2 = svc.embed_text("a dark forest at night")
    assert e1 != e2
