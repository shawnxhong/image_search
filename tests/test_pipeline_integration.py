"""Integration test for the full ingestion pipeline (all steps end-to-end)."""

from pathlib import Path
from unittest.mock import patch

from image_search_app.db import Base, ImageRecord, PersonRecord, create_all, get_session
from image_search_app.ingestion.pipeline import IngestionPipeline


def _setup_db(tmp_path: Path):
    """Point the DB at a temp SQLite and create tables."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    with patch("image_search_app.db.engine") as mock_engine:
        from sqlalchemy import create_engine
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
    return db_url


def test_pipeline_ingest_returns_image_id(sample_image_no_exif: Path, tmp_path: Path):
    """Full pipeline should return a valid image_id string."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("image_search_app.config.settings.sqlite_url", db_url),
        patch("image_search_app.config.settings.chroma_path", chroma_path),
        patch("image_search_app.db.engine", _make_engine(db_url)),
    ):
        create_all()
        pipeline = IngestionPipeline()
        image_id = pipeline.ingest(str(sample_image_no_exif))

        assert isinstance(image_id, str)
        assert len(image_id) > 0


def test_pipeline_sets_ready_status(sample_image_no_exif: Path, tmp_path: Path):
    """After successful ingestion, status should be 'ready'."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("image_search_app.config.settings.sqlite_url", db_url),
        patch("image_search_app.config.settings.chroma_path", chroma_path),
        patch("image_search_app.db.engine", _make_engine(db_url)),
    ):
        create_all()
        pipeline = IngestionPipeline()
        image_id = pipeline.ingest(str(sample_image_no_exif))

        with get_session() as session:
            record = session.query(ImageRecord).filter_by(image_id=image_id).first()
            assert record is not None
            assert record.ingestion_status == "ready"


def test_pipeline_stores_caption(sample_image_no_exif: Path, tmp_path: Path):
    """Pipeline should generate and store a caption."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("image_search_app.config.settings.sqlite_url", db_url),
        patch("image_search_app.config.settings.chroma_path", chroma_path),
        patch("image_search_app.db.engine", _make_engine(db_url)),
    ):
        create_all()
        pipeline = IngestionPipeline()
        image_id = pipeline.ingest(str(sample_image_no_exif))

        with get_session() as session:
            record = session.query(ImageRecord).filter_by(image_id=image_id).first()
            assert record.caption is not None
            assert len(record.caption) > 0


def test_pipeline_nonexistent_file_raises(tmp_path: Path):
    """Pipeline should raise for a nonexistent file."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    chroma_path = str(tmp_path / "chroma")

    with (
        patch("image_search_app.config.settings.sqlite_url", db_url),
        patch("image_search_app.config.settings.chroma_path", chroma_path),
        patch("image_search_app.db.engine", _make_engine(db_url)),
    ):
        create_all()
        pipeline = IngestionPipeline()
        import pytest
        with pytest.raises(FileNotFoundError):
            pipeline.ingest("/nonexistent/image.jpg")


def _make_engine(db_url: str):
    from sqlalchemy import create_engine
    return create_engine(db_url)
