"""Tests for caption refinement after face labeling."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from image_search_app.ingestion.captioner import CaptionResult, _format_names


# --- Unit tests for _format_names ---


class TestFormatNames:
    def test_empty_names(self):
        assert _format_names([]) == ""

    def test_single_name(self):
        assert _format_names(["Alice"]) == "Alice"

    def test_two_names(self):
        assert _format_names(["Alice", "Bob"]) == "Alice, Bob"

    def test_three_names(self):
        assert _format_names(["Alice", "Bob", "Eve"]) == "Alice, Bob, Eve"

    def test_uncommon_names(self):
        result = _format_names(["Doglashi", "Ximenez"])
        assert "Doglashi" in result
        assert "Ximenez" in result


# --- Integration tests for refine_after_labeling ---


def _unique_path(name: str = "photo") -> str:
    return f"/fake/{name}_{uuid.uuid4().hex[:8]}.jpg"


class TestRefineAfterLabeling:
    @patch("image_search_app.ingestion.pipeline.ChromaStore")
    @patch("image_search_app.ingestion.pipeline.EmbeddingService")
    @patch("image_search_app.ingestion.pipeline.Captioner")
    @patch("image_search_app.ingestion.pipeline.FaceRecognizer")
    def test_refine_updates_caption_in_db(
        self, mock_face_cls, mock_captioner_cls, mock_embed_cls, mock_store_cls,
    ):
        """After labeling, refine_after_labeling should update the DB caption."""
        from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
        from image_search_app.ingestion.pipeline import IngestionPipeline

        create_all()

        file_path = _unique_path("photo")

        with get_session() as session:
            record = ImageRecord(
                file_path=file_path,
                caption="a man and a woman standing together",
                ingestion_status="refining_caption",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            image_id = record.image_id

            session.add(PersonRecord(
                image_id=image_id, name="Bob", face_id="f1",
                bbox="10,20,100,200", confidence=0.9, source="user_tag",
            ))
            session.add(PersonRecord(
                image_id=image_id, name="Alice", face_id="f2",
                bbox="200,20,300,200", confidence=0.9, source="user_tag",
            ))
            session.commit()

        mock_captioner = mock_captioner_cls.return_value
        mock_captioner.generate_with_names.return_value = CaptionResult(
            caption="Bob and Alice standing together on a beach",
            confidence=0.85,
        )
        mock_embed = mock_embed_cls.return_value
        mock_embed.embed_text.return_value = [0.1] * 384
        mock_store = mock_store_cls.return_value

        pipeline = IngestionPipeline()
        new_caption = pipeline.refine_after_labeling(image_id)

        # Verify captioner was called with names sorted by bbox x_min and original caption
        mock_captioner.generate_with_names.assert_called_once_with(
            file_path, ["Bob", "Alice"],  # Bob x=10, Alice x=200
            original_caption="a man and a woman standing together",
        )

        assert new_caption == "Bob and Alice standing together on a beach"

        with get_session() as session:
            rec = session.query(ImageRecord).filter_by(image_id=image_id).first()
            assert rec.caption == new_caption
            assert rec.ingestion_status == "ready"

        mock_embed.embed_text.assert_called_once_with(new_caption)
        mock_store.upsert_caption_embedding.assert_called_once()

    @patch("image_search_app.ingestion.pipeline.ChromaStore")
    @patch("image_search_app.ingestion.pipeline.EmbeddingService")
    @patch("image_search_app.ingestion.pipeline.Captioner")
    @patch("image_search_app.ingestion.pipeline.FaceRecognizer")
    def test_refine_skips_when_no_named_people(
        self, mock_face_cls, mock_captioner_cls, mock_embed_cls, mock_store_cls,
    ):
        """If all faces are dismissed, skip refinement and set status to ready."""
        from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
        from image_search_app.ingestion.pipeline import IngestionPipeline

        create_all()

        file_path = _unique_path("no_names")

        with get_session() as session:
            record = ImageRecord(
                file_path=file_path,
                caption="a person sitting",
                ingestion_status="refining_caption",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            image_id = record.image_id

            session.add(PersonRecord(
                image_id=image_id, name=None, face_id="f1",
                bbox="10,20,100,200", confidence=0.5, dismissed=True,
            ))
            session.commit()

        pipeline = IngestionPipeline()
        result = pipeline.refine_after_labeling(image_id)

        assert result is None
        mock_captioner_cls.return_value.generate_with_names.assert_not_called()

        with get_session() as session:
            rec = session.query(ImageRecord).filter_by(image_id=image_id).first()
            assert rec.ingestion_status == "ready"

    @patch("image_search_app.ingestion.pipeline.ChromaStore")
    @patch("image_search_app.ingestion.pipeline.EmbeddingService")
    @patch("image_search_app.ingestion.pipeline.Captioner")
    @patch("image_search_app.ingestion.pipeline.FaceRecognizer")
    def test_refine_sets_failed_on_error(
        self, mock_face_cls, mock_captioner_cls, mock_embed_cls, mock_store_cls,
    ):
        """If captioning fails, status should be set to 'failed'."""
        from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
        from image_search_app.ingestion.pipeline import IngestionPipeline

        create_all()

        file_path = _unique_path("error")

        with get_session() as session:
            record = ImageRecord(
                file_path=file_path,
                caption="a man",
                ingestion_status="refining_caption",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            image_id = record.image_id

            session.add(PersonRecord(
                image_id=image_id, name="Charlie", face_id="f1",
                bbox="10,20,100,200", confidence=0.9, source="user_tag",
            ))
            session.commit()

        mock_captioner_cls.return_value.generate_with_names.side_effect = RuntimeError("VLM error")

        pipeline = IngestionPipeline()
        with pytest.raises(RuntimeError, match="VLM error"):
            pipeline.refine_after_labeling(image_id)

        with get_session() as session:
            rec = session.query(ImageRecord).filter_by(image_id=image_id).first()
            assert rec.ingestion_status == "failed"

    @patch("image_search_app.ingestion.pipeline.ChromaStore")
    @patch("image_search_app.ingestion.pipeline.EmbeddingService")
    @patch("image_search_app.ingestion.pipeline.Captioner")
    @patch("image_search_app.ingestion.pipeline.FaceRecognizer")
    def test_refine_sorts_names_by_bbox_position(
        self, mock_face_cls, mock_captioner_cls, mock_embed_cls, mock_store_cls,
    ):
        """Names should be passed to captioner sorted by bbox x_min (left to right)."""
        from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
        from image_search_app.ingestion.pipeline import IngestionPipeline

        create_all()

        file_path = _unique_path("three_people")

        with get_session() as session:
            record = ImageRecord(
                file_path=file_path,
                caption="three people standing",
                ingestion_status="refining_caption",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            image_id = record.image_id

            # Add people in non-sorted order
            session.add(PersonRecord(
                image_id=image_id, name="Charlie", face_id="f1",
                bbox="300,20,400,200", confidence=0.9, source="user_tag",
            ))
            session.add(PersonRecord(
                image_id=image_id, name="Alice", face_id="f2",
                bbox="10,20,100,200", confidence=0.9, source="user_tag",
            ))
            session.add(PersonRecord(
                image_id=image_id, name="Bob", face_id="f3",
                bbox="150,20,250,200", confidence=0.9, source="user_tag",
            ))
            session.commit()

        mock_captioner = mock_captioner_cls.return_value
        mock_captioner.generate_with_names.return_value = CaptionResult(
            caption="Alice, Bob, and Charlie standing together", confidence=0.9,
        )
        mock_embed_cls.return_value.embed_text.return_value = [0.1] * 384

        pipeline = IngestionPipeline()
        pipeline.refine_after_labeling(image_id)

        # Should be sorted: Alice (x=10), Bob (x=150), Charlie (x=300)
        mock_captioner.generate_with_names.assert_called_once_with(
            file_path, ["Alice", "Bob", "Charlie"],
            original_caption="three people standing",
        )
