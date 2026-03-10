from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from image_search_app.db import ImageRecord, PersonRecord, get_session, upsert_image
from image_search_app.ingestion.captioner import Captioner
from image_search_app.ingestion.exif import extract_exif
from image_search_app.ingestion.faces import FaceRecognizer
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService


class IngestionPipeline:
    def __init__(self) -> None:
        self.captioner = Captioner()
        self.face_recognizer = FaceRecognizer()
        self.embeddings = EmbeddingService()
        self.store = ChromaStore()

    def ingest(self, image_path: str) -> str:
        image = upsert_image(image_path)

        with get_session() as session:
            record = session.scalar(select(ImageRecord).where(ImageRecord.image_id == image.image_id))
            if record is None:
                raise ValueError("Image was not persisted")

            exif = extract_exif(image_path)
            record.capture_timestamp = exif.capture_timestamp
            record.lat = exif.lat
            record.lon = exif.lon
            record.geo_confidence = exif.geo_confidence
            record.ingestion_status = "exif_extracted"

            caption_result = self.captioner.generate(image_path)
            record.caption = caption_result.caption
            record.caption_confidence = caption_result.confidence
            record.ingestion_status = "captioned"

            faces = self.face_recognizer.detect(image_path)
            for face in faces:
                session.add(
                    PersonRecord(
                        image_id=record.image_id,
                        name=None,
                        face_id=face.face_id,
                        bbox=",".join(map(str, face.bbox)),
                        confidence=face.confidence,
                        source="auto",
                    )
                )
            record.face_confidence = max([f.confidence for f in faces], default=0.0)
            record.ingestion_status = "faces_detected"

            caption_embedding = self.embeddings.embed_text(record.caption or "")
            image_embedding = self.embeddings.embed_image(record.file_path)
            self.store.upsert_caption_embedding(record.image_id, caption_embedding, record.caption)
            self.store.upsert_image_embedding(record.image_id, image_embedding)

            now = datetime.now(timezone.utc)
            record.caption_indexed_at = now
            record.image_indexed_at = now
            record.embedding_model_version = "starter-v1"
            record.ingestion_status = "ready"

            session.commit()
            return record.image_id
