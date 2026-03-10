from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from image_search_app.db import ImageRecord, PersonRecord, get_session, upsert_image
from image_search_app.ingestion.captioner import Captioner
from image_search_app.ingestion.exif import extract_exif
from image_search_app.ingestion.faces import FaceRecognizer
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.captioner = Captioner()
        self.face_recognizer = FaceRecognizer()
        self.embeddings = EmbeddingService()
        self.store = ChromaStore()

    def ingest(self, image_path: str) -> str:
        """Run the full ingestion pipeline for a single image.

        Steps:
        1. Validate file exists
        2. Upsert image record in DB
        3. Extract EXIF (timestamp, GPS)
        4. Generate caption via BLIP
        5. Detect faces via MediaPipe
        6. Generate CLIP embeddings (caption + image)
        7. Index embeddings in ChromaDB
        8. Mark as ready

        Raises on unrecoverable errors; the API layer catches and returns 'failed'.
        """
        # 1. Validate
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 2. Upsert DB record
        image = upsert_image(str(path))

        with get_session() as session:
            record = session.scalar(
                select(ImageRecord).where(ImageRecord.image_id == image.image_id)
            )
            if record is None:
                raise ValueError("Image was not persisted")

            try:
                # 3. EXIF extraction
                exif = extract_exif(str(path))
                record.capture_timestamp = exif.capture_timestamp
                record.lat = exif.lat
                record.lon = exif.lon
                record.geo_confidence = exif.geo_confidence
                record.ingestion_status = "exif_extracted"
                logger.info("EXIF extracted for %s", image.image_id)

                # 4. Caption generation
                caption_result = self.captioner.generate(str(path))
                record.caption = caption_result.caption
                record.caption_confidence = caption_result.confidence
                record.ingestion_status = "captioned"
                logger.info("Caption generated for %s: %s", image.image_id, caption_result.caption)

                # 5. Face detection
                faces = self.face_recognizer.detect(str(path))
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
                record.face_confidence = max((f.confidence for f in faces), default=0.0)
                record.ingestion_status = "faces_detected"
                logger.info("Detected %d face(s) for %s", len(faces), image.image_id)

                # 6-7. Embeddings + indexing
                caption_embedding = self.embeddings.embed_text(record.caption or "")
                image_embedding = self.embeddings.embed_image(str(path))
                self.store.upsert_caption_embedding(
                    record.image_id, caption_embedding, record.caption
                )
                self.store.upsert_image_embedding(record.image_id, image_embedding)

                now = datetime.now(timezone.utc)
                record.caption_indexed_at = now
                record.image_indexed_at = now
                record.embedding_model_version = "clip-vit-base-patch32"
                record.ingestion_status = "ready"
                logger.info("Ingestion complete for %s", image.image_id)

            except Exception:
                record.ingestion_status = "failed"
                session.commit()
                raise

            session.commit()
            return record.image_id
