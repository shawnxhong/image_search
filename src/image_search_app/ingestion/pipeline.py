from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from image_search_app.config import settings
from image_search_app.db import ImageRecord, PersonRecord, get_session
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

        ML inference (captioning, face detection, embeddings) runs outside
        any DB session so the SQLite lock is not held during slow work.
        A short DB session is opened only at the end to persist results.

        Returns the image_id on success, raises on failure.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        path_str = str(path)

        # --- Phase 0: skip if already fully ingested ---
        with get_session() as session:
            existing = session.scalar(
                select(ImageRecord).where(ImageRecord.file_path == path_str)
            )
            if existing and existing.ingestion_status in ("ready", "pending_labels"):
                logger.info("Already ingested, skipping: %s", existing.image_id)
                return existing.image_id

        # --- Phase 1: quick DB upsert to get an image_id ---
        with get_session() as session:
            record = session.scalar(
                select(ImageRecord).where(ImageRecord.file_path == path_str)
            )
            if record is None:
                record = ImageRecord(file_path=path_str, ingestion_status="processing")
                session.add(record)
                session.commit()
                session.refresh(record)
            else:
                record.ingestion_status = "processing"
                session.commit()
            image_id = record.image_id

        # --- Phase 2: slow ML work (no DB session held) ---
        try:
            exif = extract_exif(path_str)
            logger.info("EXIF extracted for %s", image_id)

            caption_result = self.captioner.generate(path_str)
            logger.info("Caption generated for %s: %s", image_id, caption_result.caption)

            faces = self.face_recognizer.detect(path_str)
            logger.info("Detected %d face(s) for %s", len(faces), image_id)

            # Try to match each face against known identities.
            # Never auto-assign — always present candidates for the user to confirm.
            for face in faces:
                if face.descriptor:
                    candidates = self.store.match_face_candidates(
                        face.descriptor, top_k=3, threshold=0.8,
                    )
                    face.candidates = candidates  # [(name, distance), ...]
                    if candidates:
                        logger.info(
                            "Face %s candidates: %s for %s",
                            face.face_id,
                            [(n, f"{d:.3f}") for n, d in candidates],
                            image_id,
                        )
                else:
                    face.candidates = []
                face.matched_name = None

            caption_embedding = self.embeddings.embed_text(caption_result.caption or "")
            image_embedding = self.embeddings.embed_image(path_str)

            # ChromaDB upserts are idempotent — safe even if DB write below fails
            self.store.upsert_caption_embedding(
                image_id, caption_embedding, caption_result.caption
            )
            self.store.upsert_image_embedding(image_id, image_embedding)

        except Exception as exc:
            # Mark as failed in DB
            try:
                with get_session() as session:
                    rec = session.scalar(
                        select(ImageRecord).where(ImageRecord.image_id == image_id)
                    )
                    if rec:
                        rec.ingestion_status = "failed"
                        session.commit()
            except Exception:
                pass
            logger.error("Ingestion failed for %s: %s", image_path, exc)
            raise

        # --- Phase 3: short DB write to persist all results ---
        has_unlabeled_faces = False
        with get_session() as session:
            record = session.scalar(
                select(ImageRecord).where(ImageRecord.image_id == image_id)
            )
            if record is None:
                raise ValueError(f"Image record disappeared for {image_id}")

            # Clear old faces on re-ingest
            for person in list(record.people):
                session.delete(person)

            record.capture_timestamp = exif.capture_timestamp
            record.lat = exif.lat
            record.lon = exif.lon
            record.geo_confidence = exif.geo_confidence
            record.caption = caption_result.caption
            record.caption_confidence = caption_result.confidence
            record.face_confidence = max((f.confidence for f in faces), default=0.0)

            for face in faces:
                matched_name = getattr(face, "matched_name", None)
                candidates = getattr(face, "candidates", [])
                source = "auto_matched" if matched_name else "auto"
                # Store candidates as JSON: [{"name": "...", "distance": 0.xx}, ...]
                candidates_json = json.dumps(
                    [{"name": n, "distance": round(d, 4)} for n, d in candidates]
                ) if candidates else None
                session.add(
                    PersonRecord(
                        image_id=image_id,
                        name=matched_name,
                        face_id=face.face_id,
                        bbox=",".join(map(str, face.bbox)),
                        confidence=face.confidence,
                        source=source,
                        descriptor=json.dumps(face.descriptor) if face.descriptor else None,
                        candidates=candidates_json,
                    )
                )
                if not matched_name:
                    has_unlabeled_faces = True

            now = datetime.now(timezone.utc)
            record.caption_indexed_at = now
            record.image_indexed_at = now
            record.embedding_model_version = "all-MiniLM-L6-v2"

            # Set status based on whether there are unlabeled faces
            if has_unlabeled_faces:
                record.ingestion_status = "pending_labels"
            else:
                record.ingestion_status = "ready"

            session.commit()

        logger.info("Ingestion complete for %s (status=%s)",
                     image_id, "pending_labels" if has_unlabeled_faces else "ready")
        return image_id

    def refine_after_labeling(self, image_id: str) -> str | None:
        """Re-generate caption with person names after face labeling.

        Called when all faces in an image have been named or dismissed.
        Uses the VLM captioner with the names in the prompt,
        then updates the DB caption and ChromaDB embedding.

        Returns the new caption, or None if refinement was skipped.
        """
        # Load record and collect named people
        with get_session() as session:
            record = session.scalar(
                select(ImageRecord).where(ImageRecord.image_id == image_id)
            )
            if record is None:
                logger.warning("refine_after_labeling: image %s not found", image_id)
                return None

            file_path = record.file_path

            # Collect non-dismissed people sorted by bbox x_min (left to right)
            people = [
                p for p in record.people
                if not p.dismissed and p.name
            ]

        if not people:
            # No named people — nothing to refine
            logger.info("No named people for %s, skipping caption refinement", image_id)
            with get_session() as session:
                rec = session.scalar(
                    select(ImageRecord).where(ImageRecord.image_id == image_id)
                )
                if rec and rec.ingestion_status == "refining_caption":
                    rec.ingestion_status = "ready"
                    session.commit()
            return None

        # Sort by bbox x_min (left to right in the image)
        def bbox_x(p):
            try:
                return int(p.bbox.split(",")[0])
            except (ValueError, IndexError):
                return 0

        people.sort(key=bbox_x)
        names = [p.name for p in people]

        # Generate refined caption
        try:
            logger.info("Refining caption for %s with names: %s", image_id, names)
            caption_result = self.captioner.generate_with_names(file_path, names)
            new_caption = caption_result.caption
            logger.info("Refined caption for %s: %s", image_id, new_caption)

            # Re-embed and update ChromaDB
            caption_embedding = self.embeddings.embed_text(new_caption)
            self.store.upsert_caption_embedding(image_id, caption_embedding, new_caption)

            # Update DB
            with get_session() as session:
                rec = session.scalar(
                    select(ImageRecord).where(ImageRecord.image_id == image_id)
                )
                if rec:
                    rec.caption = new_caption
                    rec.caption_confidence = caption_result.confidence
                    rec.ingestion_status = "ready"
                    session.commit()

            return new_caption

        except Exception as exc:
            logger.exception("Caption refinement failed for %s", image_id)
            with get_session() as session:
                rec = session.scalar(
                    select(ImageRecord).where(ImageRecord.image_id == image_id)
                )
                if rec:
                    rec.ingestion_status = "failed"
                    session.commit()
            raise
