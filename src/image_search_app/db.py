from __future__ import annotations

from datetime import datetime
from typing import Iterable
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from image_search_app.config import settings


class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    __tablename__ = "images"

    image_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    file_path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    capture_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    caption: Mapped[str | None] = mapped_column(String, nullable=True)
    caption_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    geo_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str | None] = mapped_column(String, nullable=True)
    city: Mapped[str | None] = mapped_column(String, nullable=True)
    ingestion_status: Mapped[str] = mapped_column(String, default="received")
    caption_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    image_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    embedding_model_version: Mapped[str | None] = mapped_column(String, nullable=True)

    people: Mapped[list[PersonRecord]] = relationship(back_populates="image", cascade="all, delete-orphan")


class PersonRecord(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.image_id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    face_id: Mapped[str] = mapped_column(String, nullable=False)
    bbox: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String, default="auto")
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    descriptor: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-encoded face embedding
    candidates: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: [{"name": "...", "distance": 0.xx}]

    image: Mapped[ImageRecord] = relationship(back_populates="people")


from sqlalchemy import event as sa_event

engine = create_engine(
    settings.sqlite_url,
    future=True,
    connect_args={"timeout": 30},  # SQLite busy timeout in seconds
)


@sa_event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable WAL mode and busy timeout for concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def create_all() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def upsert_image(file_path: str) -> ImageRecord:
    with get_session() as session:
        existing = session.scalar(select(ImageRecord).where(ImageRecord.file_path == file_path))
        if existing:
            return existing
        image = ImageRecord(file_path=file_path)
        session.add(image)
        session.commit()
        session.refresh(image)
        return image


def list_images() -> Iterable[ImageRecord]:
    with get_session() as session:
        return session.scalars(select(ImageRecord)).all()


def list_images_paginated(limit: int = 50, cursor: str | None = None) -> tuple[list[ImageRecord], int]:
    """Return images sorted by capture_timestamp DESC (nulls last), with cursor pagination.

    Returns (images, total_count).  The cursor is an image_id; when supplied,
    results start after that image's position in the sort order.
    """
    from sqlalchemy import case, func

    with get_session() as session:
        total = session.scalar(select(func.count()).select_from(ImageRecord)) or 0

        # Sort: images with timestamps first (DESC), then images without timestamps
        has_ts = case(
            (ImageRecord.capture_timestamp.isnot(None), 0),
            else_=1,
        )
        ordering = [has_ts, ImageRecord.capture_timestamp.desc(), ImageRecord.image_id.desc()]

        if cursor:
            # Find the cursor row's position values
            cursor_rec = session.scalar(
                select(ImageRecord).where(ImageRecord.image_id == cursor)
            )
            if cursor_rec and cursor_rec.capture_timestamp is not None:
                # After this timestamped row
                from sqlalchemy import or_, and_, tuple_
                stmt = (
                    select(ImageRecord)
                    .where(
                        or_(
                            ImageRecord.capture_timestamp < cursor_rec.capture_timestamp,
                            and_(
                                ImageRecord.capture_timestamp == cursor_rec.capture_timestamp,
                                ImageRecord.image_id < cursor,
                            ),
                            ImageRecord.capture_timestamp.is_(None),
                        )
                    )
                    .order_by(*ordering)
                    .limit(limit)
                )
            elif cursor_rec:
                # Cursor row has no timestamp — only get nulls after it
                stmt = (
                    select(ImageRecord)
                    .where(
                        ImageRecord.capture_timestamp.is_(None),
                        ImageRecord.image_id < cursor,
                    )
                    .order_by(*ordering)
                    .limit(limit)
                )
            else:
                # Cursor not found, start from beginning
                stmt = select(ImageRecord).order_by(*ordering).limit(limit)
        else:
            stmt = select(ImageRecord).order_by(*ordering).limit(limit)

        images = list(session.scalars(stmt).all())
        return images, total
