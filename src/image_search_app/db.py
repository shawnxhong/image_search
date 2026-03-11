from __future__ import annotations

from datetime import datetime
from typing import Iterable
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, create_engine, select
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

    image: Mapped[ImageRecord] = relationship(back_populates="people")


engine = create_engine(settings.sqlite_url, future=True)


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
