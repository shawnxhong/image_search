from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PersonTag(BaseModel):
    name: str | None = None
    face_id: str
    bbox: list[int] = Field(description="[x_min, y_min, x_max, y_max]")
    confidence: float = 0.0
    source: Literal["user_tag", "auto"] = "auto"


class ImageMetadata(BaseModel):
    image_id: UUID
    file_path: str
    capture_timestamp: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    caption: str | None = None
    caption_confidence: float | None = None
    face_confidence: float | None = None
    geo_confidence: float | None = None
    ingestion_status: str = "received"
    people: list[PersonTag] = Field(default_factory=list)


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 20


class ImageSearchRequest(BaseModel):
    image_path: str
    query: str | None = None
    top_k: int = 20


class MatchExplanation(BaseModel):
    image_id: UUID
    reason: str
    matched_constraints: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    image_id: UUID
    file_path: str
    score: float
    explanation: MatchExplanation


class DualListSearchResponse(BaseModel):
    solid_results: list[SearchResultItem]
    soft_results: list[SearchResultItem]
