from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExifData:
    capture_timestamp: datetime | None
    lat: float | None
    lon: float | None
    geo_confidence: float | None


def extract_exif(image_path: str) -> ExifData:
    """Starter stub for EXIF extraction.

    Replace with Pillow/piexif implementation in the next iteration.
    """
    _ = image_path
    return ExifData(capture_timestamp=None, lat=None, lon=None, geo_confidence=None)
