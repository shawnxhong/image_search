from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


@dataclass
class ExifData:
    capture_timestamp: datetime | None
    lat: float | None
    lon: float | None
    geo_confidence: float | None


def _get_exif_dict(image_path: str) -> dict:
    """Extract EXIF data from an image as a human-readable dict."""
    path = Path(image_path)
    if not path.exists():
        return {}

    try:
        img = Image.open(path)
        raw = img.getexif()
        if not raw:
            return {}
    except Exception:
        return {}

    exif: dict = {}
    for tag_id, value in raw.items():
        tag = TAGS.get(tag_id, tag_id)
        exif[tag] = value

    # Decode GPSInfo sub-IFD
    gps_ifd = raw.get_ifd(0x8825)
    if gps_ifd:
        gps_data: dict = {}
        for tag_id, value in gps_ifd.items():
            tag = GPSTAGS.get(tag_id, tag_id)
            gps_data[tag] = value
        exif["GPSInfo"] = gps_data

    return exif


def _parse_timestamp(exif: dict) -> datetime | None:
    """Parse capture timestamp from EXIF DateTimeOriginal or DateTime."""
    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw = exif.get(key)
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
    return None


def _dms_to_decimal(dms: tuple, ref: str) -> float | None:
    """Convert GPS DMS (degrees, minutes, seconds) tuple to decimal degrees."""
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError, IndexError):
        return None


def _parse_gps(exif: dict) -> tuple[float | None, float | None]:
    """Parse latitude and longitude from EXIF GPSInfo."""
    gps = exif.get("GPSInfo")
    if not gps or not isinstance(gps, dict):
        return None, None

    lat_dms = gps.get("GPSLatitude")
    lat_ref = gps.get("GPSLatitudeRef", "N")
    lon_dms = gps.get("GPSLongitude")
    lon_ref = gps.get("GPSLongitudeRef", "E")

    lat = _dms_to_decimal(lat_dms, lat_ref) if lat_dms else None
    lon = _dms_to_decimal(lon_dms, lon_ref) if lon_dms else None

    return lat, lon


def extract_exif(image_path: str) -> ExifData:
    """Extract EXIF metadata from an image file."""
    exif = _get_exif_dict(image_path)

    timestamp = _parse_timestamp(exif)
    lat, lon = _parse_gps(exif)

    # Confidence: 1.0 if both lat/lon present, 0.5 if partial, None if absent
    if lat is not None and lon is not None:
        geo_confidence = 1.0
    elif lat is not None or lon is not None:
        geo_confidence = 0.5
    else:
        geo_confidence = None

    return ExifData(
        capture_timestamp=timestamp,
        lat=lat,
        lon=lon,
        geo_confidence=geo_confidence,
    )
