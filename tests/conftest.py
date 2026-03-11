"""Shared fixtures for ingestion pipeline tests."""

import io
import struct
from pathlib import Path

import pytest
from PIL import Image


def _build_exif_bytes() -> bytes:
    """Build raw EXIF bytes with DateTimeOriginal and GPS using Pillow's Exif API."""
    img = Image.new("RGB", (1, 1))
    exif = img.getexif()

    # 0x9003 = DateTimeOriginal (lives in Exif sub-IFD)
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = "2025:07:04 14:30:00"

    # GPS IFD (tag 0x8825)
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[0x0001] = "N"                          # GPSLatitudeRef
    gps_ifd[0x0002] = (40.0, 42.0, 46.08)          # GPSLatitude
    gps_ifd[0x0003] = "W"                          # GPSLongitudeRef
    gps_ifd[0x0004] = (74.0, 0.0, 21.5)            # GPSLongitude

    return exif.tobytes()


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a minimal JPEG with EXIF (timestamp + GPS) for testing."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    exif_bytes = _build_exif_bytes()

    out = tmp_path / "test_photo.jpg"
    img.save(str(out), "JPEG", exif=exif_bytes)
    return out


@pytest.fixture()
def sample_image_no_exif(tmp_path: Path) -> Path:
    """Create a minimal JPEG with no EXIF data."""
    img = Image.new("RGB", (200, 200), color=(50, 100, 150))
    out = tmp_path / "no_exif.jpg"
    img.save(str(out), "JPEG")
    return out


@pytest.fixture()
def sample_face_image(tmp_path: Path) -> Path:
    """Create an image with a simple face-like pattern for face detection tests."""
    from PIL import ImageDraw

    img = Image.new("RGB", (400, 400), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    # Skin-colored oval
    draw.ellipse([120, 80, 280, 300], fill=(220, 190, 170))
    # Eyes
    draw.ellipse([160, 160, 190, 180], fill=(50, 50, 50))
    draw.ellipse([210, 160, 240, 180], fill=(50, 50, 50))
    # Mouth
    draw.arc([170, 220, 230, 260], start=0, end=180, fill=(150, 50, 50), width=2)

    out = tmp_path / "face_test.jpg"
    img.save(str(out), "JPEG")
    return out
