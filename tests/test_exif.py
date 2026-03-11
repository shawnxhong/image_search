"""Tests for EXIF extraction (ingestion step 3-4)."""

from pathlib import Path

from image_search_app.ingestion.exif import ExifData, extract_exif


def test_extract_exif_returns_dataclass(sample_image: Path):
    result = extract_exif(str(sample_image))
    assert isinstance(result, ExifData)


def test_extract_timestamp(sample_image: Path):
    result = extract_exif(str(sample_image))
    assert result.capture_timestamp is not None
    assert result.capture_timestamp.year == 2025
    assert result.capture_timestamp.month == 7
    assert result.capture_timestamp.day == 4
    assert result.capture_timestamp.hour == 14
    assert result.capture_timestamp.minute == 30


def test_extract_gps(sample_image: Path):
    result = extract_exif(str(sample_image))
    assert result.lat is not None
    assert result.lon is not None
    # NYC: lat ~40.7, lon ~-74.0
    assert 40.0 < result.lat < 41.0
    assert -75.0 < result.lon < -73.0


def test_geo_confidence_present_when_gps_available(sample_image: Path):
    result = extract_exif(str(sample_image))
    assert result.geo_confidence == 1.0


def test_no_exif_returns_nones(sample_image_no_exif: Path):
    result = extract_exif(str(sample_image_no_exif))
    assert result.capture_timestamp is None
    assert result.lat is None
    assert result.lon is None
    assert result.geo_confidence is None


def test_nonexistent_file_returns_nones():
    result = extract_exif("/nonexistent/path/fake.jpg")
    assert result.capture_timestamp is None
    assert result.lat is None
    assert result.lon is None
