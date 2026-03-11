"""Tests for image captioning (ingestion step 5)."""

from pathlib import Path

from image_search_app.ingestion.captioner import CaptionResult, Captioner


def test_captioner_returns_caption_result(sample_image_no_exif: Path):
    captioner = Captioner()
    result = captioner.generate(str(sample_image_no_exif))
    assert isinstance(result, CaptionResult)


def test_caption_is_nonempty_string(sample_image_no_exif: Path):
    captioner = Captioner()
    result = captioner.generate(str(sample_image_no_exif))
    assert isinstance(result.caption, str)
    assert len(result.caption) > 0


def test_confidence_is_valid_range(sample_image_no_exif: Path):
    captioner = Captioner()
    result = captioner.generate(str(sample_image_no_exif))
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_captioner_handles_different_images(sample_image: Path, sample_image_no_exif: Path):
    """Captioner should produce results for different images without crashing."""
    captioner = Captioner()
    r1 = captioner.generate(str(sample_image))
    r2 = captioner.generate(str(sample_image_no_exif))
    assert len(r1.caption) > 0
    assert len(r2.caption) > 0
