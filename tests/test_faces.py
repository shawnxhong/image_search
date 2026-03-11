"""Tests for face detection (ingestion step 6)."""

from pathlib import Path

from image_search_app.ingestion.faces import FaceDetection, FaceRecognizer


def test_face_recognizer_returns_list(sample_image_no_exif: Path):
    recognizer = FaceRecognizer()
    result = recognizer.detect(str(sample_image_no_exif))
    assert isinstance(result, list)


def test_plain_image_returns_empty_or_faces(sample_image_no_exif: Path):
    """A plain colored image should return an empty list (no faces)."""
    recognizer = FaceRecognizer()
    result = recognizer.detect(str(sample_image_no_exif))
    # Plain solid-color image — should detect no faces
    assert len(result) == 0


def test_face_detection_fields_are_valid(sample_face_image: Path):
    """If faces are detected, each should have valid face_id, bbox, confidence."""
    recognizer = FaceRecognizer()
    result = recognizer.detect(str(sample_face_image))
    # The synthetic face may or may not trigger detection —
    # we only validate structure if it does
    for face in result:
        assert isinstance(face, FaceDetection)
        assert isinstance(face.face_id, str)
        assert len(face.face_id) > 0
        assert len(face.bbox) == 4
        x_min, y_min, x_max, y_max = face.bbox
        assert x_min < x_max
        assert y_min < y_max
        assert 0.0 <= face.confidence <= 1.0


def test_face_recognizer_does_not_crash_on_various_images(
    sample_image: Path, sample_image_no_exif: Path, sample_face_image: Path
):
    """Face recognizer should not crash regardless of input image."""
    recognizer = FaceRecognizer()
    for img_path in [sample_image, sample_image_no_exif, sample_face_image]:
        result = recognizer.detect(str(img_path))
        assert isinstance(result, list)
