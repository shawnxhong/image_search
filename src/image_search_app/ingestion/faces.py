from dataclasses import dataclass


@dataclass
class FaceDetection:
    face_id: str
    bbox: list[int]
    confidence: float


class FaceRecognizer:
    """Face detection / recognition adapter stub."""

    def detect(self, image_path: str) -> list[FaceDetection]:
        _ = image_path
        return []
