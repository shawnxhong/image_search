from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from PIL import Image


@dataclass
class FaceDetection:
    face_id: str
    bbox: list[int]
    confidence: float


class FaceRecognizer:
    """Face detection using MediaPipe Face Detection."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._detector = None
        self._min_confidence = min_confidence

    def _load(self) -> None:
        if self._detector is not None:
            return

        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (better for varied distances)
            min_detection_confidence=self._min_confidence,
        )

    def detect(self, image_path: str) -> list[FaceDetection]:
        import numpy as np

        self._load()

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        results = self._detector.process(img_array)

        faces: list[FaceDetection] = []
        if not results.detections:
            return faces

        for detection in results.detections:
            bbox_rel = detection.location_data.relative_bounding_box

            # Convert relative bbox to absolute pixel coordinates
            x_min = max(0, int(bbox_rel.xmin * w))
            y_min = max(0, int(bbox_rel.ymin * h))
            x_max = min(w, int((bbox_rel.xmin + bbox_rel.width) * w))
            y_max = min(h, int((bbox_rel.ymin + bbox_rel.height) * h))

            confidence = detection.score[0] if detection.score else 0.0

            faces.append(
                FaceDetection(
                    face_id=str(uuid4()),
                    bbox=[x_min, y_min, x_max, y_max],
                    confidence=round(confidence, 4),
                )
            )

        return faces
