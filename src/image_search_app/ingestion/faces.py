from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


@dataclass
class FaceDetection:
    face_id: str
    bbox: list[int]
    confidence: float


class FaceRecognizer:
    """Face detection using OpenCV's Haar cascade classifier."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        self._cascade = None
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def _load(self) -> None:
        if self._cascade is not None:
            return

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_path: str) -> list[FaceDetection]:
        self._load()

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(30, 30),
        )

        faces: list[FaceDetection] = []
        if len(detections) == 0:
            return faces

        for x, y, w, h in detections:
            faces.append(
                FaceDetection(
                    face_id=str(uuid4()),
                    bbox=[int(x), int(y), int(x + w), int(y + h)],
                    confidence=0.85,  # Haar doesn't provide per-detection confidence
                )
            )

        return faces
