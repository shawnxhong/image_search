from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from image_search_app.config import settings

logger = logging.getLogger(__name__)

# The face_recognition package lives outside src; add it to sys.path once.
_FACE_REC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "face_recognition"


@dataclass
class FaceDetection:
    face_id: str
    bbox: list[int]
    confidence: float
    descriptor: list[float] = field(default_factory=list)


class FaceRecognizer:
    """Face detection using OpenVINO models (face-detection-retail-0004)."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            # Ensure the colleague's face_recognition package is importable
            fr_dir = str(_FACE_REC_DIR)
            if fr_dir not in sys.path:
                sys.path.insert(0, fr_dir)

            from face_recognition_ov import FaceRecognitionOV

            project_root = _FACE_REC_DIR.parent

            pipeline = FaceRecognitionOV()
            pipeline.initialize(
                model_dir=str(project_root),
                fd_model_path=str(project_root / settings.face_detection_model),
                lm_model_path=str(project_root / settings.face_landmarks_model),
                reid_model_path=str(project_root / settings.face_reidentification_model),
                t_fd=settings.face_detection_confidence,
            )
            self._pipeline = pipeline
            logger.info("OpenVINO face detection pipeline loaded")

    def detect(self, image_path: str) -> list[FaceDetection]:
        self._load()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to read image for face detection: %s", image_path)
            return []

        results = self._pipeline.infer(image)

        faces: list[FaceDetection] = []
        for r in results:
            bbox = r["bbox"]  # [xmin, ymin, xmax, ymax]
            descriptor = r.get("descriptor")
            desc_list: list[float] = []
            if descriptor is not None:
                if isinstance(descriptor, np.ndarray):
                    desc_list = descriptor.tolist()
                else:
                    desc_list = list(descriptor)

            faces.append(
                FaceDetection(
                    face_id=str(uuid4()),
                    bbox=[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    confidence=0.9,
                    descriptor=desc_list,
                )
            )

        return faces
