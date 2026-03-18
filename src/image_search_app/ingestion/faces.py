from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from image_search_app.config import settings

logger = logging.getLogger(__name__)

# Project root (where pyproject.toml lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Models that must be present for face recognition
_REQUIRED_MODELS = [
    "face-detection-retail-0004",
    "landmarks-regression-retail-0009",
    "face-reidentification-retail-0095",
]


def _ensure_face_models() -> None:
    """Download face recognition models if they are missing."""
    models_dir = _PROJECT_ROOT / settings.face_models_dir
    fd_path = models_dir / settings.face_detection_model

    if fd_path.exists():
        return  # All models likely present

    logger.info("Face detection models not found, downloading to %s ...", models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_names = ",".join(_REQUIRED_MODELS)
    base_args = [
        "--name", model_names,
        "--precisions", "FP32",
        "-o", str(models_dir),
    ]

    # Try multiple ways to find omz_downloader
    candidates = [
        # 1. Scripts dir next to the Python executable (conda/venv)
        [str(Path(sys.executable).parent / "Scripts" / "omz_downloader"), *base_args],
        # 2. Same dir as Python (Linux venvs)
        [str(Path(sys.executable).parent / "omz_downloader"), *base_args],
        # 3. Bare command on PATH
        ["omz_downloader", *base_args],
    ]

    for cmd in candidates:
        exe_path = Path(cmd[0])
        # Skip if it's a path-based candidate and the file doesn't exist
        if "/" in cmd[0] or "\\" in cmd[0]:
            if not exe_path.exists() and not exe_path.with_suffix(".exe").exists():
                continue
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("Face models downloaded successfully")
                return
            else:
                logger.warning("omz_downloader failed (rc=%d): %s", result.returncode, result.stderr)
        except FileNotFoundError:
            continue

    raise RuntimeError(
        f"omz_downloader not found. Install openvino-dev and run:\n"
        f"  omz_downloader --name {model_names} --precisions FP32 -o {models_dir}"
    )


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

            _ensure_face_models()

            from image_search_app.face_recognition import FaceRecognitionOV

            models_dir = _PROJECT_ROOT / settings.face_models_dir

            pipeline = FaceRecognitionOV()
            pipeline.initialize(
                model_dir=str(models_dir),
                fd_model_path=str(models_dir / settings.face_detection_model),
                lm_model_path=str(models_dir / settings.face_landmarks_model),
                reid_model_path=str(models_dir / settings.face_reidentification_model),
                t_fd=settings.face_detection_confidence,
            )
            self._pipeline = pipeline
            logger.info("OpenVINO face detection pipeline loaded")

    def unload(self) -> None:
        """Release the face detection pipeline and free memory."""
        with self._lock:
            if self._pipeline is not None:
                import gc
                self._pipeline.release()
                self._pipeline = None
                gc.collect()
                logger.info("Face detection pipeline unloaded")

    def status(self) -> dict:
        return {
            "loaded": self._pipeline is not None,
            "name": "Face Detection",
            "model_name": "Intel OMZ Face Pipeline",
            "device": "CPU",
        }

    def detect(self, image_path: str) -> list[FaceDetection]:
        self._load()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to read image for face detection: %s", image_path)
            return []

        try:
            results = self._pipeline.infer(image)
        except AssertionError as e:
            logger.warning("Face pipeline assertion error for %s: %s — skipping face detection", image_path, e)
            return []

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
