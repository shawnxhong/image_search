"""
OpenVINO-based face recognition pipeline.

Originally from Intel OpenVINO demo samples, adapted for use as an
importable package within the image_search_app.
"""

import gc
import time
from pathlib import Path

import cv2
import numpy as np
from openvino import Core, get_version

from .face_detector import FaceDetector
from .landmarks_detector import LandmarksDetector
from .face_identifier import FaceIdentifier
from .faces_database import FacesDatabase


class FaceRecognitionOV:
    def __init__(self, model_dir=None):
        self.core = Core()
        self.model_dir = model_dir
        self.initialized = False

    def _resolve_model_path(self, model_dir, explicit_path, candidates, model_name):
        if explicit_path:
            path = Path(explicit_path)
            if not path.exists():
                raise FileNotFoundError(f"{model_name} model not found: {path}")
            return str(path)

        if model_dir is None:
            raise ValueError(f"model_dir is required when {model_name} path is not provided")

        base = Path(model_dir)
        if not base.exists():
            raise FileNotFoundError(f"model_dir does not exist: {base}")

        for candidate in candidates:
            candidate_path = base / candidate
            if candidate_path.exists():
                return str(candidate_path)

        raise FileNotFoundError(
            f"{model_name} model not found in {base}. Tried: {', '.join(candidates)}"
        )

    def initialize(
        self,
        model_dir,
        device="CPU",
        gallery_dir=None,
        run_detector=False,
        no_show=True,
        fd_model_path=None,
        lm_model_path=None,
        reid_model_path=None,
        fd_input_size=(0, 0),
        t_fd=0.6,
        t_id=0.3,
        exp_r_fd=1.15,
        match_algo="MIN_DIST",
    ):
        try:
            fd_path = self._resolve_model_path(
                model_dir,
                fd_model_path,
                ["face-detection-retail-0004.xml", "face-detection-adas-0001.xml"],
                "face detection",
            )
            lm_path = self._resolve_model_path(
                model_dir,
                lm_model_path,
                ["landmarks-regression-retail-0009.xml"],
                "landmarks",
            )
            reid_path = self._resolve_model_path(
                model_dir,
                reid_model_path,
                [
                    "face-reidentification-retail-0095.xml",
                    "face-recognition-resnet100-arcface-onnx.xml",
                    "facenet-20180408-102900.xml",
                ],
                "face reidentification",
            )

            if isinstance(device, dict):
                d_fd = device.get("fd", "CPU")
                d_lm = device.get("lm", "CPU")
                d_reid = device.get("reid", "CPU")
            else:
                d_fd = d_lm = d_reid = device

            self.face_detector = FaceDetector(
                self.core,
                fd_path,
                fd_input_size,
                confidence_threshold=t_fd,
                roi_scale_factor=exp_r_fd,
            )
            self.landmarks_detector = LandmarksDetector(self.core, lm_path)
            self.face_identifier = FaceIdentifier(
                self.core,
                reid_path,
                match_threshold=t_id,
                match_algo=match_algo,
            )

            self.face_detector.deploy(d_fd)
            self.landmarks_detector.deploy(d_lm, 16)
            self.face_identifier.deploy(d_reid, 16)

            self.faces_database = None
            if gallery_dir:
                self.faces_database = FacesDatabase(
                    gallery_dir,
                    self.face_identifier,
                    self.landmarks_detector,
                    self.face_detector if run_detector else None,
                    no_show,
                )
                self.face_identifier.set_faces_database(self.faces_database)

            self.device = device
            self.model_dir = model_dir
            self.gallery_dir = gallery_dir
            self.initialized = True

        except Exception as e:
            print(f"Error initializing FaceRecognition pipeline: {e}")
            raise

    def release(self):
        try:
            print("Releasing FaceRecognition resources...")
            attrs_to_release = [
                "face_detector",
                "landmarks_detector",
                "face_identifier",
                "faces_database",
                "device",
                "gallery_dir",
            ]

            for attr in attrs_to_release:
                if hasattr(self, attr):
                    delattr(self, attr)

            self.initialized = False
            gc.collect()
            print("All FaceRecognition resources released successfully!")

        except Exception as e:
            print(f"Error during release: {e}")
            self.initialized = False

    def _infer_descriptors_without_gallery(self, image, rois, landmarks):
        self.face_identifier.clear()
        self.face_identifier.start_async(image, rois, landmarks)
        descriptors = self.face_identifier.get_descriptors()

        results = []
        for descriptor in descriptors:
            results.append(
                {
                    "id": FaceIdentifier.UNKNOWN_ID,
                    "label": FaceIdentifier.UNKNOWN_ID_LABEL,
                    "distance": None,
                    "descriptor": descriptor,
                }
            )
        return results

    def infer(self, image, frame_name=None):
        if not self.initialized:
            raise RuntimeError("Model not initialized. Please call initialize() first.")

        if image is None:
            raise ValueError("Input image is None")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            frame_name = frame_name or image_path.name
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

        rois = self.face_detector.infer((image,))
        landmarks = self.landmarks_detector.infer((image, rois)) if rois else []

        if self.faces_database is not None:
            identities, _ = self.face_identifier.infer((image, rois, landmarks, frame_name))
            identity_info = [
                {
                    "id": identity.id,
                    "label": self.face_identifier.get_identity_label(identity.id),
                    "distance": float(identity.distance),
                    "descriptor": identity.descriptor,
                }
                for identity in identities
            ]
        else:
            identity_info = self._infer_descriptors_without_gallery(image, rois, landmarks)

        results = []
        for i, roi in enumerate(rois):
            xmin = int(max(roi.position[0], 0))
            ymin = int(max(roi.position[1], 0))
            xmax = int(max(roi.position[0] + roi.size[0], 0))
            ymax = int(max(roi.position[1] + roi.size[1], 0))

            result = {
                "bbox": [xmin, ymin, xmax, ymax],
                "landmarks": landmarks[i].tolist() if i < len(landmarks) else [],
                "id": identity_info[i]["id"] if i < len(identity_info) else FaceIdentifier.UNKNOWN_ID,
                "label": identity_info[i]["label"] if i < len(identity_info) else FaceIdentifier.UNKNOWN_ID_LABEL,
                "distance": identity_info[i]["distance"] if i < len(identity_info) else None,
                "descriptor": identity_info[i]["descriptor"] if i < len(identity_info) else None,
            }
            results.append(result)

        return results

    def infer_from_path(self, image_path):
        return self.infer(image_path)
