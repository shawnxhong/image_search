from __future__ import annotations

import threading
from dataclasses import dataclass

from PIL import Image

from image_search_app.config import settings


@dataclass
class CaptionResult:
    caption: str
    confidence: float


class Captioner:
    """Image captioning using BLIP (Salesforce/blip-image-captioning-base)."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_name = settings.caption_model_name
            try:
                processor = BlipProcessor.from_pretrained(model_name)
                model = BlipForConditionalGeneration.from_pretrained(model_name)
            except Exception:
                # Fall back to cached model if network is unavailable
                processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
                model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self._device)
            model.eval()
            self._processor = processor
            self._model = model

    def generate(self, image_path: str) -> CaptionResult:
        import torch

        self._load()

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                output_scores=True,
                return_dict_in_generate=True,
            )

        caption = self._processor.decode(output.sequences[0], skip_special_tokens=True).strip()

        # Approximate confidence from sequence scores
        if output.sequences_scores is not None:
            # scores are log-probabilities; convert to 0-1 range
            log_prob = output.sequences_scores[0].item()
            confidence = min(1.0, max(0.0, 1.0 + log_prob / 10.0))
        else:
            confidence = 0.5

        return CaptionResult(caption=caption, confidence=round(confidence, 4))
