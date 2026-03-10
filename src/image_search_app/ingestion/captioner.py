from __future__ import annotations

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

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(settings.caption_model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(settings.caption_model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

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
