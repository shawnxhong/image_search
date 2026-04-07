"""Image captioning using Qwen2.5-VL via OpenVINO GenAI VLMPipeline."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_search_app.config import settings

logger = logging.getLogger(__name__)

# Project root (where pyproject.toml lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Limit image size to avoid GPU OOM on large photos
MAX_IMAGE_PIXELS = 1024 * 1024  # ~1 megapixel

CAPTION_PROMPT = "Describe this image in one sentence."

CAPTION_REFINE_PROMPT_SINGLE = (
    "The person in this photo is named {names}. "
    "Describe this photo in one sentence using their name."
)

CAPTION_REFINE_PROMPT_MULTI = (
    "The people in this photo are named {names}. "
    "Describe this photo in one sentence using their names."
)


@dataclass
class CaptionResult:
    caption: str
    confidence: float


def _load_image_as_tensor(image_path: str):
    """Load an image and convert to OpenVINO Tensor (NHWC uint8).

    Large images are resized to stay within MAX_IMAGE_PIXELS.
    """
    import openvino as ov

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.debug("Resized %dx%d -> %dx%d", w, h, img.width, img.height)
    arr = np.expand_dims(np.array(img), axis=0)  # NHWC
    return ov.Tensor(arr)


def _format_names(names: list[str]) -> str:
    """Format a list of names for the prompt.

    Examples:
        ["Alice"]               -> "Alice"
        ["Alice", "Bob"]        -> "Alice, Bob"
        ["Alice", "Bob", "Eve"] -> "Alice, Bob, Eve"
    """
    return ", ".join(names)


class Captioner:
    """Image captioning using Qwen2.5-VL via OpenVINO GenAI VLMPipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            import openvino_genai as ov_genai

            model_path = Path(settings.vlm_model_path)
            if not model_path.is_absolute():
                model_path = _PROJECT_ROOT / model_path
            device = settings.vlm_device
            logger.info("Loading VLM captioner from %s on %s", model_path, device)
            self._pipeline = ov_genai.VLMPipeline(str(model_path), device)
            logger.info("VLM captioner loaded")

    def unload(self) -> None:
        """Release the VLM pipeline and free memory."""
        with self._lock:
            if self._pipeline is not None:
                import gc
                del self._pipeline
                self._pipeline = None
                gc.collect()
                logger.info("VLM captioner unloaded")

    def status(self) -> dict:
        model_path = Path(settings.vlm_model_path)
        if not model_path.is_absolute():
            model_path = _PROJECT_ROOT / model_path
        model_name = model_path.parent.name if model_path.name in ("INT4", "INT8", "FP16", "FP32") else model_path.name
        return {
            "loaded": self._pipeline is not None,
            "name": "VL Captioner",
            "model_name": model_name,
            "device": settings.vlm_device,
        }

    def generate(self, image_path: str) -> CaptionResult:
        """Generate an unconditional caption for an image."""
        self._load()

        image_tensor = _load_image_as_tensor(image_path)
        config = self._gen_config()

        with self._lock:
            result = self._pipeline.generate(
                CAPTION_PROMPT,
                images=[image_tensor],
                generation_config=config,
            )
            self._pipeline.finish_chat()

        caption = str(result).strip()
        return CaptionResult(caption=caption, confidence=0.8)

    def generate_with_names(
        self, image_path: str, names: list[str], original_caption: str | None = None,
    ) -> CaptionResult:
        """Generate a caption that includes person names.

        Uses a short, direct prompt that tells the VLM who is in the photo
        and asks it to describe the scene using their names.
        """
        self._load()

        image_tensor = _load_image_as_tensor(image_path)
        config = self._gen_config()

        names_str = _format_names(names)
        if len(names) == 1:
            prompt = CAPTION_REFINE_PROMPT_SINGLE.format(names=names_str)
        else:
            prompt = CAPTION_REFINE_PROMPT_MULTI.format(names=names_str)

        with self._lock:
            result = self._pipeline.generate(
                prompt,
                images=[image_tensor],
                generation_config=config,
            )
            self._pipeline.finish_chat()

        caption = str(result).strip()
        return CaptionResult(caption=caption, confidence=0.85)

    def _gen_config(self):
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 100
        config.do_sample = False
        return config
