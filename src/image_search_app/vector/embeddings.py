from __future__ import annotations

from PIL import Image

from image_search_app.config import settings


class EmbeddingService:
    """CLIP-based embedding service for text and images.

    Uses openai/clip-vit-base-patch32 via sentence-transformers which handles
    both text and image inputs into the same 512-dim vector space.
    """

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.clip_model_name)

    def embed_text(self, text: str) -> list[float]:
        self._load()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_image(self, image_path: str) -> list[float]:
        self._load()
        img = Image.open(image_path).convert("RGB")
        embedding = self._model.encode(img, convert_to_numpy=True)
        return embedding.tolist()
