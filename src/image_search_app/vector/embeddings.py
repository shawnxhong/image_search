from __future__ import annotations


class EmbeddingService:
    """Embedding adapter stub for caption and image embeddings."""

    def embed_text(self, text: str) -> list[float]:
        # deterministic placeholder for scaffold usage
        return [float((sum(ord(c) for c in text) % 997) / 997.0)]

    def embed_image(self, image_path: str) -> list[float]:
        return [float((sum(ord(c) for c in image_path) % 991) / 991.0)]
