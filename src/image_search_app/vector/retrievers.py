from __future__ import annotations

from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService


class RetrieverService:
    def __init__(self, store: ChromaStore, embeddings: EmbeddingService) -> None:
        self.store = store
        self.embeddings = embeddings

    def text_semantic_search(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        query_embedding = self.embeddings.embed_text(query)
        return self.store.query_caption(query_embedding, top_k=top_k)

    def image_semantic_search(self, image_path: str, top_k: int) -> tuple[list[str], list[float]]:
        query_embedding = self.embeddings.embed_image(image_path)
        return self.store.query_image(query_embedding, top_k=top_k)
