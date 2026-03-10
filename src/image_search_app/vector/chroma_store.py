from __future__ import annotations

import chromadb

from image_search_app.config import settings


class ChromaStore:
    def __init__(
        self,
        path: str | None = None,
        caption_collection: str | None = None,
        image_collection: str | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=path or settings.chroma_path)
        self.caption_collection = self.client.get_or_create_collection(caption_collection or settings.caption_collection)
        self.image_collection = self.client.get_or_create_collection(image_collection or settings.image_collection)

    def upsert_caption_embedding(self, image_id: str, embedding: list[float], caption: str | None) -> None:
        self.caption_collection.upsert(ids=[image_id], embeddings=[embedding], documents=[caption or ""])

    def upsert_image_embedding(self, image_id: str, embedding: list[float]) -> None:
        self.image_collection.upsert(ids=[image_id], embeddings=[embedding])

    def query_caption(self, embedding: list[float], top_k: int) -> tuple[list[str], list[float]]:
        count = self.caption_collection.count()
        if count == 0:
            return [], []
        result = self.caption_collection.query(query_embeddings=[embedding], n_results=min(top_k, count))
        return result.get("ids", [[]])[0], result.get("distances", [[]])[0]

    def query_image(self, embedding: list[float], top_k: int) -> tuple[list[str], list[float]]:
        count = self.image_collection.count()
        if count == 0:
            return [], []
        result = self.image_collection.query(query_embeddings=[embedding], n_results=min(top_k, count))
        return result.get("ids", [[]])[0], result.get("distances", [[]])[0]
