from __future__ import annotations

import chromadb

from image_search_app.config import settings


class ChromaStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.caption_collection = self.client.get_or_create_collection(settings.caption_collection)
        self.image_collection = self.client.get_or_create_collection(settings.image_collection)

    def upsert_caption_embedding(self, image_id: str, embedding: list[float], caption: str | None) -> None:
        self.caption_collection.upsert(ids=[image_id], embeddings=[embedding], documents=[caption or ""])

    def upsert_image_embedding(self, image_id: str, embedding: list[float]) -> None:
        self.image_collection.upsert(ids=[image_id], embeddings=[embedding])

    def query_caption(self, embedding: list[float], top_k: int) -> tuple[list[str], list[float]]:
        result = self.caption_collection.query(query_embeddings=[embedding], n_results=top_k)
        return result.get("ids", [[]])[0], result.get("distances", [[]])[0]

    def query_image(self, embedding: list[float], top_k: int) -> tuple[list[str], list[float]]:
        result = self.image_collection.query(query_embeddings=[embedding], n_results=top_k)
        return result.get("ids", [[]])[0], result.get("distances", [[]])[0]
