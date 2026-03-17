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
        self.caption_collection = self.client.get_or_create_collection(
            caption_collection or settings.caption_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            image_collection or settings.image_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.face_collection = self.client.get_or_create_collection(
            "face_identities",
            metadata={"hnsw:space": "cosine"},
        )

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

    # --- Face identity methods ---

    def upsert_face_identity(self, face_id: str, descriptor: list[float], name: str) -> None:
        """Store a labeled face descriptor. ID is face_id, metadata carries the name."""
        self.face_collection.upsert(
            ids=[face_id],
            embeddings=[descriptor],
            metadatas=[{"name": name}],
        )

    def match_face(self, descriptor: list[float], threshold: float = 0.5) -> str | None:
        """Find the closest known face. Returns the name if cosine distance < threshold, else None."""
        candidates = self.match_face_candidates(descriptor, top_k=1, threshold=threshold)
        if candidates:
            return candidates[0][0]
        return None

    def match_face_candidates(
        self, descriptor: list[float], top_k: int = 3, threshold: float = 0.8
    ) -> list[tuple[str, float]]:
        """Return top-K candidate matches as [(name, distance), ...].

        Only includes candidates within the threshold. Deduplicates by name
        (keeps the closest distance for each unique name).
        Cosine distance: 0 = identical, 2 = opposite.
        """
        count = self.face_collection.count()
        if count == 0:
            return []
        # Query more than top_k to allow deduplication by name
        n = min(count, top_k * 3)
        result = self.face_collection.query(query_embeddings=[descriptor], n_results=n)
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        # Deduplicate: keep closest distance per unique name
        best_by_name: dict[str, float] = {}
        for dist, meta in zip(distances, metadatas):
            if dist >= threshold:
                continue
            name = meta.get("name", "")
            if not name:
                continue
            if name not in best_by_name or dist < best_by_name[name]:
                best_by_name[name] = dist

        # Sort by distance and return top_k
        candidates = sorted(best_by_name.items(), key=lambda x: x[1])
        return candidates[:top_k]
