from __future__ import annotations

from dataclasses import dataclass

from image_search_app.config import settings
from image_search_app.db import ImageRecord, get_session
from image_search_app.schemas import DualListSearchResponse, MatchExplanation, SearchResultItem
from image_search_app.tools.filters import apply_hard_filters
from image_search_app.tools.intent_parser import IntentParser
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService
from image_search_app.vector.retrievers import RetrieverService


@dataclass
class RankedCandidate:
    image_id: str
    score: float


class SearchAgent:
    """Starter orchestration class aligned with the planned graph nodes.

    This can be migrated to a full LangGraph StateGraph once tools stabilize.
    """

    def __init__(self) -> None:
        store = ChromaStore()
        self.retriever = RetrieverService(store=store, embeddings=EmbeddingService())
        self.intent_parser = IntentParser()

    def search_text(self, query: str, top_k: int | None = None) -> DualListSearchResponse:
        k = top_k or settings.default_top_k
        intent = self.intent_parser.parse_text_query(query)
        ids, distances = self.retriever.text_semantic_search(query=query, top_k=k)
        candidates = [RankedCandidate(image_id=i, score=1.0 - float(d or 0.0)) for i, d in zip(ids, distances)]
        return self._assemble(candidates, intent)

    def search_image(self, image_path: str, query: str | None, top_k: int | None = None) -> DualListSearchResponse:
        k = top_k or settings.default_top_k
        intent = self.intent_parser.parse_image_query(query)
        ids, distances = self.retriever.image_semantic_search(image_path=image_path, top_k=k)
        candidates = [RankedCandidate(image_id=i, score=1.0 - float(d or 0.0)) for i, d in zip(ids, distances)]
        return self._assemble(candidates, intent)

    def _assemble(self, candidates: list[RankedCandidate], intent) -> DualListSearchResponse:
        with get_session() as session:
            records = {
                rec.image_id: rec
                for rec in session.query(ImageRecord).filter(ImageRecord.image_id.in_([c.image_id for c in candidates]))
            }

        solid: list[SearchResultItem] = []
        soft: list[SearchResultItem] = []

        for candidate in sorted(candidates, key=lambda x: x.score, reverse=True):
            record = records.get(candidate.image_id)
            if record is None:
                continue
            outcome = apply_hard_filters(record, intent)
            explanation = MatchExplanation(
                image_id=record.image_id,
                reason="Matched hard filters and semantic relevance" if outcome.accepted else "High semantic relevance but failed hard filters",
                matched_constraints=outcome.matched_constraints,
                missing_metadata=outcome.missing_metadata,
            )
            item = SearchResultItem(
                image_id=record.image_id,
                file_path=record.file_path,
                score=candidate.score,
                explanation=explanation,
            )
            if outcome.accepted:
                solid.append(item)
            else:
                soft.append(item)

        # Keep soft as semantic list; optionally de-duplicate from solid by construction.
        return DualListSearchResponse(solid_results=solid, soft_results=soft)
