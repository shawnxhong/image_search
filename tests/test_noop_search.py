from pathlib import Path

from image_search_app.agent.graph import SearchAgent
from image_search_app.vector.chroma_store import ChromaStore
from image_search_app.vector.embeddings import EmbeddingService
from image_search_app.vector.retrievers import RetrieverService


def test_noop_text_search_returns_structured_empty_lists(tmp_path: Path):
    store = ChromaStore(
        path=str(tmp_path / "chroma"),
        caption_collection="test_caption_embeddings",
        image_collection="test_image_embeddings",
    )
    retriever = RetrieverService(store=store, embeddings=EmbeddingService())
    agent = SearchAgent(retriever=retriever)

    response = agent.search_text("find photos with tom in it", top_k=10)

    assert response.solid_results == []
    assert response.soft_results == []
