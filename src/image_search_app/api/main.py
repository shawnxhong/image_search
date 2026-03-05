from fastapi import FastAPI

from image_search_app.agent.graph import SearchAgent
from image_search_app.db import create_all
from image_search_app.ingestion.pipeline import IngestionPipeline
from image_search_app.schemas import DualListSearchResponse, ImageSearchRequest, TextSearchRequest

app = FastAPI(title="Agentic Image Search")

agent = SearchAgent()
ingestion = IngestionPipeline()


@app.on_event("startup")
def startup() -> None:
    create_all()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_image(image_path: str) -> dict[str, str]:
    image_id = ingestion.ingest(image_path)
    return {"image_id": image_id}


@app.post("/search/text", response_model=DualListSearchResponse)
def search_text(request: TextSearchRequest) -> DualListSearchResponse:
    return agent.search_text(request.query, top_k=request.top_k)


@app.post("/search/image", response_model=DualListSearchResponse)
def search_image(request: ImageSearchRequest) -> DualListSearchResponse:
    return agent.search_image(image_path=request.image_path, query=request.query, top_k=request.top_k)
