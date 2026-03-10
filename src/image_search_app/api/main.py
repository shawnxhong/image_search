from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from image_search_app.agent.graph import SearchAgent
from image_search_app.db import create_all
from image_search_app.ingestion.pipeline import IngestionPipeline
from image_search_app.schemas import (
    DualListSearchResponse,
    ImageSearchRequest,
    IngestRequest,
    IngestResponse,
    TextSearchRequest,
)

app = FastAPI(title="Agentic Image Search")

agent = SearchAgent()
ingestion = IngestionPipeline()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    create_all()


@app.get("/")
@app.get("/index.html")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/image-preview")
def image_preview(path: str = Query(..., description="Absolute or repo-relative image path")) -> FileResponse:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidate)


@app.post("/ingest", response_model=IngestResponse)
def ingest_image(request: IngestRequest) -> IngestResponse:
    image_id = ingestion.ingest(request.image_path)
    return IngestResponse(image_id=image_id, ingestion_status="ready")


@app.post("/search/text", response_model=DualListSearchResponse)
def search_text(request: TextSearchRequest) -> DualListSearchResponse:
    return agent.search_text(request.query, top_k=request.top_k)


@app.post("/search/image", response_model=DualListSearchResponse)
def search_image(request: ImageSearchRequest) -> DualListSearchResponse:
    return agent.search_image(image_path=request.image_path, query=request.query, top_k=request.top_k)
