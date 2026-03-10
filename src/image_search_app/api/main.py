from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from image_search_app.agent.graph import SearchAgent
from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
from image_search_app.ingestion.pipeline import IngestionPipeline
from image_search_app.schemas import (
    DetectedFace,
    DualListSearchResponse,
    ImageSearchRequest,
    IngestRequest,
    IngestResponse,
    TextSearchRequest,
    UpdateFacesRequest,
    UpdateFacesResponse,
)

app = FastAPI(title="Agentic Image Search")

agent = SearchAgent()
ingestion = IngestionPipeline()

# React build output (npm run build in frontend/)
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"


@app.on_event("startup")
def startup() -> None:
    create_all()
    if FRONTEND_ASSETS.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.get("/")
@app.get("/index.html")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIST / "index.html")


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
    try:
        image_id = ingestion.ingest(request.image_path)
    except Exception:
        return IngestResponse(
            image_id="00000000-0000-0000-0000-000000000000",
            file_path=request.image_path,
            ingestion_status="failed",
        )

    with get_session() as session:
        record = session.query(ImageRecord).filter_by(image_id=image_id).first()
        faces: list[DetectedFace] = []
        if record:
            for p in record.people:
                bbox = list(map(int, p.bbox.split(",")))
                faces.append(DetectedFace(face_id=p.face_id, bbox=bbox, confidence=p.confidence, name=p.name))

    return IngestResponse(
        image_id=image_id,
        file_path=request.image_path,
        ingestion_status="ready",
        faces=faces,
    )


@app.put("/images/{image_id}/faces", response_model=UpdateFacesResponse)
def update_faces(image_id: str, request: UpdateFacesRequest) -> UpdateFacesResponse:
    with get_session() as session:
        record = session.query(ImageRecord).filter_by(image_id=image_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Image not found")

        updated = 0
        for entry in request.faces:
            person = session.query(PersonRecord).filter_by(image_id=image_id, face_id=entry.face_id).first()
            if person:
                person.name = entry.name
                person.source = "user_tag"
                updated += 1

        session.commit()

    return UpdateFacesResponse(image_id=image_id, updated=updated)


@app.get("/browse-images")
def browse_images() -> dict[str, list[str]]:
    """Open a native file dialog and return the selected image paths."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"), ("All files", "*.*")],
    )
    root.destroy()
    return {"paths": list(paths)}


@app.post("/search/text", response_model=DualListSearchResponse)
def search_text(request: TextSearchRequest) -> DualListSearchResponse:
    return agent.search_text(request.query, top_k=request.top_k)


@app.post("/search/image", response_model=DualListSearchResponse)
def search_image(request: ImageSearchRequest) -> DualListSearchResponse:
    return agent.search_image(image_path=request.image_path, query=request.query, top_k=request.top_k)
