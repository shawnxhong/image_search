from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import json
import logging

from image_search_app.agent.graph import SearchAgent
from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session
from image_search_app.ingestion.pipeline import IngestionPipeline
from image_search_app.vector.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
from image_search_app.schemas import (
    DetectedFace,
    DismissFaceResponse,
    DualListSearchResponse,
    FaceCandidate,
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
face_store = ChromaStore()

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
    except FileNotFoundError as exc:
        logger.warning("Ingest file not found: %s", exc)
        return IngestResponse(
            image_id="00000000-0000-0000-0000-000000000000",
            file_path=request.image_path,
            ingestion_status="failed",
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Ingest failed for %s", request.image_path)
        return IngestResponse(
            image_id="00000000-0000-0000-0000-000000000000",
            file_path=request.image_path,
            ingestion_status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    with get_session() as session:
        record = session.query(ImageRecord).filter_by(image_id=image_id).first()
        faces: list[DetectedFace] = []
        caption = None
        capture_timestamp = None
        lat = None
        lon = None
        status = "ready"
        if record:
            caption = record.caption
            capture_timestamp = record.capture_timestamp
            lat = record.lat
            lon = record.lon
            status = record.ingestion_status
            for p in record.people:
                bbox = list(map(int, p.bbox.split(",")))
                candidates: list[FaceCandidate] = []
                if p.candidates:
                    try:
                        for c in json.loads(p.candidates):
                            candidates.append(FaceCandidate(name=c["name"], distance=c["distance"]))
                    except Exception:
                        pass
                faces.append(DetectedFace(
                    face_id=p.face_id, bbox=bbox, confidence=p.confidence,
                    name=p.name, dismissed=p.dismissed, candidates=candidates,
                ))

    return IngestResponse(
        image_id=image_id,
        file_path=request.image_path,
        ingestion_status=status,
        caption=caption,
        capture_timestamp=capture_timestamp,
        lat=lat,
        lon=lon,
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

                # Store face embedding as ground truth for this person
                if person.descriptor:
                    try:
                        descriptor = json.loads(person.descriptor)
                        face_store.upsert_face_identity(person.face_id, descriptor, entry.name)
                        logger.info("Stored face identity for '%s' (face_id=%s)", entry.name, person.face_id)
                    except Exception as exc:
                        logger.warning("Failed to store face identity: %s", exc)

        # Check if all non-dismissed faces now have names
        all_labeled = all(
            p.name or p.dismissed
            for p in record.people
        )
        if all_labeled and record.ingestion_status == "pending_labels":
            record.ingestion_status = "ready"

        session.commit()

    return UpdateFacesResponse(image_id=image_id, updated=updated)


@app.put("/images/{image_id}/faces/{face_id}/dismiss", response_model=DismissFaceResponse)
def dismiss_face(image_id: str, face_id: str) -> DismissFaceResponse:
    with get_session() as session:
        person = session.query(PersonRecord).filter_by(image_id=image_id, face_id=face_id).first()
        if not person:
            raise HTTPException(status_code=404, detail="Face not found")
        person.dismissed = True

        # Check if all faces are now resolved (named or dismissed)
        record = session.query(ImageRecord).filter_by(image_id=image_id).first()
        if record and record.ingestion_status == "pending_labels":
            all_resolved = all(p.name or p.dismissed for p in record.people)
            if all_resolved:
                record.ingestion_status = "ready"

        session.commit()

    return DismissFaceResponse(image_id=image_id, face_id=face_id, dismissed=True)


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
