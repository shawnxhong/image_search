from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import json
import logging

from image_search_app.agent.graph import SearchAgent
from image_search_app.config import settings
from image_search_app.db import ImageRecord, PersonRecord, create_all, get_session, list_images_paginated
from image_search_app.ingestion.pipeline import IngestionPipeline
from image_search_app.tools.llm import get_llm_service, scan_available_models
from image_search_app.vector.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
from image_search_app.schemas import (
    AgentStep,
    DetectedFace,
    DismissFaceResponse,
    DualListSearchResponse,
    FaceCandidate,
    ImageSearchRequest,
    IngestRequest,
    IngestResponse,
    LibraryImageItem,
    LibraryResponse,
    LLMAvailableResponse,
    LLMLoadRequest,
    LLMStatusResponse,
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


@app.get("/library", response_model=LibraryResponse)
def library(
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> LibraryResponse:
    """List all images sorted by date (most recent first), with cursor pagination."""
    images, total = list_images_paginated(limit=limit, cursor=cursor)
    items = [
        LibraryImageItem(
            image_id=img.image_id,
            file_path=img.file_path,
            caption=img.caption,
            capture_timestamp=img.capture_timestamp,
            country=img.country,
            state=img.state,
            city=img.city,
            ingestion_status=img.ingestion_status,
        )
        for img in images
    ]
    next_cursor = items[-1].image_id if len(items) == limit else None
    return LibraryResponse(images=items, total=total, next_cursor=str(next_cursor) if next_cursor else None)


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
    should_refine = False
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
            record.ingestion_status = "refining_caption"
            should_refine = True

        session.commit()

    # Refine caption with person names after all faces are resolved
    new_caption = None
    final_status = None
    if should_refine:
        try:
            new_caption = ingestion.refine_after_labeling(image_id)
        except Exception as exc:
            logger.exception("Caption refinement failed for %s: %s", image_id, exc)
        # Read final status after refinement
        with get_session() as session:
            rec = session.query(ImageRecord).filter_by(image_id=image_id).first()
            if rec:
                final_status = rec.ingestion_status

    return UpdateFacesResponse(image_id=image_id, updated=updated, caption=new_caption, ingestion_status=final_status)


@app.put("/images/{image_id}/faces/{face_id}/dismiss", response_model=DismissFaceResponse)
def dismiss_face(image_id: str, face_id: str) -> DismissFaceResponse:
    should_refine = False
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
                record.ingestion_status = "refining_caption"
                should_refine = True

        session.commit()

    new_caption = None
    final_status = None
    if should_refine:
        try:
            new_caption = ingestion.refine_after_labeling(image_id)
        except Exception as exc:
            logger.exception("Caption refinement failed for %s: %s", image_id, exc)
        with get_session() as session:
            rec = session.query(ImageRecord).filter_by(image_id=image_id).first()
            if rec:
                final_status = rec.ingestion_status

    return DismissFaceResponse(
        image_id=image_id, face_id=face_id, dismissed=True,
        caption=new_caption, ingestion_status=final_status,
    )


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


@app.get("/llm/status", response_model=LLMStatusResponse)
def llm_status() -> LLMStatusResponse:
    llm = get_llm_service()
    return LLMStatusResponse(**llm.status())


@app.get("/llm/available", response_model=LLMAvailableResponse)
def llm_available() -> LLMAvailableResponse:
    models = scan_available_models()
    return LLMAvailableResponse(models=models, devices=["CPU", "GPU"])


@app.post("/llm/load", response_model=LLMStatusResponse)
def llm_load(request: LLMLoadRequest) -> LLMStatusResponse:
    from pathlib import Path as P

    base = P(settings.llm_models_dir)
    if not base.is_absolute():
        base = P(__file__).resolve().parent.parent.parent.parent / base
    model_path = str(base / request.model_name)
    llm = get_llm_service()
    try:
        llm.load(model_path=model_path, device=request.device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM: {exc}")
    return LLMStatusResponse(**llm.status())


@app.post("/llm/unload", response_model=LLMStatusResponse)
def llm_unload() -> LLMStatusResponse:
    llm = get_llm_service()
    llm.unload()
    return LLMStatusResponse(**llm.status())


@app.get("/models/status")
def models_status() -> dict:
    """Return load status of all model services."""
    llm = get_llm_service()
    return {
        "llm": llm.status(),
        "vlm": ingestion.captioner.status(),
        "embeddings": ingestion.embeddings.status(),
        "face_detection": ingestion.face_recognizer.status(),
    }


@app.post("/models/{name}/load")
def model_load(name: str) -> dict:
    """Load a specific model service."""
    if name == "llm":
        llm = get_llm_service()
        try:
            llm.load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return llm.status()
    elif name == "vlm":
        try:
            ingestion.captioner._load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return ingestion.captioner.status()
    elif name == "embeddings":
        try:
            ingestion.embeddings._load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return ingestion.embeddings.status()
    elif name == "face_detection":
        try:
            ingestion.face_recognizer._load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return ingestion.face_recognizer.status()
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")


@app.post("/models/{name}/unload")
def model_unload(name: str) -> dict:
    """Unload a specific model service."""
    if name == "llm":
        llm = get_llm_service()
        llm.unload()
        return llm.status()
    elif name == "vlm":
        ingestion.captioner.unload()
        return ingestion.captioner.status()
    elif name == "embeddings":
        ingestion.embeddings.unload()
        return ingestion.embeddings.status()
    elif name == "face_detection":
        ingestion.face_recognizer.unload()
        return ingestion.face_recognizer.status()
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")


@app.post("/search/text", response_model=DualListSearchResponse)
def search_text(request: TextSearchRequest) -> DualListSearchResponse:
    return agent.search_text(request.query, top_k=request.top_k)


@app.post("/search/text/stream")
def search_text_stream(request: TextSearchRequest) -> StreamingResponse:
    def event_generator():
        for item in agent.search_text_stream(request.query, top_k=request.top_k):
            if isinstance(item, AgentStep):
                yield f"event: step\ndata: {item.model_dump_json()}\n\n"
            elif isinstance(item, DualListSearchResponse):
                yield f"event: result\ndata: {item.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/search/image", response_model=DualListSearchResponse)
def search_image(request: ImageSearchRequest) -> DualListSearchResponse:
    return agent.search_image(image_path=request.image_path, query=request.query, top_k=request.top_k)
