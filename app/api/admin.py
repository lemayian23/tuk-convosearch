"""
Admin routes: corpus management, query logs, statistics.

Location: app/api/admin.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.api.schemas import DocumentOut, SessionOut
from app.config import get_settings
from app.ingestion.documents import SUPPORTED_EXTENSIONS, save_upload
from app.ingestion.index import get_index
from app.services import database
from app.services.auth import require_admin
from app.services.ollama import model_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/documents", response_model=List[DocumentOut])
async def list_documents(include_inactive: bool = False) -> List[Dict[str, Any]]:
    return database.list_documents(active_only=not include_inactive)


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Store the file in the corpus directory and index it on its own. Adding a
    document embeds only that document; the rest of the corpus is untouched.
    """
    settings = get_settings()
    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File is larger than {settings.max_upload_mb} MB.",
        )

    if Path(file.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        saved = await run_in_threadpool(
            save_upload, content, file.filename or "document", settings.corpus_dir
        )
        result = await run_in_threadpool(
            get_index().ingest_path, saved, int(admin["sub"])
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingest failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Could not index this file: {exc}")

    if result["status"] == "empty":
        raise HTTPException(
            status_code=422,
            detail="No readable text found in this file. A scanned PDF needs OCR first.",
        )
    return result


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: int) -> None:
    document = database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    removed = await run_in_threadpool(get_index().remove_document, document_id)
    logger.info(
        "Deactivated document %s (%s), removed %d chunks", document_id, document["filename"], removed
    )

    # The file itself goes with it, otherwise the next full re-ingest would
    # quietly bring the deleted document back.
    target = get_settings().corpus_dir / document["filename"]
    try:
        if target.exists():
            target.unlink()
    except OSError:
        logger.warning("Could not delete corpus file %s", target)
    return None


@router.post("/documents/ingest-all", status_code=status.HTTP_202_ACCEPTED)
async def ingest_corpus(background: BackgroundTasks) -> Dict[str, Any]:
    """
    Re-ingest every supported file in the corpus directory. Runs in the
    background so a large upload set cannot time out the request; poll
    /api/admin/index for progress.
    """
    background.add_task(_ingest_corpus_sync)
    return {"started": True, "corpus_dir": str(get_settings().corpus_dir)}


def _ingest_corpus_sync() -> Dict[str, Any]:
    settings = get_settings()
    index = get_index()
    summary = {"indexed": 0, "unchanged": 0, "failed": []}
    files = sorted(
        p for p in settings.corpus_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for path in files:
        try:
            result = index.ingest_path(path)
            if result["status"] == "indexed":
                summary["indexed"] += 1
            elif result["status"] == "unchanged":
                summary["unchanged"] += 1
        except Exception as exc:
            logger.exception("Failed to ingest %s", path)
            summary["failed"].append({"file": path.name, "error": str(exc)})
    logger.info("Corpus ingest finished: %s", summary)
    return summary


@router.get("/index")
async def index_status() -> Dict[str, Any]:
    settings = get_settings()
    corpus_files = [
        p.name
        for p in settings.corpus_dir.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return {
        "index": get_index().get_stats(),
        "corpus_dir": str(settings.corpus_dir),
        "corpus_files": corpus_files,
        "documents": len(database.list_documents()),
        "ollama": await run_in_threadpool(model_status),
    }


@router.get("/logs")
async def recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 500))
    return database.get_recent_queries(limit=limit)


@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Every conversation on the box, titles included. Admin-only: students have
    no login, so this list would be a directory of other people's transcripts.
    """
    limit = max(1, min(limit, 500))
    return database.list_sessions(limit=limit)


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    payload = database.get_query_stats()
    payload["index"] = get_index().get_stats()
    payload["documents"] = len(database.list_documents())
    payload["sessions"] = len(database.list_sessions(limit=1000))
    return payload
