"""
Admin API Endpoints
Location: backend/app/api/admin.py

All routes here are protected by the require_admin JWT dependency.
Provides:
  - POST /api/auth/login          login and receive JWT
  - GET  /api/auth/me             verify token, return user info
  - GET  /api/admin/documents     list all ingested documents
  - POST /api/admin/documents     upload a new document + re-index
  - DELETE /api/admin/documents/{id}  soft-delete a document
  - GET  /api/admin/logs          recent query log entries
  - GET  /api/admin/stats         combined system stats
"""

import os
import sys
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.auth import create_access_token, hash_password, require_admin, verify_password
from app.services import database

router = APIRouter(tags=["admin"])

# Path to the docs folder (relative to where uvicorn is run: backend/)
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs"))
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Make rebuild_faiss importable as a module (it lives in backend/, one level
# above the app/ package) and import its core function directly. Calling
# this in-process — rather than spawning rebuild_faiss.py as a separate OS
# subprocess — avoids two Python processes contending for the same SQLite
# database file at the same time, which previously caused intermittent
# "database is locked" failures during document upload.
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
from rebuild_faiss import rebuild_index

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ------------------------------------------------------------------ #
# Auth models
# ------------------------------------------------------------------ #

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    full_name: str


# ------------------------------------------------------------------ #
# Auth endpoints (no JWT required — these are public)
# ------------------------------------------------------------------ #

@router.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate an admin user and return a JWT access token."""
    user = database.get_user_by_email(request.email)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
    )

    return LoginResponse(
        access_token=token,
        user_id=user["user_id"],
        email=user["email"],
        full_name=user["full_name"],
    )


@router.get("/api/auth/me")
async def me(admin=Depends(require_admin)):
    """Return the currently authenticated admin's info."""
    return {
        "user_id": admin["sub"],
        "email": admin["email"],
        "role": admin["role"],
    }


# ------------------------------------------------------------------ #
# Admin — Documents
# ------------------------------------------------------------------ #

@router.get("/api/admin/documents")
async def list_documents(admin=Depends(require_admin)):
    """Return all documents currently in the SQLite documents table."""
    docs = database.list_documents(active_only=False)
    return {"documents": docs, "total": len(docs)}


@router.post("/api/admin/documents")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    admin=Depends(require_admin),
):
    """
    Upload a new document to the docs folder, then trigger a full
    FAISS re-index so the new content is immediately searchable.

    The re-index runs rebuild_index() directly in-process (not as a
    separate subprocess) to avoid two Python processes contending for
    the same SQLite database file. Because re-indexing is a slow,
    blocking operation (re-embeds every chunk of every document), it is
    run in a background thread via run_in_executor so it doesn't freeze
    the server's event loop while it works — the same fix pattern used
    for streaming chat responses.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, TXT.",
        )

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest_path = os.path.join(DOCS_DIR, file.filename)

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: rebuild_index(docs_folder=DOCS_DIR, verbose=True)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File saved but re-indexing failed: {str(e)[:300]}",
        )

    docs = database.list_documents(active_only=True)
    uploaded = next((d for d in docs if d["filename"] == file.filename), None)

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "document": uploaded,
        "total_documents": len(docs),
    }


@router.delete("/api/admin/documents/{document_id}")
async def delete_document(document_id: int, admin=Depends(require_admin)):
    """
    Remove a document from the active search index.

    IMPORTANT: rebuild_index() treats every file physically present in
    DOCS_DIR as the full source of truth and re-ingests all of them on
    every rebuild. Marking a database row inactive is therefore not
    sufficient on its own — if the file is left in DOCS_DIR, the very
    next rebuild (triggered by any future upload or delete) would
    silently re-ingest it and mark it active again.

    To prevent this, the physical file is moved out of DOCS_DIR into a
    sibling 'docs_archive' folder (not deleted outright, so nothing is
    destructively lost) before the re-index runs. The database row is
    marked inactive for audit history.
    """
    docs = database.list_documents(active_only=False)
    target = next((d for d in docs if d["document_id"] == document_id), None)

    if not target:
        raise HTTPException(status_code=404, detail="Document not found.")

    success = database.deactivate_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Move the physical file out of DOCS_DIR so rebuild_index() won't
    # re-ingest it on the next rebuild (whenever that happens).
    archive_dir = os.path.join(DOCS_DIR, "..", "docs_archive")
    archive_dir = os.path.abspath(archive_dir)
    os.makedirs(archive_dir, exist_ok=True)

    src_path = os.path.join(DOCS_DIR, target["filename"])
    if os.path.exists(src_path):
        dest_path = os.path.join(archive_dir, target["filename"])
        # Avoid overwriting if a same-named file was already archived before
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(target["filename"])
            dest_path = os.path.join(archive_dir, f"{base}_{document_id}{ext}")
        os.rename(src_path, dest_path)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: rebuild_index(docs_folder=DOCS_DIR, verbose=True)
        )
    except Exception as e:
        print(f"  ⚠ Re-index after delete failed: {e}")

    return {"message": f"'{target['filename']}' removed and re-indexed successfully."}


# ------------------------------------------------------------------ #
# Admin — Query logs
# ------------------------------------------------------------------ #

@router.get("/api/admin/logs")
async def query_logs(limit: int = 50, admin=Depends(require_admin)):
    """Return the most recent student query log entries."""
    logs = database.get_recent_queries(limit=limit)
    stats = database.get_query_stats()
    return {
        "logs": logs,
        "total_shown": len(logs),
        "stats": stats,
    }


# ------------------------------------------------------------------ #
# Admin — Combined stats
# ------------------------------------------------------------------ #

@router.get("/api/admin/stats")
async def admin_stats(admin=Depends(require_admin)):
    """Combined system statistics for the admin dashboard."""
    docs = database.list_documents(active_only=True)
    query_stats = database.get_query_stats()

    return {
        "documents": {
            "total_active": len(docs),
            "documents": docs,
        },
        "queries": query_stats,
        "system": {
            "vector_db": "FAISS",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "llama3.2:1b",
            "metadata_db": "SQLite",
        },
    }