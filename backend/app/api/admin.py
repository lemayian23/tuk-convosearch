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
import shutil
import subprocess
import sys
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.auth import create_access_token, hash_password, require_admin, verify_password
from app.services import database

router = APIRouter(tags=["admin"])

# Path to the docs folder (relative to where uvicorn is run: backend/)
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs"))
REBUILD_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rebuild_faiss.py"))

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
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, TXT.",
        )

    # Save file to docs directory
    os.makedirs(DOCS_DIR, exist_ok=True)
    dest_path = os.path.join(DOCS_DIR, file.filename)

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Trigger rebuild_faiss.py as a subprocess so it runs in the same venv
    try:
        result = subprocess.run(
            [sys.executable, REBUILD_SCRIPT],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(REBUILD_SCRIPT),
            timeout=120,
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
    except Exception as e:
        # File was saved but indexing failed — report it clearly
        raise HTTPException(
            status_code=500,
            detail=f"File saved but re-indexing failed: {str(e)[:300]}",
        )

    # Return updated document list
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
    Soft-delete a document from the SQLite documents table (marks it
    inactive). The physical file is retained. A full re-index is triggered
    so FAISS no longer returns chunks from the deleted document.
    """
    success = database.deactivate_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Re-index so FAISS reflects the removal
    try:
        subprocess.run(
            [sys.executable, REBUILD_SCRIPT],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(REBUILD_SCRIPT),
            timeout=120,
        )
    except Exception as e:
        # Non-fatal for the delete operation itself — log and continue
        print(f"  ⚠ Re-index after delete failed: {e}")

    return {"message": f"Document {document_id} deactivated successfully."}


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