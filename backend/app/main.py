"""
TUK-ConvoSearch - Main Application Entry Point
Location: backend/app/main.py
Version: 3.0.0 — FAISS + SQLite + JWT Auth + Admin Panel
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat_proposal import router as chat_router
from app.api.admin import router as admin_router
from app.services import database
from app.services.auth import hash_password

app = FastAPI(
    title="TUK-ConvoSearch",
    description="RAG AI Assistant for The Technical University of Kenya — with Admin Panel",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins so the vanilla JS student UI and the React
# admin panel (different ports during dev) can both reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router)
app.include_router(admin_router)


@app.on_event("startup")
async def startup_event():
    # Ensure SQLite schema exists
    database.init_db()

    # Seed a default admin account if no users exist yet.
    # On first boot after adding auth, this creates the admin@tuk.ac.ke account.
    # Change the password immediately via the admin panel after first login.
    if database.user_count() == 0:
        database.create_user(
            full_name="Denis Kirionki",
            email="admin@tuk.ac.ke",
            password_hash=hash_password("Admin2026!"),
            role="admin",
        )
        print("✓ Default admin account created: admin@tuk.ac.ke / Admin2026!")
        print("  ⚠  Change this password after first login.")


@app.get("/")
async def root():
    return {
        "message": "Welcome to TUK-ConvoSearch API!",
        "status": "running",
        "version": "3.0.0",
        "vector_db": "FAISS",
        "metadata_db": "SQLite",
        "auth": "JWT",
        "endpoints": {
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream",
            "login": "/api/auth/login",
            "admin_documents": "/api/admin/documents",
            "admin_logs": "/api/admin/logs",
            "admin_stats": "/api/admin/stats",
            "health": "/api/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)