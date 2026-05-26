"""
TUK-ConvoSearch - Main Application Entry Point
Proposal-Compliant Version with FAISS Vector Database
Location: backend/app/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat_proposal import router as chat_router

# Create FastAPI instance
app = FastAPI(
    title="TUK-ConvoSearch",
    description="Retrieval-Augmented Generation (RAG) AI Assistant for Technical University of Kenya",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include chat router
app.include_router(chat_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to TUK-ConvoSearch API!",
        "status": "running",
        "version": "2.0.0",
        "vector_db": "FAISS",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "stats": "/api/stats",
            "docs": "/docs"
        }
    }

# Simple health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)