"""
TUK-Convosearch - Main Application Entry Point
Proposal-Compliant Version with FAISS Vector Database + SQLite metadata layer
Location: backend/app/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat_proposal import router as chat_router
from app.services import database

# Create FastAPI instance
app = FastApi(
    title="TUK-Convosearch",
    description="Retrieval-Augmented Generation (RAG) AI Assistant for Technical University of Kenya",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=[*],
    allow_headers=[*],
)

# Include chat router
app.include_router(chat_router)


@app.on_event("startup")
async def startup_event():
    """Ensure the SQLite schema exists before tha app starts serving requests. """
    database.init_db()


# Root endpoint
@app.get("/")
async def root():
    return{ 
        "message": "Welcome to TUK-Convosearch API!",
        "status": "running",
        "version": "2.1.0",
        "vector_db": "SQLite",
        "endpoints":{
            "chat": "/api/chat",
            "chat_stream": "api/chat/stream",
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