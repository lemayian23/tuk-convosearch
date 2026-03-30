"""
TUK-ConvoSearch - Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat  # Import chat router

# Create FastAPI instance
app = FastAPI(
    title="TUK-ConvoSearch",
    description="Retrieval-Augmented Generation AI Assistant for Technical University of Kenya",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to TUK-ConvoSearch API!",
        "status": "running",
        "version": "1.0.0"
    }

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)