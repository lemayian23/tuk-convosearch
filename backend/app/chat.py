"""
Chat API Endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel

# Create router
router = APIRouter(prefix="/api", tags=["chat"])

# Define request model
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Define response model
class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    chunks_found: int = 0

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response"""
    return ChatResponse(
        answer=f"You said: {request.message}",
        sources=[],
        chunks_found=0
    )

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/stats")
async def stats():
    return {"total_chunks": 0}