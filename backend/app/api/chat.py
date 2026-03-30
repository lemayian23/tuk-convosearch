"""
Chat API Endpoints
Location: backend/app/api/chat.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.rag_service import RAGService

# Create router
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize RAG service once
rag_service = RAGService()

# Define request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    chunks_found: int

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response from TUK-ConvoSearch
    """
    try:
        result = rag_service.answer_question(request.message)
        return ChatResponse(
            answer=result['answer'],
            sources=result['sources'],
            chunks_found=result['chunks_found']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "model": rag_service.model_name,
        "chunks_in_db": rag_service.vector_store.get_stats()['total_chunks']
    }

@router.get("/stats")
async def stats():
    """Get statistics about the system"""
    return rag_service.vector_store.get_stats()