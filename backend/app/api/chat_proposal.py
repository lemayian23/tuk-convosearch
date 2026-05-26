"""
Chat API Endpoints - Proposal-Compliant Version
Uses FAISS vector database
Location: backend/app/api/chat_proposal.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.rag_service_proposal import RAGServiceProposal

# Create router - THIS MUST BE NAMED 'router'
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize RAG service once
rag_service = RAGServiceProposal()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    chunks_found: int
    response_time: Optional[float] = None
    vector_db: str = "FAISS"

class StatsResponse(BaseModel):
    vector_db: Dict[str, Any]
    model: str
    cache_size: int
    active_sessions: int

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response from TUK-ConvoSearch"""
    try:
        result = rag_service.answer_question(
            question=request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            answer=str(result['answer']),  # Ensure string
            sources=result['sources'],
            chunks_found=int(result['chunks_found']),  # Ensure int
            response_time=float(result['response_time']) if result.get('response_time') else None,
            vector_db=str(result.get('vector_db', 'FAISS'))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    """Check if the service is healthy"""
    stats = rag_service.get_stats()
    return {
        "status": "healthy",
        "model": rag_service.model_name,
        "vector_db": "FAISS",
        "total_chunks": stats['vector_db']['total_chunks']
    }

@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Get system statistics"""
    return rag_service.get_stats()

@router.post("/chat/clear")
async def clear_history(session_id: str = None):
    """Clear conversation history"""
    rag_service.clear_history(session_id)
    return {"message": "History cleared", "session_id": session_id}

@router.post("/cache/clear")
async def clear_cache():
    """Clear response cache"""
    rag_service.clear_cache()
    return {"message": "Cache cleared"}