"""
Chat API Endpoints - Proposal-Compliant Version
Uses FAISS vector database
Location: backend/app/api/chat_proposal.py
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
    """Send a message and get a complete (non-streamed) response from TUK-ConvoSearch"""
    try:
        result = rag_service.answer_question(
            question=request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            answer=str(result['answer']),
            sources=result['sources'],
            chunks_found=int(result['chunks_found']),
            response_time=float(result['response_time']) if result.get('response_time') else None,
            vector_db=str(result.get('vector_db', 'FAISS'))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream the response token-by-token using Server-Sent Events (SSE).

    Each event is a JSON object on a single line prefixed with 'data: ',
    one of:
      {"type": "sources", "sources": [...], "chunks_found": N}
      {"type": "token", "content": "..."}
      {"type": "done", "response_time": float}

    The frontend (see index.html sendMessageStream) reads this with the
    Fetch API's ReadableStream reader and appends each token as it arrives.
    """

    def event_generator():
        try:
            for event in rag_service.stream_answer(
                question=request.message,
                session_id=request.session_id
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {"type": "token", "content": f"\n[Error: {e}]"}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'response_time': 0.0})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering, if ever deployed behind nginx
        }
    )


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