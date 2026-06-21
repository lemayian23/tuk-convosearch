"""
Chat API Endpoints - Proposal-Compliant Version
Uses FAISS vector database
Location: backend/app/api/chat_proposal.py
"""

import json
import asyncio
import queue
import threading
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

    IMPORTANT IMPLEMENTATION NOTE:
    rag_service.stream_answer() is a SYNCHRONOUS generator — internally it
    calls ollama.chat(..., stream=True), which performs blocking network
    I/O. If we iterate that generator directly inside an `async def` route,
    each blocking call freezes FastAPI's single event loop, so NOTHING else
    can be sent to the client — not even the first byte — until Ollama
    finishes. From the browser this looks exactly like a hung request with
    zero response, even though the backend is actually working underneath.

    The fix: run the synchronous generator in a background thread, and use
    a thread-safe queue.Queue to hand events back to the async event loop
    one at a time. The async side polls the queue without blocking the
    event loop, so SSE bytes can be flushed to the client immediately as
    each event becomes available.
    """

    async def event_generator():
        q: "queue.Queue" = queue.Queue()
        SENTINEL = object()

        def producer():
            """Runs in a separate thread. Safe to block here — this thread
            is not the asyncio event loop, so blocking Ollama calls don't
            freeze the rest of the server."""
            try:
                for event in rag_service.stream_answer(
                    question=request.message,
                    session_id=request.session_id
                ):
                    q.put(event)
            except Exception as e:
                q.put({"type": "token", "content": f"\n[Error: {e}]"})
                q.put({"type": "done", "response_time": 0.0})
            finally:
                q.put(SENTINEL)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            # q.get is blocking, so run it in the default executor to avoid
            # blocking the event loop while waiting for the next event.
            item = await loop.run_in_executor(None, q.get)
            if item is SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
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