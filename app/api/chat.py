"""
Chat routes: blocking question, streaming answer, conversation history.

Location: app/api/chat.py
"""

import json
import logging
import uuid
from typing import AsyncIterator, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.deps import enforce_rate_limit
from app.api.schemas import ChatRequest, ChatResponse, MessageOut, SessionOut
from app.config import get_settings
from app.services import database
from app.services.chat import ChatBusyError, ChatTimeoutError, get_chat_service
from app.services.pipelines import PipelineUnavailableError, available_pipelines

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

FALLBACK_ANSWER = (
    "I could not reach the answer service just now. Please try again in a moment."
)


def _resolve_session_id(requested: str | None) -> str:
    """
    Session ids are server-generated 128-bit hex strings. There is no login
    for students, so unguessability is what keeps one student's history out
    of another student's browser; a client-chosen short name would not.
    """
    return requested or uuid.uuid4().hex


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    _: None = Depends(enforce_rate_limit),
) -> ChatResponse:
    settings = get_settings()
    session_id = _resolve_session_id(payload.session_id)
    service = get_chat_service()

    try:
        result = await service.answer(
            question=payload.message[: settings.max_question_chars],
            session_id=session_id,
            pipeline_name=payload.pipeline,
        )
    except ChatBusyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ChatTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if result.error and not result.answer:
        logger.warning("Chat failed for session %s: %s", session_id, result.error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The assistant could not form an answer. Please try again.",
        )

    return ChatResponse(**result.to_dict())


def _sse(event: Dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/api/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    _: None = Depends(enforce_rate_limit),
) -> StreamingResponse:
    settings = get_settings()
    session_id = _resolve_session_id(payload.session_id)
    service = get_chat_service()

    async def event_source() -> AsyncIterator[str]:
        yield _sse({"type": "start", "session_id": session_id})
        try:
            async for event in service.stream(
                question=payload.message[: settings.max_question_chars],
                session_id=session_id,
                pipeline_name=payload.pipeline,
            ):
                if await request.is_disconnected():
                    break
                yield _sse(event)
        except ChatBusyError as exc:
            yield _sse({"type": "error", "message": str(exc), "code": 503})
        except ChatTimeoutError as exc:
            yield _sse({"type": "error", "message": str(exc), "code": 504})
        except PipelineUnavailableError as exc:
            yield _sse({"type": "error", "message": str(exc), "code": 404})
        except Exception:
            logger.exception("Streaming failed for session %s", session_id)
            yield _sse({"type": "error", "message": FALLBACK_ANSWER, "code": 500})
        yield _sse({"type": "end"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/sessions", response_model=list[SessionOut])
async def list_sessions(limit: int = 30, _: None = Depends(enforce_rate_limit)):
    limit = max(1, min(limit, 100))
    return database.list_sessions(limit=limit)


@router.get("/api/sessions/{session_id}/messages", response_model=list[MessageOut])
async def session_messages(session_id: str, _: None = Depends(enforce_rate_limit)):
    _validate_session_id(session_id)
    return database.get_session_messages(session_id)


@router.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, _: None = Depends(enforce_rate_limit)):
    _validate_session_id(session_id)
    database.delete_session(session_id)
    return None


@router.get("/api/config")
async def client_config() -> Dict:
    """What the student UI needs to know at runtime, instead of a rebuild."""
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "pipelines": available_pipelines(),
        "default_pipeline": "tuk",
        "max_question_chars": settings.max_question_chars,
        "streaming": True,
        "quick_prompts": [
            "When do the second semester exams start?",
            "What are the tuition fees for this year?",
            "How do I register for units?",
            "What is the academic calendar for 2026?",
            "Where do I submit my project proposal?",
            "What are the graduation requirements?",
        ],
    }


def _validate_session_id(session_id: str) -> None:
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
