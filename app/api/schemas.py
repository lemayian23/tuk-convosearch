"""Request/response models. Location: app/api/schemas.py"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Student question")
    session_id: Optional[str] = Field(
        None,
        description=(
            "Conversation id. Omit to start a new conversation; the response "
            "echoes the id to use for follow-up questions."
        ),
    )
    pipeline: str = Field("tuk", description="tuk | cancer")

    @field_validator("message")
    @classmethod
    def _strip_and_bound(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be blank")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def _safe_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", value):
            raise ValueError("session_id must be 8-64 characters of letters, digits, _ or -")
        return value

    @field_validator("pipeline")
    @classmethod
    def _known_pipeline(cls, value: str) -> str:
        if value not in ("tuk", "cancer"):
            raise ValueError("pipeline must be 'tuk' or 'cancer'")
        return value


class SourceOut(BaseModel):
    source: str
    quote: str = ""
    relevance_score: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceOut]
    route: str
    grounded: bool
    chunks_found: int
    retry_count: int
    response_time: float
    error: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: Dict[str, Any]


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: float


class SessionOut(BaseModel):
    session_id: str
    title: str
    created_at: float
    last_active_at: float
    message_count: int


class DocumentOut(BaseModel):
    document_id: int
    title: str
    filename: str
    file_type: str
    upload_date: float
    chunk_count: int
    is_active: int
    indexed_at: Optional[float] = None


class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute path inside the corpus directory")


class StatusResponse(BaseModel):
    ready: bool
    chunks: int
    documents: int
    detail: Optional[str] = None
