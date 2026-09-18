"""
Shared RAGContext passed between all agents in a pipeline.
Location: backend/app/rag/core/context.py
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time


@dataclass
class Source:
    source: str
    quote: str = ""
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGContext:
    question: str
    session_id: str = "default"
    top_k: int = 5

    route: str = "unclassified"
    route_reason: str = ""

    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    retrieval_query: str = ""

    answer: str = ""
    answer_tokens: List[str] = field(default_factory=list)

    grounded: bool = False
    grounded_reason: str = ""
    retry_count: int = 0
    max_retries: int = 2

    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def mark_done(self) -> "RAGContext":
        self.finished_at = time.time()
        return self

    def request_retry(self, reason: str = "") -> "RAGContext":
        self.retry_count += 1
        self.metadata["retry_requested"] = True
        if reason:
            self.metadata["retry_reason"] = reason
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "session_id": self.session_id,
            "route": self.route,
            "route_reason": self.route_reason,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "chunks_found": len(self.retrieved_chunks),
            "grounded": self.grounded,
            "grounded_reason": self.grounded_reason,
            "retry_count": self.retry_count,
            "response_time": float(self.elapsed),
            "error": self.error,
        }

    def __repr__(self) -> str:
        q = self.question[:40] + ("..." if len(self.question) > 40 else "")
        return (
            f"<RAGContext q={q!r} route={self.route} "
            f"chunks={len(self.retrieved_chunks)} grounded={self.grounded}>"
        )