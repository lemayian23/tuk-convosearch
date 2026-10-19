"""
Chat service: the async boundary between FastAPI and a blocking pipeline.

Two production problems are solved here.

* The old endpoint called `ollama.chat` directly inside an `async def`, so a
  single student's generation blocked the event loop and every other student
  waited behind it. Pipeline runs now happen on a dedicated thread pool, and
  the async layer only ever awaits a queue.
* A 4GB VPS serving llama3.2 can generate for roughly two students at a time.
  A semaphore turns excess traffic into a fast, honest 503 with a Retry-After
  instead of letting the model thrash and every request time out.

Location: app/services/chat.py
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import Settings, get_settings
from app.rag.core.context import RAGContext
from app.services import database
from app.services.pipelines import get_pipeline

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ChatBusyError(RuntimeError):
    """The generation queue is full; the caller should retry later."""


class ChatTimeoutError(RuntimeError):
    """The model did not produce an answer inside the configured budget."""


@dataclass
class ChatResult:
    session_id: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    route: str = ""
    grounded: bool = False
    grounded_reason: str = ""
    chunks_found: int = 0
    retry_count: int = 0
    response_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "sources": self.sources,
            "route": self.route,
            "grounded": self.grounded,
            "grounded_reason": self.grounded_reason,
            "chunks_found": self.chunks_found,
            "retry_count": self.retry_count,
            "response_time": round(self.response_time, 3),
            "error": self.error,
        }


class ChatService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_concurrent_llm_calls,
            thread_name_prefix="rag",
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------ #
    # Loop-bound primitives (built on first use inside the running loop)
    # ------------------------------------------------------------------ #

    def _gate(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_llm_calls)
            self._loop = loop
        return self._semaphore

    async def _reserve_slot(self):
        gate = self._gate()
        try:
            await asyncio.wait_for(
                gate.acquire(), timeout=self.settings.llm_queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ChatBusyError(
                f"All {self.settings.max_concurrent_llm_calls} model slots are busy. "
                "Please try again in a moment."
            ) from exc

    # ------------------------------------------------------------------ #
    # Context preparation
    # ------------------------------------------------------------------ #

    def _build_context(self, question: str, session_id: str, pipeline_name: str) -> RAGContext:
        history_rows = database.recent_messages(session_id, self.settings.history_turns * 2)
        context = RAGContext(
            question=question,
            session_id=session_id,
            top_k=self.settings.top_k,
        )
        context.metadata["history"] = self._format_history(history_rows)
        context.metadata["pipeline"] = pipeline_name
        return context

    @staticmethod
    def _format_history(rows: List[Dict[str, str]]) -> str:
        if not rows:
            return "No previous conversation."
        return "\n".join(f"{r['role'].upper()}: {r['content']}" for r in rows)

    def _persist(self, context: RAGContext, result: ChatResult, pipeline_name: str) -> None:
        try:
            database.touch_session(context.session_id, context.question)
            database.add_message(context.session_id, "user", context.question)
            if result.answer:
                database.add_message(context.session_id, "assistant", result.answer)
            database.log_query(
                session_id=context.session_id,
                question=context.question,
                answer=result.answer,
                sources=result.sources,
                chunks_found=result.chunks_found,
                response_time=result.response_time,
                route=result.route,
                grounded=result.grounded,
                retry_count=result.retry_count,
                model=pipeline_name,
            )
        except Exception:
            logger.exception("Failed to persist exchange for session %s", context.session_id)

    @staticmethod
    def _result_from_context(context: RAGContext) -> ChatResult:
        payload = context.to_dict()
        return ChatResult(
            session_id=context.session_id,
            answer=payload["answer"],
            sources=payload["sources"],
            route=payload["route"],
            grounded=payload["grounded"],
            grounded_reason=payload["grounded_reason"],
            chunks_found=payload["chunks_found"],
            retry_count=payload["retry_count"],
            response_time=payload["response_time"],
            error=payload["error"],
        )

    # ------------------------------------------------------------------ #
    # Blocking answer
    # ------------------------------------------------------------------ #

    async def answer(self, question: str, session_id: str, pipeline_name: str = "tuk") -> ChatResult:
        pipeline = get_pipeline(pipeline_name)
        context = self._build_context(question, session_id, pipeline_name)
        loop = asyncio.get_running_loop()

        await self._reserve_slot()
        try:
            future = loop.run_in_executor(self._executor, pipeline.run, context)
            try:
                finished = await asyncio.wait_for(
                    asyncio.shield(future), timeout=self.settings.llm_timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                raise ChatTimeoutError(
                    "The model took too long to answer. Try a shorter question."
                ) from exc
        finally:
            self._semaphore.release()

        result = self._result_from_context(finished)
        self._persist(finished, result, pipeline_name)
        return result

    # ------------------------------------------------------------------ #
    # Streaming answer
    # ------------------------------------------------------------------ #

    async def stream(
        self, question: str, session_id: str, pipeline_name: str = "tuk"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield pipeline events as they happen.

        The pipeline is a synchronous generator, so a worker thread drives it
        and hands events back over `call_soon_threadsafe`. `cancel` lets a
        disconnected client stop generation instead of burning model time on
        an answer nobody is reading.
        """
        pipeline = get_pipeline(pipeline_name)
        context = self._build_context(question, session_id, pipeline_name)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        await self._reserve_slot()

        def drive() -> None:
            try:
                for event in pipeline.stream(context):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # agent raised before the pipeline caught it
                logger.exception("Streaming pipeline failed")
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "message": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        self._executor.submit(drive)

        timed_out = False
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=self.settings.llm_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    cancel.set()
                    timed_out = True
                    yield {
                        "type": "error",
                        "message": "Generation timed out. Please try again.",
                    }
                    break
                if event is _SENTINEL:
                    break
                yield event
        finally:
            cancel.set()
            self._semaphore.release()

        result = self._result_from_context(context)
        if timed_out and not result.answer:
            result.error = "timeout"
        self._persist(context, result, pipeline_name)


_service: Optional[ChatService] = None
_service_lock = threading.Lock()


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ChatService()
    return _service
