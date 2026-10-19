"""
Tests for the settings layer and the concurrency gate in ChatService.

Location: tests/test_chat_service.py
"""

import asyncio
import time

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services import database
from app.services.chat import ChatBusyError, ChatService


class TestSettings:
    def test_environment_vars_use_the_tuk_prefix(self, monkeypatch):
        monkeypatch.setenv("TUK_OLLAMA_MODEL", "qwen2.5:7b-instruct")
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert Settings().ollama_model == "qwen2.5:7b-instruct"

    def test_cors_origins_accepts_a_comma_separated_string(self):
        parsed = Settings(cors_origins="https://a.tuk.ac.ke, https://b.tuk.ac.ke")
        assert parsed.cors_origins == ["https://a.tuk.ac.ke", "https://b.tuk.ac.ke"]

    def test_production_refuses_a_short_secret(self):
        with pytest.raises(ValidationError, match="TUK_SECRET_KEY"):
            Settings(environment="production", secret_key="hunter2")

    def test_production_refuses_wildcard_cors(self):
        with pytest.raises(ValidationError, match="CORS"):
            Settings(
                environment="production",
                secret_key="x" * 40,
                cors_origins=["*"],
            )

    def test_development_allows_the_convenient_defaults(self):
        settings = Settings(environment="development", secret_key="")
        assert settings.max_upload_bytes == 15 * 1024 * 1024


class _SlowPipeline:
    def __init__(self, delay: float, streaming: bool = True):
        self.delay = delay
        self.streaming = streaming

    def run(self, context):
        time.sleep(self.delay)
        context.route = "internal_docs"
        context.answer = "done"
        return context.mark_done()

    def stream(self, context):
        time.sleep(self.delay)
        context.route = "internal_docs"
        context.answer = "done"
        yield {"type": "token", "content": "done"}
        yield {"type": "done", "context": context.to_dict()}


@pytest.fixture
def service(monkeypatch, settings):
    monkeypatch.setattr(settings, "max_concurrent_llm_calls", 1)
    monkeypatch.setattr(settings, "llm_queue_timeout_seconds", 0.05)
    return ChatService(settings=settings)


async def test_answers_are_written_to_the_session_and_the_log(service, monkeypatch):
    monkeypatch.setattr("app.services.chat.get_pipeline", lambda name="tuk": _SlowPipeline(0))

    result = await service.answer("When do exams start?", session_id="sess123456")

    assert result.answer == "done"
    roles = [m["role"] for m in database.recent_messages("sess123456", 10)]
    assert roles == ["user", "assistant"]
    logged = database.get_recent_queries(5)
    assert any(row["question"] == "When do exams start?" for row in logged)


async def test_excess_traffic_gets_busy_not_a_hanging_request(service, monkeypatch):
    monkeypatch.setattr("app.services.chat.get_pipeline", lambda name="tuk": _SlowPipeline(0.4))

    first = asyncio.create_task(service.answer("question one", session_id="sessAAAAAA"))
    await asyncio.sleep(0.05)  # let the first task take the only slot

    with pytest.raises(ChatBusyError):
        await service.answer("question two", session_id="sessBBBBBB")

    assert (await first).answer == "done"


async def test_blocking_pipeline_does_not_freeze_the_event_loop(service, monkeypatch):
    """
    The regression this guards: the original endpoint called ollama.chat
    directly inside `async def`, so one student's generation stalled every
    other request on the same worker.
    """
    monkeypatch.setattr("app.services.chat.get_pipeline", lambda name="tuk": _SlowPipeline(0.3))

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(12):
            await asyncio.sleep(0.02)
            ticks += 1

    answer_task = asyncio.create_task(service.answer("q", session_id="sessDDDDDD"))
    await ticker()
    await answer_task

    assert ticks >= 10, f"event loop was blocked: only {ticks} ticks during a 0.3s generation"


async def test_streaming_releases_its_slot_when_the_client_disconnects(service, monkeypatch):
    monkeypatch.setattr("app.services.chat.get_pipeline", lambda name="tuk": _SlowPipeline(0.2))

    generator = service.stream("q one", session_id="sessCCCCCC")
    await generator.__anext__()
    await generator.aclose()

    # The single slot must be free again, otherwise a student who closes the
    # tab would permanently occupy the model.
    result = await service.answer("q two", session_id="sessEEEEEE")
    assert result.answer == "done"


def test_ollama_client_is_built_from_settings_not_localhost(monkeypatch):
    """`ollama.chat()` at module level reads OLLAMA_HOST, never our settings,
    so it would talk to localhost:11434 — a different container on the VPS.
    Every answer would fail while /api/health still reported the model
    present, which is the worst possible failure mode to debug remotely."""
    import threading

    import ollama

    from app.services import ollama as ollama_service

    seen = {}
    monkeypatch.setattr(ollama_service, "_THREAD_LOCAL", threading.local())
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host=None, **kwargs: seen.update(host=host) or object(),
    )

    ollama_service.client()
    assert seen["host"] == "http://127.0.0.1:1"


def test_synthesis_failure_never_shows_the_student_a_host(monkeypatch):
    from app.rag.agents.synthesizer_agent import SynthesizerAgent
    from app.rag.core.context import RAGContext
    from app.services import ollama as ollama_service

    class Boom:
        def chat(self, **kwargs):
            raise ConnectionError("[Errno 111] Connection refused: 127.0.0.1:11434")

    monkeypatch.setattr(ollama_service, "client", lambda: Boom())
    context = SynthesizerAgent(model_name="m").run(RAGContext(question="q"))

    assert "11434" not in context.answer
    assert "127.0.0.1" not in context.answer
    assert "Connection refused" in (context.error or "")
