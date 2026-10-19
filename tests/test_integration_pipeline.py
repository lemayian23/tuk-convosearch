"""
End-to-end smoke test: the real TU-K pipeline over the real corpus with a
live Ollama.

Excluded from the default run — pytest.ini adds `-m "not integration"` — so CI
and a laptop without a model stay green. On a box that has Ollama running and
the corpus indexed:

    pytest -m integration

Location: tests/test_integration_pipeline.py
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tuk_pipeline():
    """
    Builds the same pipeline the API serves, against the real data/ and
    corpus/ in this checkout.

    tests/conftest.py points every TUK_* path at a temp directory and sends
    Ollama calls to a dead port so the unit suite stays offline. Dropping those
    keys puts Settings back on exactly what app.main reads: the repo-root .env
    plus the defaults in app/config.py.
    """
    for key in (
        "TUK_ENVIRONMENT",
        "TUK_DATA_DIR",
        "TUK_CORPUS_DIR",
        "TUK_PUBMED_INDEX_DIR",
        "TUK_OLLAMA_HOST",
        "TUK_WARM_MODEL_ON_BOOT",
        "TUK_OLLAMA_MODEL",
    ):
        os.environ.pop(key, None)

    from app.config import get_settings
    from app.ingestion import index as index_module
    from app.services.ollama import model_status

    get_settings.cache_clear()
    index_module.reset_index()

    settings = get_settings()
    stats = index_module.get_index().get_stats()
    if stats["total_chunks"] == 0:
        pytest.skip(f"nothing indexed in {settings.data_dir} — run scripts/ingest_corpus.py")

    status = model_status()
    if not status["reachable"]:
        pytest.skip(f"Ollama is not reachable at {settings.ollama_host}: {status['error']}")
    if not status["model_present"]:
        pytest.skip(f"model {status['expected_model']} is not pulled")

    from app.rag.configs.tuk_pipeline import build_tuk_pipeline

    return build_tuk_pipeline(critic_mode="heuristic")


def test_greeting_short_circuits_before_retrieval(tuk_pipeline):
    """No model call, no retrieval — the router answers chatter itself."""
    from app.rag.core.context import RAGContext

    ctx = tuk_pipeline.run(RAGContext(question="Hi, are you there?", session_id="it-greet"))
    assert ctx.route == "small_talk"
    assert ctx.retrieved_chunks == []
    assert "TUK" in ctx.answer or "exam" in ctx.answer.lower()


def test_calendar_question_is_grounded_and_cited(tuk_pipeline):
    from app.rag.core.context import RAGContext

    ctx = tuk_pipeline.run(
        RAGContext(
            question="When does Semester II begin at TUK in 2026?",
            session_id="it-calendar",
        )
    )
    assert ctx.route == "internal_docs", ctx.route_reason
    assert ctx.retrieved_chunks, "retrieval returned nothing"
    assert ctx.sources, "an in-domain answer must cite a document"
    assert ctx.answer and "could not form an answer" not in ctx.answer.lower()
    # The calendar is in the corpus, so a correct answer carries a date.
    assert any(ch.isdigit() for ch in ctx.answer), ctx.answer


def test_question_with_no_answer_in_the_corpus_refuses(tuk_pipeline):
    """Fee amounts are not in the corpus; the student must not get a number."""
    from evals.runner import REFUSAL_MARKERS

    from app.rag.core.context import RAGContext

    ctx = tuk_pipeline.run(
        RAGContext(question="How much are tuition fees for 2026?", session_id="it-refusal")
    )
    assert ctx.route == "internal_docs"
    lowered = ctx.answer.lower()
    assert any(marker in lowered for marker in REFUSAL_MARKERS), ctx.answer
