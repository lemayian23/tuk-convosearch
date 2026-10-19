"""
Tests for the evaluation scorer.

The harness is what tells us whether a change helped, so a scorer that can
turn green on a wrong answer is worse than no scorer at all. These are unit
tests — no Ollama, no embeddings — because `python -m evals.runner` itself
needs both and cannot run in CI.

Location: tests/test_evals.py
"""

from app.rag.core.context import RAGContext, Source
from evals.questions import TUK_EVAL_QUESTIONS
from evals.runner import score_question


def _context(answer="", chunks=(), sources=(), route="internal_docs", grounded=True):
    ctx = RAGContext(question="q", route=route, grounded=grounded)
    ctx.answer = answer
    ctx.retrieved_chunks = [{"text": text, "metadata": {"source": text}} for text in chunks]
    ctx.sources = [Source(source=name) for name in sources]
    return ctx


def test_keyword_and_source_must_both_match():
    entry = {
        "question": "q",
        "expected_keywords": ["march 23"],
        "expected_sources": ["Calendar"],
    }
    good = _context(
        answer="Second CATs run March 23.",
        chunks=["Second CATs run March 23 - 28"],
        sources=["2026 Academic Year Calendar.docx"],
    )
    wrong_document = _context(
        answer="Second CATs run March 23.",
        chunks=["Second CATs run March 23 - 28"],
        sources=["rag_train_docs.docx"],
    )
    assert score_question(good, entry)["passed"] is True
    assert score_question(wrong_document, entry)["source_ok"] is False
    assert score_question(wrong_document, entry)["passed"] is False


def test_refusal_question_fails_when_the_model_answers_anyway():
    entry = {"question": "q", "expected_keywords": [], "expect_no_answer": True}
    hallucinating = _context(
        answer="Tuition fees are Ksh 45,000 per semester.",
        chunks=["something unrelated"],
        sources=["rag_train_docs.docx"],
    )
    honest = _context(
        answer="I cannot find this information in the available TU-K documents.",
        chunks=["weakly matched text"],
        sources=["rag_train_docs.docx"],
    )
    assert score_question(hallucinating, entry)["refusal_ok"] is False
    assert score_question(honest, entry)["passed"] is True


def test_small_talk_is_scored_on_route_not_on_retrieval():
    entry = {"question": "hi", "expected_route": "small_talk", "category": "router_control"}
    ctx = _context(answer="Hello.", route="small_talk")
    scored = score_question(ctx, entry)
    assert scored["route_ok"] is True
    assert scored["passed"] is True

    leaked = _context(answer="Hello.", route="internal_docs")
    assert score_question(leaked, entry)["passed"] is False


def test_every_tuk_question_declares_its_category():
    """A new question with no category lands in an unnamed bucket and quietly
    disappears from the per-category table."""
    assert all(entry.get("category") for entry in TUK_EVAL_QUESTIONS)
    assert all("question" in entry for entry in TUK_EVAL_QUESTIONS)
