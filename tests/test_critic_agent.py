"""Unit tests for CriticAgent — heuristic mode only (no LLM calls)."""

from app.rag.agents.critic_agent import CriticAgent
from app.rag.core.context import RAGContext


def make_ctx(answer, context_str):
    ctx = RAGContext(question="q")
    ctx.route = "internal_docs"
    ctx.answer = answer
    ctx.metadata["context_string"] = context_str
    return ctx


def test_critic_skips_off_topic():
    c = CriticAgent(mode="heuristic")
    ctx = RAGContext(question="q")
    ctx.route = "off_topic"
    ctx = c.run(ctx)
    assert ctx.grounded is True


def test_critic_skips_short_answers():
    c = CriticAgent(mode="heuristic")
    ctx = make_ctx("OK", "some context here")
    ctx = c.run(ctx)
    assert ctx.grounded is True


def test_critic_detects_refusal_phrase():
    c = CriticAgent(mode="heuristic")
    answer = "The retrieved literature does not address this question."
    ctx = make_ctx(answer, "context about other topics entirely")
    ctx = c.run(ctx)
    assert ctx.grounded is True
    assert "refusal" in ctx.grounded_reason.lower()


def test_critic_accepts_grounded_answer():
    c = CriticAgent(mode="heuristic", overlap_threshold=0.4)
    answer = "Trastuzumab binds HER2 and inhibits downstream signaling pathways."
    context = (
        "[Source: doc]\nTrastuzumab is a monoclonal antibody that binds "
        "HER2 and inhibits downstream signaling pathways in cancer cells."
    )
    ctx = make_ctx(answer, context)
    ctx = c.run(ctx)
    assert ctx.grounded is True


def test_critic_rejects_ungrounded_answer():
    c = CriticAgent(mode="heuristic", overlap_threshold=0.5)
    answer = "Aspirin cures migraines by reducing planetary radiation."
    context = "[Source: doc]\nCancer is a disease of uncontrolled cell division."
    ctx = make_ctx(answer, context)
    ctx = c.run(ctx)
    assert ctx.grounded is False
    assert ctx.metadata.get("retry_requested") is True