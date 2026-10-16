"""Unit tests for RouterAgent."""

from app.rag.agents.router_agent import RouterAgent
from app.rag.core.context import RAGContext


def make_router():
    return RouterAgent(
        domain_keywords=["cancer", "tumor", "oncology"],
        easter_eggs={"hello": "hi there"},
        off_topic_message="Off-topic.",
    )


def test_router_matches_easter_egg():
    r = make_router()
    ctx = RAGContext(question="hello")
    ctx = r.run(ctx)
    assert ctx.route == "small_talk"
    assert ctx.answer == "hi there"


def test_router_matches_easter_egg_case_insensitive():
    r = make_router()
    ctx = RAGContext(question="HELLO there")
    ctx = r.run(ctx)
    assert ctx.route == "small_talk"


def test_router_routes_in_domain_question():
    r = make_router()
    ctx = RAGContext(question="What is the best treatment for cancer?")
    ctx = r.run(ctx)
    assert ctx.route == "internal_docs"


def test_router_rejects_off_topic_question():
    r = make_router()
    ctx = RAGContext(question="What is the weather in Paris?")
    ctx = r.run(ctx)
    assert ctx.route == "off_topic"
    assert ctx.answer == "Off-topic."


def test_router_allows_short_questions_through():
    """Very short questions bypass the off-topic filter."""
    r = make_router()
    ctx = RAGContext(question="hi")
    ctx = r.run(ctx)
    assert ctx.route == "internal_docs"


def test_router_records_reason():
    r = make_router()
    ctx = RAGContext(question="A question about tumors")
    ctx = r.run(ctx)
    assert ctx.route_reason != ""