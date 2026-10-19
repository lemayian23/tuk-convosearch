"""Unit tests for RouterAgent."""

import pytest

from app.rag.agents.router_agent import RouterAgent
from app.rag.configs.tuk_pipeline import TUK_GREETING
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


# ---------------------------------------------------------------------- #
# Bare-greeting handling (opt-in via small_talk_message)
# ---------------------------------------------------------------------- #


def make_tuk_like_router():
    from app.rag.configs.tuk_pipeline import TUK_EASTER_EGGS, TUK_KEYWORDS

    return RouterAgent(
        domain_keywords=TUK_KEYWORDS,
        easter_eggs=TUK_EASTER_EGGS,
        off_topic_message="Off-topic.",
        small_talk_message=TUK_GREETING,
    )


@pytest.mark.parametrize(
    "question",
    [
        "hi",
        "Hi!",
        "hello",
        "hey there",
        "good morning",
        "thanks",
        "are you there?",
        "hi, are you there?",
        "hello, how are you?",
    ],
)
def test_bare_greeting_gets_the_canned_reply(question):
    r = make_tuk_like_router()
    ctx = r.run(RAGContext(question=question))
    assert ctx.route == "small_talk"
    assert ctx.route_reason == "bare greeting"
    assert ctx.grounded is True
    assert ctx.answer == TUK_GREETING


@pytest.mark.parametrize(
    "question",
    [
        "which units are offered this semester",
        "hi, when do the second semester exams start?",
        "this is a hint about the cat dates",
        "are you able to tell me the exam dates",
        "just confirming the timetable",
    ],
)
def test_greeting_detection_needs_every_word_to_be_chatter(question):
    """"hi" is a substring of "which"/"hint"; a real question must never be
    answered with the canned greeting."""
    r = make_tuk_like_router()
    ctx = r.run(RAGContext(question=question))
    assert ctx.route != "small_talk"


def test_greeting_without_config_falls_through():
    """small_talk_message=None keeps the old behaviour: greetings are judged
    by the off-topic filter rather than answered with a canned reply."""
    r = make_router()
    ctx = r.run(RAGContext(question="good morning"))
    assert ctx.route == "off_topic"
    assert ctx.answer == "Off-topic."