"""
RouterAgent — classifies a question before retrieval.

Decides one of:
    - "small_talk"    -> canned reply (jokes, capability questions)
    - "off_topic"     -> polite rejection
    - "internal_docs" -> proceed to retrieval + synthesis

Reads:  context.question
Writes: context.route, context.route_reason
        context.answer (only for small_talk / off_topic)
        context.metadata["context_string"] (empty for short-circuits)

Location: app/rag/agents/router_agent.py
"""

from typing import Dict, List, Optional
import re

from app.rag.core.base_agent import Agent
from app.rag.core.context import RAGContext

# Words that only ever appear in greeting / presence-check chatter. A
# question is small talk when EVERY word of it is in this set, which is why
# it is a set and not a regex: "hi, are you there?" has to match, while
# "which units are offered" must not — substring matching on "hi" would
# swallow "which", and half the questions in a student corpus start with it.
SMALL_TALK_WORDS = frozenset(
    """
    hi hello hey yo hola sup good morning afternoon evening day
    how are r you your ur there real human bot ai ok okay
    thanks thank you sorry please yes no fine well just and the a an it's its
    """.split()
)

# Longest chatter worth answering with a canned reply. Beyond this the message
# almost certainly carries a question ("hi i want to know when exams start").
MAX_GREETING_WORDS = 6


_WORD = re.compile(r"[a-z']+")


def _is_bare_greeting(q_lower: str) -> bool:
    words = _WORD.findall(q_lower)
    return (
        bool(words)
        and len(words) <= MAX_GREETING_WORDS
        and all(w in SMALL_TALK_WORDS for w in words)
    )


class RouterAgent(Agent):
    """
    Keyword-based router. Kept intentionally simple and deterministic so
    it can be swapped for an LLM router later without changing the
    pipeline contract.

    Args:
        domain_keywords: terms that indicate an in-domain question
        easter_eggs:     dict of {lowercase trigger -> canned reply}
        off_topic_message: reply when no keyword matches
        small_talk_message: reply to a bare greeting ("hi", "thanks"). None
                         disables greeting handling and lets those through to
                         the off-topic filter.
        min_length:      questions shorter than this bypass the off-topic
                         check (prevents rejecting "hi", "ok", etc.)
    """

    def __init__(
        self,
        domain_keywords: List[str],
        easter_eggs: Optional[Dict[str, str]] = None,
        off_topic_message: str = "I can only answer questions about this domain.",
        small_talk_message: Optional[str] = None,
        min_length: int = 5,
        name: str = "Router",
    ):
        self.domain_keywords = [kw.lower() for kw in domain_keywords]
        self.easter_eggs = {k.lower(): v for k, v in (easter_eggs or {}).items()}
        self.off_topic_message = off_topic_message
        self.small_talk_message = small_talk_message
        self.min_length = min_length
        self.name = name

    # ------------------------------------------------------------------ #
    # Agent API
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        q_lower = context.question.lower().strip()

        # 1. Small talk (canned replies) — highest priority
        for trigger, reply in self.easter_eggs.items():
            if trigger in q_lower:
                context.route = "small_talk"
                context.route_reason = f"matched easter egg: {trigger!r}"
                context.answer = reply
                context.grounded = True  # canned replies are "grounded" by definition
                return context

        if self.small_talk_message and _is_bare_greeting(q_lower):
            context.route = "small_talk"
            context.route_reason = "bare greeting"
            context.answer = self.small_talk_message
            context.grounded = True
            return context

        # 2. Off-topic filter — only if the question is long enough to judge
        if len(q_lower) > self.min_length and not self._is_in_domain(q_lower):
            context.route = "off_topic"
            context.route_reason = "no domain keyword matched"
            context.answer = self.off_topic_message
            context.grounded = False
            return context

        # 3. Default: proceed to retrieval
        context.route = "internal_docs"
        context.route_reason = "domain keyword matched or question too short to filter"
        return context

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_in_domain(self, q_lower: str) -> bool:
        return any(kw in q_lower for kw in self.domain_keywords)