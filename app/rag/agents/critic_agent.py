"""
CriticAgent — validates that the generated answer is grounded in context.

Reads:  context.answer, context.metadata["context_string"]
Writes: context.grounded, context.grounded_reason
        context.metadata["retry_requested"]  (if validation fails and retries remain)

Two validation modes:
    - "heuristic": fast token-overlap check
    - "llm":       asks the LLM to judge grounding (slower, more accurate)

Location: app/rag/agents/critic_agent.py
"""

import re

import ollama

from app.rag.core.base_agent import ValidationAgent
from app.rag.core.context import RAGContext


# Common English stopwords — ignored in token-overlap scoring
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those", "i",
    "you", "we", "they", "but", "not", "no", "yes", "can", "could", "would",
    "should", "may", "might", "do", "does", "did", "so", "if", "then",
}


class CriticAgent(ValidationAgent):
    """
    Validates that the answer is grounded in the retrieved context.

    Args:
        mode: "heuristic" or "llm"
        overlap_threshold: min fraction of answer tokens found in context
                           (only used in heuristic mode)
        model_name: Ollama model for llm mode
        min_answer_length: answers shorter than this bypass validation
                           (prevents retrying on "I cannot find..." replies)
    """

    def __init__(
        self,
        mode: str = "heuristic",
        overlap_threshold: float = 0.4,
        model_name: str = "llama3.2:1b",
        min_answer_length: int = 20,
        name: str = "Critic",
    ):
        if mode not in ("heuristic", "llm"):
            raise ValueError(f"Unknown critic mode: {mode!r}")
        self.mode = mode
        self.overlap_threshold = overlap_threshold
        self.model_name = model_name
        self.min_answer_length = min_answer_length
        self.name = name

    # ------------------------------------------------------------------ #
    # Agent API
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        # Skip if router already answered (small_talk / off_topic)
        if context.route in ("small_talk", "off_topic"):
            context.grounded = True
            context.grounded_reason = f"route={context.route}, no validation needed"
            return context

        # Skip empty/very short answers — likely "I cannot find..." refusals
        if len(context.answer.strip()) < self.min_answer_length:
            context.grounded = True
            context.grounded_reason = "answer too short to validate (likely refusal)"
            return context

        # No context = nothing to ground against; reject if answer is long
        context_str = context.metadata.get("context_string", "")
        if not context_str.strip():
            context.grounded = False
            context.grounded_reason = "no retrieved context available"
            self._maybe_request_retry(context)
            return context

        if self.mode == "heuristic":
            grounded, reason = self._check_heuristic(context.answer, context_str)
        else:
            grounded, reason = self._check_llm(context.answer, context_str)

        context.grounded = grounded
        context.grounded_reason = reason

        if not grounded:
            self._maybe_request_retry(context, reason)

        return context

    # ------------------------------------------------------------------ #
    # Retry logic
    # ------------------------------------------------------------------ #

    def _maybe_request_retry(self, context: RAGContext, reason: str = "") -> None:
        """Request a retry if we have attempts left."""
        if context.retry_count < context.max_retries:
            context.request_retry(reason or "answer not grounded in context")

    # ------------------------------------------------------------------ #
    # Heuristic grounding check
    # ------------------------------------------------------------------ #

    def _check_heuristic(self, answer: str, context_str: str) -> tuple[bool, str]:
        answer_tokens = self._tokenize(answer)
        context_tokens = self._tokenize(context_str)

        if not answer_tokens:
            return True, "answer has no content tokens"

        overlap = answer_tokens & context_tokens
        ratio = len(overlap) / len(answer_tokens)

        if ratio >= self.overlap_threshold:
            return True, f"token overlap {ratio:.2f} >= {self.overlap_threshold}"

        return False, (
            f"token overlap {ratio:.2f} < {self.overlap_threshold} "
            f"(answer may not be grounded)"
        )

    @staticmethod
    def _tokenize(text: str) -> set:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    # ------------------------------------------------------------------ #
    # LLM grounding check
    # ------------------------------------------------------------------ #

    def _check_llm(self, answer: str, context_str: str) -> tuple[bool, str]:
        prompt = (
            "Compare the ANSWER to the CONTEXT. Decide if the ANSWER is FAITHFUL "
            "or UNFAITHFUL.\n\n"
            "FAITHFUL = every fact in the answer is stated in the context, "
            "even if the wording is different.\n"
            "UNFAITHFUL = the answer contains a fact that the context does not "
            "state or contradicts.\n\n"
            "Reply with exactly one word: FAITHFUL or UNFAITHFUL.\n\n"
            f"CONTEXT:\n{context_str[:4000]}\n\n"
            f"ANSWER:\n{answer[:800]}\n\n"
            "Verdict:"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 5,
                    "temperature": 0.0,
                    "num_ctx": 4096,
                },
                keep_alive=-1,
            )
            verdict = response["message"]["content"].strip().upper()
        except Exception as e:
            # Fail-closed: if the LLM itself errors, treat as ungrounded
            return False, f"LLM check failed ({e}); assuming ungrounded"

        # Check UNFAITHFUL first — the substring 'FAITHFUL' is inside it
        if "UNFAITHFUL" in verdict or "UNFAITH" in verdict:
            return False, "LLM verdict: UNFAITHFUL"
        if "FAITHFUL" in verdict:
            return True, "LLM verdict: FAITHFUL"

        # Fail-closed: unparseable verdict → treat as ungrounded
        return False, f"LLM verdict unclear ({verdict!r}); assuming ungrounded"