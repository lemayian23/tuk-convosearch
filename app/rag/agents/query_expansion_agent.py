"""
QueryExpansionAgent — rewrites a user question into a domain-optimized
retrieval query using the local LLM.

Sits between Router and Retriever. Reads context.question, writes
context.retrieval_query. The Retriever already prefers retrieval_query
when present, so no changes to RetrieverAgent are needed.

Location: app/rag/agents/query_expansion_agent.py
"""

from typing import Optional

import ollama

from app.rag.core.base_agent import Agent
from app.rag.core.context import RAGContext


class QueryExpansionAgent(Agent):
    """
    Rewrites the user question into a better retrieval query.

    Args:
        model_name:      Ollama model to use for rewriting
        prompt_template: instructions with {question} placeholder
        max_tokens:      cap the expansion length
        temperature:     keep low — we want deterministic output
        name:            agent name (shows up in pipeline describe())
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        prompt_template: Optional[str] = None,
        max_tokens: int = 80,
        temperature: float = 0.1,
        name: str = "QueryExpander",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template or self._default_prompt()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = name

    def run(self, context: RAGContext) -> RAGContext:
        # Skip if router short-circuited
        if context.route in ("small_talk", "off_topic"):
            return context

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": self.prompt_template.format(question=context.question),
                }],
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "num_ctx": 512,
                },
                keep_alive=-1,
            )
            expanded = response["message"]["content"].strip()

            # Sanity: must be non-empty and reasonably short
            if expanded and len(expanded) < 300:
                context.retrieval_query = expanded
                context.metadata["original_question"] = context.question
                context.metadata["expanded_query"] = expanded
            else:
                context.retrieval_query = context.question
                context.metadata["expansion_skipped"] = "empty or too long"

        except Exception as e:
            # Never fail hard — fall back to the original question
            print(f"  [QueryExpansion] Failed ({e}), using original question")
            context.retrieval_query = context.question
            context.metadata["expansion_error"] = str(e)

        return context

    @staticmethod
    def _default_prompt() -> str:
        """Generic prompt. Pipelines should override with domain version."""
        return (
            "Rewrite the following question into a short, keyword-rich "
            "search query. Output ONLY the query, no explanation, no quotes.\n\n"
            "Question: {question}\n\nQuery:"
        )