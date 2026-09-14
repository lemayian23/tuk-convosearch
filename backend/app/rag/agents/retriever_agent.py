"""
RetrieverAgent — wraps a BaseVectorStore as a pipeline Agent.

Reads:  context.question, context.top_k, context.route
Writes: context.retrieved_chunks, context.sources, context.retrieval_query

Location: backend/app/rag/agents/retriever_agent.py
"""

from typing import List, Dict, Any, Optional

from app.rag.core.base_agent import RetrievalAgent
from app.rag.core.context import RAGContext, Source
from app.rag.stores.base_store import BaseVectorStore


# Routes that skip retrieval entirely
SKIP_ROUTES = {"off_topic", "small_talk"}


class RetrieverAgent(RetrievalAgent):
    """
    Retrieves top-k relevant chunks from a vector store and populates
    context.sources with typed Source objects.

    Args:
        store: any BaseVectorStore implementation (FaissStore, PubMedStore, ...)
        name:  override agent name (useful when multiple retrievers run)
        max_context_chars: truncate each chunk before adding to context
        max_quote_chars:   truncate each source quote shown in the UI
        max_sources:       cap the number of sources returned to the user
    """

    def __init__(
        self,
        store: BaseVectorStore,
        name: str = "Retriever",
        max_context_chars: int = 600,
        max_quote_chars: int = 250,
        max_sources: int = 3,
    ):
        self.store = store
        self.name = name
        self.max_context_chars = max_context_chars
        self.max_quote_chars = max_quote_chars
        self.max_sources = max_sources

    # ------------------------------------------------------------------ #
    # Agent API
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        # Short-circuit routes never hit the store
        if context.route in SKIP_ROUTES:
            return context

        query = context.retrieval_query or context.question
        context.retrieval_query = query

        try:
            chunks = self.store.search(query, k=context.top_k)
        except Exception as e:
            context.error = f"Retrieval failed: {e}"
            return context

        context.retrieved_chunks = chunks
        context.sources = self._build_sources(chunks)

        # Stash the LLM-ready context string for the Synthesizer
        context.metadata["context_string"] = self.build_context_string(chunks)

        return context

    # ------------------------------------------------------------------ #
    # Helpers (public — SynthesizerAgent reuses build_context_string)
    # ------------------------------------------------------------------ #

    def build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into the [Source: X] blocks the LLM expects."""
        if not chunks:
            return ""

        parts: List[str] = []
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            text = chunk.get("text", "")[: self.max_context_chars]
            parts.append(f"[Source: {source}]\n{text}")
        return "\n\n".join(parts)

    def _build_sources(self, chunks: List[Dict[str, Any]]) -> List[Source]:
        sources: List[Source] = []
        for chunk in chunks[: self.max_sources]:
            text = chunk.get("text", "")
            quote = text[: self.max_quote_chars].strip()
            if len(text) > self.max_quote_chars:
                quote += "..."

            sources.append(
                Source(
                    source=str(chunk.get("metadata", {}).get("source", "unknown")),
                    quote=quote,
                    relevance_score=float(chunk.get("relevance_score", 0.0)),
                )
            )
        return sources