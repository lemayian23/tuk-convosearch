"""
RetrieverAgent — wraps a BaseVectorStore as a pipeline Agent.

Reads:  context.question, context.top_k, context.route
Writes: context.retrieved_chunks, context.sources, context.retrieval_query

Location: app/rag/agents/retriever_agent.py
"""

from typing import List, Dict, Any

from app.rag.core.base_agent import RetrievalAgent
from app.rag.core.context import RAGContext, Source
from app.rag.stores.base_store import BaseVectorStore


SKIP_ROUTES = {"off_topic", "small_talk"}


class RetrieverAgent(RetrievalAgent):
    """
    Args:
        store:                anything with .search(query, k)
        max_context_chars:    per-chunk trim applied when building the prompt
        max_quote_chars:      per-source trim applied to what the student sees
        max_sources:          how many sources to show
        max_chunks_per_source: 0 = no limit. A 5-chunk prompt from a 3-document
                              corpus is usually the same document three times,
                              and every extra copy is prefill the student waits for.
        min_relevance_margin: drop chunks scoring this far below the best one.
                              0 = keep everything.

    The two trimming knobs exist because prompt length IS latency on a CPU
    model: measured cold, 924 prompt tokens took 41.5 s to prefill and 432 took
    20.0 s, at about 22 tokens/s. They are applied before the sources are built,
    so a citation the student can see is always a chunk the model actually read.
    """

    def __init__(
        self,
        store: BaseVectorStore,
        name: str = "Retriever",
        max_context_chars: int = 600,
        max_quote_chars: int = 250,
        max_sources: int = 3,
        max_chunks_per_source: int = 2,
        min_relevance_margin: float = 0.05,
    ):
        self.store = store
        self.name = name
        self.max_context_chars = max_context_chars
        self.max_quote_chars = max_quote_chars
        self.max_sources = max_sources
        self.max_chunks_per_source = max_chunks_per_source
        self.min_relevance_margin = min_relevance_margin

    def run(self, context: RAGContext) -> RAGContext:
        if context.route in SKIP_ROUTES:
            return context

        query = context.retrieval_query or context.question
        context.retrieval_query = query

        try:
            chunks = self._select(self.store.search(query, k=context.top_k))
        except Exception as e:
            context.error = f"Retrieval failed: {e}"
            return context

        context.retrieved_chunks = chunks
        context.sources = self._build_sources(chunks)
        context.metadata["context_string"] = self.build_context_string(chunks)

        return context

    def _select(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank-ordered pruning. The first chunk always survives - it defines the
        margin it is measured against - so the result is empty only if the
        store returned nothing.
        """
        if not chunks:
            return []
        best = float(chunks[0].get("relevance_score", 0.0))
        per_source: Dict[str, int] = {}
        kept = []
        for chunk in chunks:
            score = float(chunk.get("relevance_score", 0.0))
            if self.min_relevance_margin and score < best - self.min_relevance_margin:
                continue
            source = str(chunk.get("metadata", {}).get("source", "unknown"))
            seen = per_source.get(source, 0)
            if self.max_chunks_per_source and seen >= self.max_chunks_per_source:
                continue
            per_source[source] = seen + 1
            kept.append(chunk)
        return kept

    def stream(self, context: RAGContext):
        """
        Retrieval is not token-streamable, but it is the slowest step the
        user cannot see, so emit the sources as soon as they are known. The
        student's browser can then show "from 2026 Academic Year Calendar"
        while the answer is still being generated.
        """
        context = self.run(context)
        yield {
            "type": "sources",
            "sources": [source.to_dict() for source in context.sources],
            "chunks_found": len(context.retrieved_chunks),
        }

    def build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        parts = []
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            text = chunk.get("text", "")[: self.max_context_chars]
            parts.append(f"[Source: {source}]\n{text}")
        return "\n\n".join(parts)

    def _build_sources(self, chunks: List[Dict[str, Any]]) -> List[Source]:
        sources = []
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