"""Unit tests for RetrieverAgent with a fake store."""

from app.rag.agents.retriever_agent import RetrieverAgent
from app.rag.core.context import RAGContext
from app.rag.stores.base_store import BaseVectorStore


class FakeStore(BaseVectorStore):
    """Returns fixed chunks regardless of query."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.search_calls = []

    def add_chunks(self, chunks):
        raise NotImplementedError

    def search(self, query, k=5):
        self.search_calls.append((query, k))
        return self._chunks[:k]

    def get_stats(self):
        return {"total_chunks": len(self._chunks), "dimension": 384}


def make_chunks(n=5):
    return [
        {
            "text": f"chunk {i} text about cancer",
            "metadata": {"source": f"doc{i}.pdf", "chunk_index": i},
            "distance": 0.5,
            "relevance_score": 0.9 - i * 0.1,
        }
        for i in range(n)
    ]


def test_retriever_populates_chunks_and_sources():
    store = FakeStore(make_chunks(5))
    agent = RetrieverAgent(store=store, max_chunks_per_source=0, min_relevance_margin=0.0)
    ctx = RAGContext(question="cancer question", top_k=5)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    assert len(ctx.retrieved_chunks) == 5
    assert len(ctx.sources) == 3  # max_sources default
    assert ctx.sources[0].source == "doc0.pdf"


def test_retriever_prunes_low_scoring_tail():
    """
    The prompt is what a student waits on: measured cold on CPU, 924 prompt
    tokens took 41.5s to prefill and 432 took 20.0s. Scores here drop 0.1 apart,
    so only the top chunk is close enough to the leader to be worth the wait.
    """
    store = FakeStore(make_chunks(5))
    agent = RetrieverAgent(store=store, min_relevance_margin=0.05)
    ctx = RAGContext(question="cancer question", top_k=5)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    assert [c["metadata"]["source"] for c in ctx.retrieved_chunks] == ["doc0.pdf"]
    assert ctx.metadata["context_string"].count("[Source:") == 1


def test_retriever_caps_chunks_from_one_source():
    """Three chunks of the same calendar doc say less than three different docs."""
    chunks = make_chunks(3)
    for chunk in chunks:
        chunk["metadata"]["source"] = "one_doc.pdf"
        chunk["relevance_score"] = 0.9
    store = FakeStore(chunks)
    agent = RetrieverAgent(store=store, max_chunks_per_source=2, min_relevance_margin=0.0)
    ctx = RAGContext(question="cancer question", top_k=3)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    assert len(ctx.retrieved_chunks) == 2


def test_retriever_keeps_a_full_context_when_both_knobs_are_off():
    """TUK_MIN_RELEVANCE_MARGIN=0 and TUK_MAX_CHUNKS_PER_SOURCE=0 restore k=5."""
    store = FakeStore(make_chunks(5))
    agent = RetrieverAgent(store=store, max_chunks_per_source=0, min_relevance_margin=0.0)
    ctx = RAGContext(question="cancer question", top_k=5)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    assert len(ctx.retrieved_chunks) == 5


def test_retriever_uses_retrieval_query_if_set():
    store = FakeStore(make_chunks(3))
    agent = RetrieverAgent(store=store)
    ctx = RAGContext(question="original")
    ctx.route = "internal_docs"
    ctx.retrieval_query = "expanded query"
    ctx = agent.run(ctx)
    assert store.search_calls[0][0] == "expanded query"


def test_retriever_skips_on_off_topic_route():
    store = FakeStore(make_chunks(3))
    agent = RetrieverAgent(store=store)
    ctx = RAGContext(question="weather")
    ctx.route = "off_topic"
    ctx = agent.run(ctx)
    assert ctx.retrieved_chunks == []
    assert store.search_calls == []


def test_retriever_builds_context_string():
    store = FakeStore(make_chunks(2))
    agent = RetrieverAgent(store=store)
    ctx = RAGContext(question="q", top_k=2)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    context_str = ctx.metadata.get("context_string", "")
    assert "[Source: doc0.pdf]" in context_str
    assert "chunk 0 text" in context_str