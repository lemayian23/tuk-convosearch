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
    agent = RetrieverAgent(store=store)
    ctx = RAGContext(question="cancer question", top_k=5)
    ctx.route = "internal_docs"
    ctx = agent.run(ctx)
    assert len(ctx.retrieved_chunks) == 5
    assert len(ctx.sources) == 3  # max_sources default
    assert ctx.sources[0].source == "doc0.pdf"


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