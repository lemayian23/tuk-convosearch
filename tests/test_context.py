"""Unit tests for RAGContext."""

from app.rag.core.context import RAGContext, Source


def test_context_defaults():
    ctx = RAGContext(question="test question")
    assert ctx.question == "test question"
    assert ctx.session_id == "default"
    assert ctx.top_k == 5
    assert ctx.route == "unclassified"
    assert ctx.answer == ""
    assert ctx.grounded is False
    assert ctx.retry_count == 0
    assert ctx.max_retries == 2
    assert ctx.retrieved_chunks == []
    assert ctx.sources == []


def test_context_mark_done_sets_finished_at():
    ctx = RAGContext(question="q")
    assert ctx.finished_at is None
    ctx.mark_done()
    assert ctx.finished_at is not None
    assert ctx.elapsed >= 0


def test_context_request_retry_increments():
    ctx = RAGContext(question="q")
    assert ctx.retry_count == 0
    ctx.request_retry("not grounded")
    assert ctx.retry_count == 1
    assert ctx.metadata["retry_requested"] is True
    assert ctx.metadata["retry_reason"] == "not grounded"


def test_context_to_dict_shape():
    ctx = RAGContext(question="q", session_id="s")
    ctx.answer = "answer"
    ctx.grounded = True
    ctx.route = "internal_docs"
    ctx.sources = [Source(source="doc1", quote="q1", relevance_score=0.9)]
    d = ctx.to_dict()
    assert d["question"] == "q"
    assert d["session_id"] == "s"
    assert d["answer"] == "answer"
    assert d["grounded"] is True
    assert d["route"] == "internal_docs"
    assert len(d["sources"]) == 1
    assert d["sources"][0]["source"] == "doc1"


def test_source_to_dict():
    s = Source(source="PMID:123", quote="text", relevance_score=0.5)
    d = s.to_dict()
    assert d == {"source": "PMID:123", "quote": "text", "relevance_score": 0.5}