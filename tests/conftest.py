"""
Shared test fixtures.

Environment variables are written at module import time, before anything in
`app` is imported, because `app.main` reads settings while building the
FastAPI object. Setting them in a fixture would be too late.

Location: tests/conftest.py
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="tuk-test-")

os.environ.update(
    {
        "TUK_ENVIRONMENT": "test",
        "TUK_SECRET_KEY": "test-secret-key-0123456789abcdef0123456789abcdef",
        "TUK_DATA_DIR": os.path.join(_TMP, "data"),
        "TUK_CORPUS_DIR": os.path.join(_TMP, "corpus"),
        "TUK_PUBMED_INDEX_DIR": os.path.join(_TMP, "pubmed_index"),
        "TUK_WARM_MODEL_ON_BOOT": "false",
        "TUK_RATE_LIMIT_PER_MINUTE": "1000",
        "TUK_RATE_LIMIT_BURST": "1000",
        "TUK_OLLAMA_HOST": "http://127.0.0.1:1",
        "TUK_LLM_TIMEOUT_SECONDS": "5",
        "TUK_LLM_QUEUE_TIMEOUT_SECONDS": "1",
        "TUK_MAX_CONCURRENT_LLM_CALLS": "2",
    }
)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_database(request):
    # Integration tests deliberately run against the REAL data directory (the
    # real corpus index and a live Ollama), so wiping it would destroy local
    # work rather than protect the suite.
    if request.node.get_closest_marker("integration"):
        yield
        return

    from app.services import database

    database.init_db()
    yield
    with database.get_connection() as conn:
        conn.executescript(
            """
            DELETE FROM messages;
            DELETE FROM sessions;
            DELETE FROM query_log;
            DELETE FROM chunks;
            DELETE FROM documents;
            DELETE FROM users;
            """
        )


@pytest.fixture
def settings():
    from app.config import get_settings

    return get_settings()


@pytest.fixture
def fake_pipeline():
    """Stands in for a real Pipeline: no Ollama, no FAISS, no network."""

    class FakePipeline:
        answer = "Second semester exams start on 5 May 2026. [Calendar]"

        def __init__(self):
            self.stream_enabled = True

        def run(self, context):
            from app.rag.core.context import Source

            context.route = "internal_docs"
            context.retrieved_chunks = [{"text": "exams start 5 May", "metadata": {}}]
            context.sources = [Source(source="2026 Academic Year Calendar.docx", quote="5 May")]
            context.answer = self.answer
            context.grounded = True
            context.grounded_reason = "fake"
            return context.mark_done()

        def stream(self, context):
            self.run(context)
            yield {"type": "route", "route": context.route, "reason": ""}
            yield {
                "type": "sources",
                "sources": [s.to_dict() for s in context.sources],
                "chunks_found": len(context.retrieved_chunks),
            }
            for token in self.answer.split(" "):
                yield {"type": "token", "content": token + " "}
            context.answer = self.answer
            yield {"type": "done", "context": context.to_dict()}

        def describe(self):
            return {"agent_count": 4}

    return FakePipeline()


@pytest.fixture
def client(monkeypatch, fake_pipeline):
    """
    TestClient with the heavy dependencies replaced: the embedding model and
    Ollama never load, so the API layer is tested as a web layer rather than
    as an integration test against a GPU-less box.
    """
    import app.main as main_module
    import app.services.chat as chat_module
    from app.ingestion.index import CorpusIndex
    from fastapi.testclient import TestClient

    class StubIndex:
        def __init__(self):
            self.stats = {
                "total_chunks": 12,
                "dimension": 384,
                "embedding_model": "stub",
                "index_type": "stub",
                "vectors_stored_in": "stub",
            }

        def get_stats(self):
            return self.stats

        def ingest_path(self, path, uploaded_by=None):
            return {"status": "indexed", "filename": path.name, "document_id": 1, "chunks": 3}

        def remove_document(self, document_id):
            # Mirror the real store: deactivating is what hides a document
            # from retrieval. Chunk rows themselves live in SQLite regardless.
            from app.services import database

            database.deactivate_document(document_id)
            return database.document_chunk_count(document_id)

        def reload(self):
            return None

    stub = StubIndex()
    monkeypatch.setattr(main_module, "get_index", lambda: stub)
    monkeypatch.setattr("app.api.admin.get_index", lambda: stub)
    monkeypatch.setattr("app.api.health.get_index", lambda: stub)
    monkeypatch.setattr(chat_module, "get_pipeline", lambda name="tuk": fake_pipeline)
    monkeypatch.setattr("app.api.admin.model_status", lambda timeout=4.0: {
        "reachable": True, "models": ["fake"], "error": None,
        "expected_model": "fake", "model_present": True,
    })
    monkeypatch.setattr(CorpusIndex, "get_stats", lambda self: stub.stats, raising=False)

    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def admin_token(client, settings):
    from app.services import database
    from app.services.auth import hash_password

    database.create_user("Test Admin", "admin@example.com", hash_password("sup3rs3cret-pass"), "admin")
    response = client.post(
        "/api/auth/login", json={"email": "admin@example.com", "password": "sup3rs3cret-pass"}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
