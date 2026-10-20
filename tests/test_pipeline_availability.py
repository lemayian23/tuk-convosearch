"""
The cancer pipeline depends on a local PubMed snapshot that most servers will
not carry. These tests pin the failure mode: a student must be told the corpus
is not installed, not wait twenty seconds for an empty index to say "nothing
found".

Location: tests/test_pipeline_availability.py
"""

import json

import pytest

from app.services.pipelines import (
    PipelineUnavailableError,
    _pubmed_corpus_present,
    available_pipelines,
    get_pipeline,
    reset_pipelines,
)

MESSAGE = "cancer corpus is not installed"


@pytest.fixture
def pubmed_corpus(tmp_path, monkeypatch, settings):
    """Point the setting at a directory and install the two index files."""
    root = tmp_path / "pubmed_index"
    root.mkdir()
    monkeypatch.setattr(settings, "pubmed_index_dir", root)
    (root / "faiss_index.bin").write_bytes(b"")
    (root / "metadata.json").write_text("[]", encoding="utf-8")
    return root


def test_missing_corpus_is_detected(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "pubmed_index_dir", tmp_path / "absent")
    assert _pubmed_corpus_present(settings) is False


def test_partial_corpus_still_counts_as_missing(settings, monkeypatch, tmp_path):
    """The .bin without its metadata would make PubMedStore rebuild from scratch."""
    root = tmp_path / "half"
    root.mkdir()
    (root / "faiss_index.bin").write_bytes(b"")
    monkeypatch.setattr(settings, "pubmed_index_dir", root)
    assert _pubmed_corpus_present(settings) is False


def test_a_pickle_snapshot_is_not_recognised(settings, monkeypatch, tmp_path):
    """
    pubmed_index/ used to be committed and restored with pickle.load, which runs
    whatever the file holds. It now comes off rsync instead of git, which is the
    same untrusted bytes, so the store reads JSON only - an old snapshot simply
    looks absent rather than executing.
    """
    root = tmp_path / "old_format"
    root.mkdir()
    (root / "faiss_index.bin").write_bytes(b"")
    (root / "metadata.pkl").write_bytes(b"\x80\x04\x95")
    monkeypatch.setattr(settings, "pubmed_index_dir", root)
    assert _pubmed_corpus_present(settings) is False


def test_installed_corpus_is_detected(pubmed_corpus, settings):
    assert _pubmed_corpus_present(settings) is True


def test_cancer_is_not_advertised_without_the_corpus(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "pubmed_index_dir", tmp_path / "absent")
    assert list(available_pipelines()) == ["tuk"]


def test_cancer_is_advertised_once_installed(pubmed_corpus, settings):
    assert "cancer" in available_pipelines()


def test_cancer_request_fails_before_loading_a_model(settings, monkeypatch, tmp_path):
    """
    The guard has to run before _build: PubMedStore loads a second Sentence-
    Transformer into RAM before it discovers there is nothing to read.
    """
    import app.services.pipelines as pipelines

    monkeypatch.setattr(settings, "pubmed_index_dir", tmp_path / "absent")

    def explode(*args, **kwargs):
        raise AssertionError("_build must not run for an unavailable pipeline")

    monkeypatch.setattr(pipelines, "_build", explode)

    with pytest.raises(PipelineUnavailableError) as exc:
        get_pipeline("cancer")
    assert "pubmed_index" in str(exc.value)


def test_unknown_pipeline_falls_back_to_tuk(monkeypatch, fake_pipeline):
    import app.services.pipelines as pipelines

    reset_pipelines()
    monkeypatch.setattr(pipelines, "_build", lambda name, settings: fake_pipeline)
    try:
        assert get_pipeline("legal") is fake_pipeline
    finally:
        reset_pipelines()


def test_chat_returns_404_for_an_unavailable_pipeline(client, monkeypatch):
    from app.api import chat as chat_api

    class UnavailableService:
        async def answer(self, *args, **kwargs):
            raise PipelineUnavailableError(MESSAGE)

    monkeypatch.setattr(chat_api, "get_chat_service", lambda: UnavailableService())

    response = client.post("/api/chat", json={"message": "PD-L1 inhibitors?", "pipeline": "cancer"})
    assert response.status_code == 404
    assert MESSAGE in response.json()["detail"]


def test_stream_reports_the_real_reason(client, monkeypatch):
    """
    Without this branch the generic handler answers with the polite fallback,
    and the student has no way to learn the corpus simply is not on the server.
    """
    from app.api import chat as chat_api

    class UnavailableService:
        async def stream(self, *args, **kwargs):
            raise PipelineUnavailableError(MESSAGE)
            yield {}

    monkeypatch.setattr(chat_api, "get_chat_service", lambda: UnavailableService())

    with client.stream(
        "POST", "/api/chat/stream", json={"message": "PD-L1?", "pipeline": "cancer"}
    ) as response:
        payload = "".join(response.iter_text())

    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in payload.splitlines()
        if line.startswith("data: ")
    ]
    errors = [e for e in events if e["type"] == "error"]
    assert errors and errors[0]["code"] == 404
    assert MESSAGE in errors[0]["message"]


def test_config_endpoint_and_schema_stay_in_sync(client):
    """The UI builds its picker from /api/config; the validator must match it."""
    from app.api.schemas import ChatRequest

    advertised = set(client.get("/api/config").json()["pipelines"])
    assert ChatRequest.model_fields["pipeline"].default == "tuk"
    assert "tuk" in advertised


def test_snapshot_round_trips_as_json(tmp_path, monkeypatch):
    """
    The embedder is replaced because S-PubMedBert-MS-MARCO is ~400 MB and needs
    the network; the store's own read/write path is exercised for real.
    """
    import faiss
    import numpy as np

    from app.rag.stores import pubmed_store as module

    class StubEmbedder:
        def encode(self, texts, show_progress_bar=False):
            rows = 1 if isinstance(texts, str) else len(texts)
            return np.zeros((rows, 8), dtype="float32")

    monkeypatch.setattr(module, "SentenceTransformer", lambda *a, **k: StubEmbedder())

    root = tmp_path / "pubmed_index"
    root.mkdir()
    faiss.write_index(faiss.IndexFlatL2(8), str(root / "faiss_index.bin"))
    (root / "metadata.json").write_text("[]", encoding="utf-8")

    store = module.PubMedStore(persist_directory=str(root), dimension=8)
    store.add_chunks([{"text": "PD-L1 inhibitors", "metadata": {"pmid": "12345"}}])

    on_disk = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk[0]["metadata"]["pmid"] == "12345"
    assert not (root / "metadata.pkl").exists()

    reopened = module.PubMedStore(persist_directory=str(root), dimension=8)
    assert reopened.metadata == on_disk


def test_pubmed_store_never_deserialises_a_pickle():
    """
    The point of the JSON switch: a metadata.pkl sitting in the mounted
    directory is now an unrecognised file instead of code the server runs.
    """
    from pathlib import Path

    from app.rag.stores import pubmed_store as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in source
    assert "pickle.load" not in source
