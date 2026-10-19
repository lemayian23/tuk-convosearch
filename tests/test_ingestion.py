"""
Tests for upload safety, extraction, chunking and the vector index.

The embedder is stubbed: these tests must run in CI without downloading
model weights, and none of the behaviour under test depends on real vectors.

Location: tests/test_ingestion.py
"""

import numpy as np
import pytest

from app.ingestion.documents import (
    chunk_text,
    extract_text,
    sanitize_filename,
    save_upload,
)
from app.ingestion.index import CorpusIndex
from app.services import database


@pytest.fixture
def deterministic_index(monkeypatch, settings):
    """CorpusIndex whose 'embeddings' are derived from the first letter."""
    index = CorpusIndex(settings)

    def fake_encode(texts):
        rows = []
        for text in texts:
            vector = [0.0] * settings.embedding_dimension
            vector[ord(text[0].lower()) % settings.embedding_dimension] = 1.0
            rows.append(vector)
        return np.asarray(rows, dtype="float32")

    monkeypatch.setattr(index, "_encode", fake_encode)
    return index


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Path components before the basename are discarded entirely.
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\evil.exe", "evil.exe"),
            ("/absolute/path/report.pdf", "report.pdf"),
            ("normal-file name.docx", "normal-file-name.docx"),
            ("café-notice.pdf", "cafe-notice.pdf"),
            ("", "document"),
            ("....", "document"),
            ("...", "document"),
        ],
    )
    def test_traversal_and_junk_are_neutralised(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_long_names_are_capped(self):
        assert len(sanitize_filename("x" * 500)) <= 90

    def test_no_path_separators_survive(self):
        for raw in ("a/b/c.txt", "..\\..\\x.txt", "%2e%2e%2f.txt"):
            cleaned = sanitize_filename(raw)
            assert "/" not in cleaned and "\\" not in cleaned


class TestSaveUpload:
    def test_writes_into_the_given_directory(self, tmp_path):
        target = save_upload(b"hello", "notice.txt", tmp_path)
        assert target.parent == tmp_path
        assert target.read_bytes() == b"hello"

    def test_refuses_unsupported_extension(self, tmp_path):
        with pytest.raises(ValueError):
            save_upload(b"MZ", "payload.exe", tmp_path)

    def test_refuses_empty_upload(self, tmp_path):
        with pytest.raises(ValueError):
            save_upload(b"", "empty.txt", tmp_path)


class TestChunking:
    def test_empty_text_produces_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_chunks_respect_the_size_cap(self):
        text = "\n\n".join(f"Paragraph {i} with a decent amount of words in it." for i in range(40))
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert chunks
        assert all(len(c["text"]) <= 200 for c in chunks)

    def test_oversized_single_paragraph_is_split_not_dropped(self):
        chunks = chunk_text("word " * 400, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c["text"]) <= 200 for c in chunks)

    def test_overlap_carries_text_forward(self):
        text = "A" * 180 + "\n\n" + "B" * 20
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) == 2
        assert chunks[1]["text"].startswith("A" * 20)

    def test_chunk_index_is_sequential(self):
        chunks = chunk_text("one\n\ntwo\n\nthree", chunk_size=5, chunk_overlap=0)
        assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


class TestExtraction:
    def test_txt_roundtrip(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("Fees are payable before registration.", encoding="utf-8")
        document = extract_text(path)
        assert document.content.startswith("Fees are payable")
        assert len(document.sha256) == 64

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "sheet.xlsx"
        path.write_bytes(b"zip")
        with pytest.raises(ValueError):
            extract_text(path)


class TestCorpusIndex:
    def test_ingest_then_search_finds_the_document(self, deterministic_index, tmp_path):
        path = tmp_path / "calendar.txt"
        path.write_text("Second semester examinations start on 5 May 2026.", encoding="utf-8")

        result = deterministic_index.ingest_path(path)
        assert result["status"] == "indexed"
        assert result["chunks"] >= 1

        hits = deterministic_index.search("Second", k=3)
        assert hits
        assert "5 May 2026" in hits[0]["text"]
        assert hits[0]["metadata"]["source"] == "calendar.txt"
        assert 0 < hits[0]["relevance_score"] <= 1

    def test_same_content_is_not_reindexed(self, deterministic_index, tmp_path):
        path = tmp_path / "notice.txt"
        path.write_text("Exams rescheduled to 12 May.", encoding="utf-8")
        first = deterministic_index.ingest_path(path)
        second = deterministic_index.ingest_path(path)

        assert second["status"] == "unchanged"
        assert second["document_id"] == first["document_id"]
        assert deterministic_index.get_stats()["total_chunks"] == first["chunks"]

    def test_removing_one_document_keeps_the_others(self, deterministic_index, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("Alpha faculty fees are due in January.", encoding="utf-8")
        b.write_text("Beta library opening hours are listed here.", encoding="utf-8")

        result_a = deterministic_index.ingest_path(a)
        deterministic_index.ingest_path(b)

        removed = deterministic_index.remove_document(result_a["document_id"])
        assert removed == result_a["chunks"]

        hits = deterministic_index.search("Beta", k=5)
        sources = {hit["metadata"]["source"] for hit in hits}
        assert "a.txt" not in sources

    def test_empty_file_is_reported_not_indexed(self, deterministic_index, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("", encoding="utf-8")
        assert deterministic_index.ingest_path(path)["status"] == "empty"
        assert deterministic_index.get_stats()["total_chunks"] == 0

    def test_vectors_survive_a_process_restart(self, deterministic_index, settings, tmp_path):
        path = tmp_path / "persist.txt"
        path.write_text("Graduation requires 90 credit hours.", encoding="utf-8")
        result = deterministic_index.ingest_path(path)

        reloaded = CorpusIndex(settings)
        # No embedder needed: reload reads the stored vectors out of SQLite.
        assert reloaded.get_stats()["total_chunks"] == result["chunks"]
        assert database.document_chunk_count(result["document_id"]) == result["chunks"]

    def test_search_on_an_empty_index_returns_nothing(self, deterministic_index):
        assert deterministic_index.search("anything", k=5) == []
