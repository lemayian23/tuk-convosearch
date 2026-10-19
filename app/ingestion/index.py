"""
CorpusIndex — FAISS over chunk vectors that live in SQLite.

The old store kept vectors in `faiss_index.bin` and their text in a
`metadata.pkl` beside it. That had three costs: loading a pickle written by
another process is a code-execution surface, deleting one document meant
re-embedding the entire corpus, and a backup had to cover two files that can
drift apart. Here the vectors are blobs in `chunks`, FAISS is a derived
in-memory structure rebuilt at boot, and deleting a document is a SQL DELETE
plus a re-derivation that touches no model.

Embeddings are L2-normalised, so the flat L2 search is rank-equivalent to
cosine similarity.

Location: app/ingestion/index.py
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.config import Settings, get_settings
from app.ingestion.documents import (
    ExtractedDocument,
    chunk_document,
    extract_text,
)
from app.rag.stores.base_store import BaseVectorStore
from app.services import database

logger = logging.getLogger(__name__)


class CorpusIndexError(RuntimeError):
    pass


class CorpusIndex(BaseVectorStore):
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._embedder = None
        self._index = faiss.IndexFlatL2(self.settings.embedding_dimension)
        self._records: List[Dict[str, Any]] = []
        self.reload()

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.settings.embedding_model)
            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._get_embedder().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    # ------------------------------------------------------------------ #
    # Index state
    # ------------------------------------------------------------------ #

    def reload(self) -> None:
        """Rebuild the in-memory FAISS index from the chunks table."""
        with self._lock:
            rows = database.iter_chunks()
            if not rows:
                self._index = faiss.IndexFlatL2(self.settings.embedding_dimension)
                self._records = []
                logger.info("Corpus index is empty")
                return

            matrix = np.vstack(
                [np.frombuffer(r["embedding"], dtype="float32") for r in rows]
            )
            if matrix.shape[1] != self.settings.embedding_dimension:
                raise CorpusIndexError(
                    f"Stored vectors are {matrix.shape[1]}-dim but "
                    f"{self.settings.embedding_model} produces "
                    f"{self.settings.embedding_dimension}. Re-run the ingest script."
                )

            index = faiss.IndexFlatL2(self.settings.embedding_dimension)
            index.add(np.ascontiguousarray(matrix))
            self._index = index
            self._records = [
                {"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"]}
                for r in rows
            ]
            logger.info("Corpus index ready: %d chunks", len(self._records))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query_vector = self._encode([query])
            k = min(k, self._index.ntotal)
            distances, indices = self._index.search(
                np.ascontiguousarray(query_vector), k
            )

        results = []
        for position, index_id in enumerate(indices[0]):
            if index_id == -1 or index_id >= len(self._records):
                continue
            record = self._records[index_id]
            distance = float(distances[0][position])
            results.append(
                {
                    "text": record["text"],
                    "metadata": {
                        "source": record["source"],
                        "file_path": "",
                        "chunk_index": 0,
                        "chunk_id": record["chunk_id"],
                    },
                    "distance": distance,
                    "relevance_score": float(1.0 / (1.0 + distance)),
                }
            )
        return results

    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: int) -> int:
        if not chunks:
            return 0
        vectors = self._encode([c["text"] for c in chunks])
        with self._lock:
            database.insert_chunks(
                document_id,
                [
                    {
                        "chunk_index": c["metadata"].get("chunk_index", i),
                        "source": c["metadata"].get("source", "unknown"),
                        "text": c["text"],
                        "embedding": vectors[i].tobytes(),
                    }
                    for i, c in enumerate(chunks)
                ],
            )
            self.reload()
        return len(chunks)

    def wipe(self) -> int:
        """Delete every stored vector. Used by --force re-indexing, never by an endpoint."""
        with self._lock:
            removed = database.clear_all_chunks()
            self.reload()
            return removed

    def clear_all(self) -> None:
        self.wipe()

    # ------------------------------------------------------------------ #
    # Document operations
    # ------------------------------------------------------------------ #

    def ingest_path(self, path: Path, uploaded_by: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract, chunk, embed and store one file. Returns a result dict with
        `status` of 'indexed', 'unchanged' or 'empty'.
        """
        path = Path(path)
        document: ExtractedDocument = extract_text(path)

        if not document.content.strip():
            return {"status": "empty", "filename": path.name, "chunks": 0}

        existing = database.find_document_by_hash(document.sha256)
        if existing and database.document_chunk_count(existing["document_id"]) > 0:
            return {
                "status": "unchanged",
                "filename": path.name,
                "document_id": existing["document_id"],
                "chunks": existing["chunk_count"],
            }

        # Same filename with different content is an update: drop its old
        # chunks, then let upsert_document bump the row's indexed timestamp.
        with self._lock:
            document_id = database.upsert_document(
                filename=path.name,
                title=path.stem,
                file_type=document.file_type,
                sha256=document.sha256,
                chunk_count=0,
                uploaded_by=uploaded_by,
            )
            # upsert_document keys on filename, so this is the same row as any
            # earlier version of this file: drop its stale chunks before adding
            # the new ones.
            database.delete_chunks(document_id)

            chunks = chunk_document(
                document,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            count = self.add_chunks(chunks, document_id)
            database.mark_document_indexed(document_id, count)
            return {
                "status": "indexed",
                "filename": path.name,
                "document_id": document_id,
                "chunks": count,
                "characters": document.content_length,
            }

    def remove_document(self, document_id: int) -> int:
        with self._lock:
            removed = database.delete_chunks(document_id)
            database.deactivate_document(document_id)
            self.reload()
            return removed

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_chunks": int(self._index.ntotal),
                "dimension": int(self.settings.embedding_dimension),
                "embedding_model": self.settings.embedding_model,
                "index_type": "IndexFlatL2 (normalised, cosine-equivalent)",
                "vectors_stored_in": str(self.settings.db_path),
            }


# ---------------------------------------------------------------------- #
# Singleton access — one embedder, one index, shared by all requests
# ---------------------------------------------------------------------- #

_instance: Optional[CorpusIndex] = None
_instance_lock = threading.Lock()


def get_index() -> CorpusIndex:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CorpusIndex()
    return _instance


def reset_index() -> None:
    """Used by tests and by the rebuild script to force a fresh load."""
    global _instance
    with _instance_lock:
        _instance = None
