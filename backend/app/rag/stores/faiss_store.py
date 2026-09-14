"""
FAISS-backed implementation of BaseVectorStore.

Behaviorally identical to the original services/faiss_vector_store.py,
but exposed through the BaseVectorStore contract so agents stay
decoupled from FAISS specifics.

Location: backend/app/rag/stores/faiss_store.py
"""

import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.stores.base_store import BaseVectorStore


class FaissStore(BaseVectorStore):
    def __init__(
        self,
        dimension: int = 384,
        persist_directory: str = "./faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.dimension = dimension
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_file = os.path.join(persist_directory, "metadata.pkl")

        os.makedirs(persist_directory, exist_ok=True)

        print("  Loading embedding model for FAISS...")
        self.embedder = SentenceTransformer(embedding_model)

        if os.path.exists(self.index_file):
            print(f"  Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"  Loaded {len(self.metadata)} chunks")
        else:
            print("  Creating new FAISS index (Flat L2)")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("  New index created")

    # ------------------------------------------------------------------ #
    # BaseVectorStore API
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        print(f"Adding {len(chunks)} chunks to FAISS...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts)

        self.index.add(np.array(embeddings).astype("float32"))

        for chunk in chunks:
            self.metadata.append({
                "id": len(self.metadata),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

        self._save()

        print(f"✓ Added {len(chunks)} chunks to FAISS")
        print(f"  Total chunks in FAISS: {self.index.ntotal}")
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print(f"  FAISS searching for: '{query}'")

        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedder.encode([query])
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )

        results: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk_data = self.metadata[idx]
                distance = float(distances[0][i])
                results.append({
                    "text": str(chunk_data["text"]),
                    "metadata": {
                        "source": str(chunk_data["metadata"].get("source", "unknown")),
                        "file_path": str(chunk_data["metadata"].get("file_path", "")),
                        "chunk_index": int(chunk_data["metadata"].get("chunk_index", 0)),
                    },
                    "distance": distance,
                    "relevance_score": float(1 / (1 + distance)),
                })

        print(f"  Found {len(results)} relevant chunks")
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": int(self.index.ntotal),
            "dimension": int(self.dimension),
            "persist_directory": str(self.persist_directory),
            "index_type": "IndexFlatL2",
        }

    def clear_all(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save()
        print("✓ FAISS index cleared")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)