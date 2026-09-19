"""
PubMedStore — placeholder vector store for cancer research literature.

Extends BaseVectorStore so it drops into any pipeline unchanged.
Real implementation will fetch from NCBI E-utilities / BioC-PMC and
embed with a biomedical model (e.g. PubMedBERT). For now it delegates
to FaissStore so the pipeline can be exercised end-to-end.

Location: app/rag/stores/pubmed_store.py
"""

from typing import List, Dict, Any

from app.rag.stores.base_store import BaseVectorStore
from app.rag.stores.faiss_store import FaissStore


class PubMedStore(BaseVectorStore):
    """
    Stub implementation. Currently wraps a FaissStore.

    TODO (real implementation):
      - Fetch abstracts from NCBI E-utilities by query
      - Chunk + embed with a biomedical embedding model
      - Persist locally to avoid re-fetching
      - Tag chunks with PMID / DOI metadata for citation
    """

    def __init__(
        self,
        persist_directory: str = "./pubmed_index",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        print("  [PubMedStore] STUB — delegating to local FaissStore")
        print("  [PubMedStore] Replace with real NCBI E-utilities fetch later")
        self._delegate = FaissStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )

    # ------------------------------------------------------------------ #
    # BaseVectorStore API — delegate to FaissStore for now
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        return self._delegate.add_chunks(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._delegate.search(query, k=k)

    def get_stats(self) -> Dict[str, Any]:
        stats = self._delegate.get_stats()
        stats["store_type"] = "PubMedStore(stub)"
        return stats

    def clear_all(self) -> None:
        self._delegate.clear_all()