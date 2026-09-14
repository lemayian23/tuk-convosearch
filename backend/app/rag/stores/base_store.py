"""
Abstract VectorStore interface.

All retrieval backends (FAISS, PubMed, Chroma, pgvector) implement this
contract so agents never depend on a specific store.

Location: backend/app/rag/stores/base_store.py
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):
    """
    Contract for a vector store used by RetrieverAgent.

    Implementations must:
    - Accept text chunks with arbitrary metadata (add_chunks)
    - Return ranked results with a relevance_score in [0, 1] (search)
    - Report total_chunks and dimension (get_stats)
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Persist chunks and their embeddings.

        Args:
            chunks: list of {'text': str, 'metadata': dict}

        Returns:
            Number of chunks added.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k chunks most similar to `query`.

        Returns:
            List of dicts, each with keys:
                'text': str
                'metadata': dict
                'distance': float
                'relevance_score': float  (higher = more relevant, in [0, 1])
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Return store diagnostics, e.g.:
            {'total_chunks': int, 'dimension': int, 'index_type': str}
        """
        raise NotImplementedError

    def clear_all(self) -> None:
        """Optional: reset the store. Default no-op."""
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} stats={self.get_stats()}>"