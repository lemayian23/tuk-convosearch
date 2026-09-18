"""Abstract VectorStore interface. Location: backend/app/rag/stores/base_store.py"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def clear_all(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} stats={self.get_stats()}>"