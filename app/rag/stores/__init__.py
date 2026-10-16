"""
Vector store implementations.

BaseVectorStore is always available (no external deps).
FaissStore and PubMedStore require the [rag] extras.
"""

from app.rag.stores.base_store import BaseVectorStore

__all__ = ["BaseVectorStore"]

try:
    from app.rag.stores.faiss_store import FaissStore
    __all__.append("FaissStore")
except ImportError:
    FaissStore = None

try:
    from app.rag.stores.pubmed_store import PubMedStore
    __all__.append("PubMedStore")
except ImportError:
    PubMedStore = None