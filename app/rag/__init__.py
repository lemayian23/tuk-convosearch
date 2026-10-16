"""
Modular RAG framework for TUK-ConvoSearch.

Public API:
    from app.rag import Pipeline, RAGContext, Agent, Source
    from app.rag import RouterAgent, RetrieverAgent

Core agents and context are always available.
Store backends (FAISS, PubMed) require the [rag] extras:

    pip install tuk-convosearch[rag]

Agents live in app.rag.agents.*
Stores live in app.rag.stores.*
Pipeline configs live in app.rag.configs.*
"""

from app.rag.core.context import RAGContext, Source
from app.rag.core.base_agent import (
    Agent,
    RetrievalAgent,
    GenerationAgent,
    ValidationAgent,
)
from app.rag.core.pipeline import Pipeline

from app.rag.agents import RetrieverAgent, RouterAgent, SynthesizerAgent, CriticAgent

__all__ = [
    "RAGContext",
    "Source",
    "Agent",
    "RetrievalAgent",
    "GenerationAgent",
    "ValidationAgent",
    "Pipeline",
    "RetrieverAgent",
    "RouterAgent",
    "SynthesizerAgent",
    "CriticAgent",
]

__version__ = "0.1.0"


# Optional: QueryExpansionAgent (no extra deps, but defined per-pipeline)
try:
    from app.rag.agents import QueryExpansionAgent
    __all__.append("QueryExpansionAgent")
except ImportError:
    pass


# Optional: Store backends (require faiss + sentence-transformers)
try:
    from app.rag.stores import BaseVectorStore, FaissStore, PubMedStore
    __all__.extend(["BaseVectorStore", "FaissStore", "PubMedStore"])
except ImportError:
    # [rag] extras not installed — stores unavailable, but pipeline core works
    BaseVectorStore = None
    FaissStore = None
    PubMedStore = None


# Optional: Pipeline configs (also require [rag] extras)
try:
    from app.rag.configs import build_tuk_pipeline, build_cancer_pipeline
    __all__.extend(["build_tuk_pipeline", "build_cancer_pipeline"])
except ImportError:
    build_tuk_pipeline = None
    build_cancer_pipeline = None