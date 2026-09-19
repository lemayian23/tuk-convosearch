"""
Modular RAG framework for TUK-ConvoSearch.

Public API:
    from app.rag import Pipeline, RAGContext, Agent, Source
    from app.rag import RouterAgent, RetrieverAgent, FaissStore

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
from app.rag.stores.base_store import BaseVectorStore
from app.rag.stores.faiss_store import FaissStore
from app.rag.agents import RetrieverAgent, RouterAgent, SynthesizerAgent, CriticAgent

__all__ = [
    "RAGContext",
    "Source",
    "Agent",
    "RetrievalAgent",
    "GenerationAgent",
    "ValidationAgent",
    "Pipeline",
    "BaseVectorStore",
    "FaissStore",
    "RetrieverAgent",
    "RouterAgent",
    "SynthesizerAgent",
    "CriticAgent"
]

__version__ = "0.1.0"