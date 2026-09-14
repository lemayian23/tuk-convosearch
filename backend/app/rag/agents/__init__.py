"""
Pipeline agents.

Each agent implements the Agent interface and operates on a shared
RAGContext. Agents are composed into Pipelines via configs.
"""

from app.rag.agents.retriever_agent import RetrieverAgent

__all__ = [
    "RetrieverAgent",
]