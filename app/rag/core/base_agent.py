"""
Abstract base classes for the modular RAG framework.
Location: backend/app/rag/core/base_agent.py
"""

from abc import ABC, abstractmethod
from typing import Generator
from app.rag.core.context import RAGContext


class Agent(ABC):
    name: str = "Agent"

    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError

    def stream(self, context: RAGContext) -> Generator[dict, None, None]:
        result = self.run(context)
        yield {"type": "done", "context": result.to_dict()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class RetrievalAgent(Agent):
    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError


class GenerationAgent(Agent):
    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError


class ValidationAgent(Agent):
    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError