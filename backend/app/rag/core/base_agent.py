"""
Abstract base classes for the modular RAG framework.

Every agent in the pipeline (Router, Retriever, Synthesizer, Critic)
implements the Agent interface and operates on a shared RAGContext.

Location: backend/app/rag/core/base_agent.py
"""

from abc import ABC, abstractmethod
from typing import Generator, Any
from app.rag.core.context import RAGContext


class Agent(ABC):
    """
    Abstract base class for all RAG pipeline agents.

    Contract:
    - Each agent receives a RAGContext, mutates/reads it, and returns it.
    - Agents must be stateless with respect to individual requests
      (any state must live in the context or be injected at construction).
    - Agents must be safe to call sequentially in a Pipeline.

    Two execution modes:
    - run(context)         -> RAGContext      (blocking, non-streaming)
    - stream(context)      -> Generator[dict] (yields SSE-style events)

    The default stream() falls back to run() for agents that do not
    need token-level streaming (e.g. Router, Retriever, Critic).
    """

    #: Human-readable agent name, used in logs and pipeline introspection.
    name: str = "Agent"

    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        """
        Execute the agent's logic against the given context.

        Implementations MUST:
        - Read from context (question, retrieved_chunks, etc.)
        - Write their output to context (route, answer, sources, etc.)
        - Return the same context object (mutated in place)

        Implementations MUST NOT:
        - Raise on expected failure modes (empty retrieval, LLM error)
          -> instead set context.error / context.route appropriately
        """
        raise NotImplementedError

    def stream(self, context: RAGContext) -> Generator[dict, None, None]:
        """
        Streaming variant. Default: run() to completion, then emit a
        single 'done' event carrying the final context snapshot.

        Agents that generate tokens incrementally (Synthesizer) should
        override this to yield {'type': 'token', 'content': ...} events.
        """
        result = self.run(context)
        yield {"type": "done", "context": result.to_dict()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class RetrievalAgent(Agent):
    """
    Marker subclass for agents that only read/annotate retrieval state.
    Useful for type-checking pipeline composition (e.g. requiring at
    least one RetrievalAgent before a SynthesizerAgent).
    """

    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError


class GenerationAgent(Agent):
    """
    Marker subclass for agents that produce the final natural-language
    answer. Pipelines should contain exactly one GenerationAgent.
    """

    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError


class ValidationAgent(Agent):
    """
    Marker subclass for agents that validate / post-process a generated
    answer (e.g. grounding checks, medical fact verification). May
    trigger retries by mutating context.retry_count and signalling via
    context.metadata['retry_requested'] = True.
    """

    @abstractmethod
    def run(self, context: RAGContext) -> RAGContext:
        raise NotImplementedError