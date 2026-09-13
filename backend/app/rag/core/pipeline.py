"""
Sequential Pipeline orchestrator for modular RAG agents.

Runs a list of Agent instances in order against a shared RAGContext,
supporting retry loops and short-circuit routes (off_topic, small_talk).

Location: backend/app/rag/core/pipeline.py
"""

from typing import List, Generator, Optional, Dict, Any
from app.rag.core.base_agent import Agent
from app.rag.core.context import RAGContext


# Routes that should skip the rest of the pipeline (no retrieval/synthesis needed)
SHORT_CIRCUIT_ROUTES = {"off_topic", "small_talk"}


class Pipeline:
    """
    Orchestrates a sequence of Agents over a single RAGContext.

    Usage:
        pipeline = Pipeline([
            RouterAgent(...),
            RetrieverAgent(...),
            SynthesizerAgent(...),
            CriticAgent(...),
        ])
        ctx = pipeline.run(RAGContext(question="...", session_id="abc"))
        print(ctx.answer, ctx.sources)

    Retry semantics:
        If an agent calls context.request_retry(), the pipeline resets
        context.answer and re-runs from `retry_from_index` (default:
        Synthesizer, i.e. the first GenerationAgent) up to max_retries.
    """

    def __init__(
        self,
        agents: List[Agent],
        retry_from_index: Optional[int] = None,
        short_circuit_routes: Optional[set] = None,
    ):
        if not agents:
            raise ValueError("Pipeline requires at least one agent")

        self.agents = agents
        # Default: retry from the first GenerationAgent (Synthesizer)
        if retry_from_index is None:
            retry_from_index = self._find_first_generation_index()
        self.retry_from_index = retry_from_index
        self.short_circuit_routes = short_circuit_routes or SHORT_CIRCUIT_ROUTES

        print(f"✓ Pipeline initialized with {len(self.agents)} agents:")
        for i, a in enumerate(self.agents):
            marker = " [retry-from]" if i == self.retry_from_index else ""
            print(f"    {i}. {a.name}{marker}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _find_first_generation_index(self) -> int:
        """Locate the first GenerationAgent so retries skip retrieval."""
        from app.rag.core.base_agent import GenerationAgent
        for i, agent in enumerate(self.agents):
            if isinstance(agent, GenerationAgent):
                return i
        # Fallback: retry from the last agent if no explicit generator
        return max(0, len(self.agents) - 1)

    def _should_short_circuit(self, context: RAGContext) -> bool:
        return context.route in self.short_circuit_routes

    def _reset_for_retry(self, context: RAGContext) -> None:
        """Clear generation outputs before re-running from retry_from_index."""
        context.answer = ""
        context.answer_tokens = []
        context.grounded = False
        context.grounded_reason = ""
        context.metadata.pop("retry_requested", None)
        context.metadata.pop("retry_reason", None)

    # ------------------------------------------------------------------ #
    # Blocking execution
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        """Run all agents to completion. Returns the mutated context."""
        while context.retry_count <= context.max_retries:
            context.metadata.pop("retry_requested", None)

            for i, agent in enumerate(self.agents):
                # Skip earlier agents on a retry (e.g. don't re-run Router)
                if context.retry_count > 0 and i < self.retry_from_index:
                    continue

                try:
                    context = agent.run(context)
                except Exception as e:
                    context.error = f"{agent.name} failed: {e}"
                    print(f"  ✗ {context.error}")
                    return context.mark_done()

                # Short-circuit: Router classified as off_topic / small_talk
                if i == 0 and self._should_short_circuit(context):
                    return context.mark_done()

            # If Critic requested a retry, loop; otherwise we're done
            if context.metadata.get("retry_requested"):
                if context.retry_count >= context.max_retries:
                    print(f"  ⚠ Max retries ({context.max_retries}) reached; accepting answer")
                    break
                print(f"  ↻ Retry {context.retry_count}/{context.max_retries} — "
                      f"{context.metadata.get('retry_reason', '')}")
                self._reset_for_retry(context)
                continue

            break

        return context.mark_done()

    # ------------------------------------------------------------------ #
    # Streaming execution
    # ------------------------------------------------------------------ #

    def stream(self, context: RAGContext) -> Generator[Dict[str, Any], None, None]:
        """
        Stream events from agents that support token streaming.

        Yields dicts shaped as:
          {'type': 'route',   'route': str, 'reason': str}
          {'type': 'sources', 'sources': [...], 'chunks_found': int}
          {'type': 'token',   'content': str}
          {'type': 'retry',   'attempt': int, 'reason': str}
          {'type': 'done',    'context': {...}}
          {'type': 'error',   'message': str}
        """
        while context.retry_count <= context.max_retries:
            context.metadata.pop("retry_requested", None)

            for i, agent in enumerate(self.agents):
                if context.retry_count > 0 and i < self.retry_from_index:
                    continue

                try:
                    # Streaming agent — yield its events directly
                    if type(agent).stream is not Agent.stream:
                        for event in agent.stream(context):
                            if event.get("type") == "token":
                                context.answer_tokens.append(event["content"])
                            yield event
                        # Reassemble full answer after stream completes
                        context.answer = "".join(context.answer_tokens)
                    else:
                        context = agent.run(context)

                except Exception as e:
                    context.error = f"{agent.name} failed: {e}"
                    yield {"type": "error", "message": context.error}
                    yield {"type": "done", "context": context.mark_done().to_dict()}
                    return

                # Emit route info as soon as Router finishes
                if i == 0 and context.route:
                    yield {"type": "route", "route": context.route,
                           "reason": context.route_reason}

                # Short-circuit for off_topic / small_talk
                if i == 0 and self._should_short_circuit(context):
                    yield {"type": "token", "content": context.answer}
                    yield {"type": "done", "context": context.mark_done().to_dict()}
                    return

            # Handle retry loop
            if context.metadata.get("retry_requested"):
                if context.retry_count >= context.max_retries:
                    break
                yield {"type": "retry", "attempt": context.retry_count + 1,
                       "reason": context.metadata.get("retry_reason", "")}
                self._reset_for_retry(context)
                continue

            break

        yield {"type": "done", "context": context.mark_done().to_dict()}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, Any]:
        """Return pipeline structure for admin/debug endpoints."""
        return {
            "agent_count": len(self.agents),
            "retry_from_index": self.retry_from_index,
            "short_circuit_routes": sorted(self.short_circuit_routes),
            "agents": [
                {"index": i, "name": a.name, "class": a.__class__.__name__}
                for i, a in enumerate(self.agents)
            ],
        }