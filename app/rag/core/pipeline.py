"""
Sequential Pipeline orchestrator for modular RAG agents.
Location: backend/app/rag/core/pipeline.py
"""

from typing import List, Generator, Optional, Dict, Any
from app.rag.core.base_agent import Agent
from app.rag.core.context import RAGContext


SHORT_CIRCUIT_ROUTES = {"off_topic", "small_talk"}


class Pipeline:
    def __init__(
        self,
        agents: List[Agent],
        retry_from_index: Optional[int] = None,
        short_circuit_routes: Optional[set] = None,
    ):
        if not agents:
            raise ValueError("Pipeline requires at least one agent")
        self.agents = agents
        if retry_from_index is None:
            retry_from_index = self._find_first_generation_index()
        self.retry_from_index = retry_from_index
        self.short_circuit_routes = short_circuit_routes or SHORT_CIRCUIT_ROUTES

        print(f"✓ Pipeline initialized with {len(self.agents)} agents:")
        for i, a in enumerate(self.agents):
            marker = " [retry-from]" if i == self.retry_from_index else ""
            print(f"    {i}. {a.name}{marker}")

    def _find_first_generation_index(self) -> int:
        from app.rag.core.base_agent import GenerationAgent
        for i, agent in enumerate(self.agents):
            if isinstance(agent, GenerationAgent):
                return i
        return max(0, len(self.agents) - 1)

    def _should_short_circuit(self, context: RAGContext) -> bool:
        return context.route in self.short_circuit_routes

    def _reset_for_retry(self, context: RAGContext) -> None:
        context.answer = ""
        context.answer_tokens = []
        context.grounded = False
        context.grounded_reason = ""
        context.metadata.pop("retry_requested", None)
        context.metadata.pop("retry_reason", None)

    def run(self, context: RAGContext) -> RAGContext:
        while context.retry_count <= context.max_retries:
            context.metadata.pop("retry_requested", None)

            for i, agent in enumerate(self.agents):
                if context.retry_count > 0 and i < self.retry_from_index:
                    continue
                try:
                    context = agent.run(context)
                except Exception as e:
                    context.error = f"{agent.name} failed: {e}"
                    print(f"  ✗ {context.error}")
                    return context.mark_done()

                if i == 0 and self._should_short_circuit(context):
                    return context.mark_done()

            if context.metadata.get("retry_requested"):
                if context.retry_count >= context.max_retries:
                    break
                self._reset_for_retry(context)
                continue

            break

        return context.mark_done()

    def stream(self, context: RAGContext) -> Generator[Dict[str, Any], None, None]:
        while context.retry_count <= context.max_retries:
            context.metadata.pop("retry_requested", None)

            for i, agent in enumerate(self.agents):
                if context.retry_count > 0 and i < self.retry_from_index:
                    continue

                try:
                    if type(agent).stream is not Agent.stream:
                        for event in agent.stream(context):
                            if event.get("type") == "token":
                                context.answer_tokens.append(event["content"])
                            yield event
                        context.answer = "".join(context.answer_tokens)
                    else:
                        context = agent.run(context)
                except Exception as e:
                    context.error = f"{agent.name} failed: {e}"
                    yield {"type": "error", "message": context.error}
                    yield {"type": "done", "context": context.mark_done().to_dict()}
                    return

                if i == 0 and context.route:
                    yield {"type": "route", "route": context.route,
                           "reason": context.route_reason}

                if i == 0 and self._should_short_circuit(context):
                    yield {"type": "token", "content": context.answer}
                    yield {"type": "done", "context": context.mark_done().to_dict()}
                    return

            if context.metadata.get("retry_requested"):
                if context.retry_count >= context.max_retries:
                    break
                yield {"type": "retry", "attempt": context.retry_count + 1,
                       "reason": context.metadata.get("retry_reason", "")}
                self._reset_for_retry(context)
                continue

            break

        yield {"type": "done", "context": context.mark_done().to_dict()}

    def describe(self) -> Dict[str, Any]:
        return {
            "agent_count": len(self.agents),
            "retry_from_index": self.retry_from_index,
            "short_circuit_routes": sorted(self.short_circuit_routes),
            "agents": [
                {"index": i, "name": a.name, "class": a.__class__.__name__}
                for i, a in enumerate(self.agents)
            ],
        }