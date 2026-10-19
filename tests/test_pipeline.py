"""Unit tests for the Pipeline orchestrator."""

import pytest

from app.rag.core.base_agent import Agent
from app.rag.core.context import RAGContext
from app.rag.core.pipeline import Pipeline


class DummyAgent(Agent):
    """Agent that writes a fixed value to the context."""

    def __init__(self, name, key, value):
        self.name = name
        self.key = key
        self.value = value

    def run(self, ctx):
        ctx.metadata[self.key] = self.value
        return ctx


def test_pipeline_requires_at_least_one_agent():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_runs_agents_in_order():
    a = DummyAgent("A", "step_a", 1)
    b = DummyAgent("B", "step_b", 2)
    p = Pipeline([a, b])
    ctx = p.run(RAGContext(question="q"))
    assert ctx.metadata["step_a"] == 1
    assert ctx.metadata["step_b"] == 2


def test_pipeline_short_circuits_off_topic():
    class Router(Agent):
        name = "Router"

        def run(self, ctx):
            ctx.route = "off_topic"
            ctx.answer = "not relevant"
            return ctx

    class ShouldNotRun(Agent):
        name = "ShouldNotRun"

        def run(self, ctx):
            raise AssertionError("This agent should not have run")

    p = Pipeline([Router(), ShouldNotRun()])
    ctx = p.run(RAGContext(question="q"))
    assert ctx.route == "off_topic"
    assert ctx.answer == "not relevant"


def test_pipeline_short_circuits_small_talk():
    class Router(Agent):
        name = "Router"

        def run(self, ctx):
            ctx.route = "small_talk"
            ctx.answer = "hi!"
            return ctx

    p = Pipeline([Router()])
    ctx = p.run(RAGContext(question="hi"))
    assert ctx.route == "small_talk"
    assert ctx.answer == "hi!"


def test_pipeline_retries_when_agent_requests():
    """An agent that requests retry should re-trigger the retry-from agent."""

    class Generator(Agent):
        name = "Generator"

        def run(self, ctx):
            ctx.answer = f"attempt_{ctx.retry_count}"
            return ctx

    class Critic(Agent):
        name = "Critic"

        def run(self, ctx):
            if ctx.retry_count < 2:
                ctx.request_retry("test retry")
            else:
                ctx.grounded = True
            return ctx

    p = Pipeline([Generator(), Critic()], retry_from_index=0)
    ctx = p.run(RAGContext(question="q", max_retries=2))
    assert ctx.retry_count == 2
    assert ctx.grounded is True


def test_streaming_answer_is_not_doubled():
    """Regression: the pipeline appended tokens the agent had already
    recorded, so every word reached the student twice."""

    class StreamingGenerator(Agent):
        name = "Generator"

        def run(self, ctx):
            raise AssertionError("a streaming agent must not be run()")

        def stream(self, ctx):
            for token in ("Hel", "lo ", "world"):
                ctx.answer_tokens.append(token)
                yield {"type": "token", "content": token}

    ctx = RAGContext(question="q")
    events = list(Pipeline([StreamingGenerator()]).stream(ctx))

    assert [e["content"] for e in events if e["type"] == "token"] == ["Hel", "lo ", "world"]
    assert ctx.answer == "Hello world"


def test_pipeline_describe():
    a = DummyAgent("A", "k", "v")
    p = Pipeline([a])
    d = p.describe()
    assert d["agent_count"] == 1
    assert d["agents"][0]["name"] == "A"