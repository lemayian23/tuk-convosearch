"""
SynthesizerAgent — generates the final answer from retrieved context.

Reads:  context.question, context.metadata["context_string"],
        context.metadata["history"]
Writes: context.answer, context.answer_tokens

Supports both blocking (run) and streaming (stream) execution.
Streaming yields {'type': 'token', 'content': ...} events.

Location: backend/app/rag/agents/synthesizer_agent.py
"""

from typing import Generator, Dict, Any, Optional

import ollama

from app.rag.core.base_agent import GenerationAgent
from app.rag.core.context import RAGContext


class SynthesizerAgent(GenerationAgent):
    """
    LLM-backed answer generator.

    Args:
        model_name:      Ollama model tag (e.g. "llama3.2:1b")
        prompt_template: str with {context}, {history}, {question} placeholders
        num_predict:     max tokens to generate
        temperature:     sampling temperature (0.0 = deterministic)
        num_ctx:         context window size
        keep_alive:      Ollama keep_alive (e.g. -1 to keep model resident)
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        prompt_template: Optional[str] = None,
        num_predict: int = 300,
        temperature: float = 0.2,
        num_ctx: int = 2048,
        keep_alive: int = -1,
        name: str = "Synthesizer",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template or self._default_prompt()
        self.num_predict = num_predict
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.name = name

    # ------------------------------------------------------------------ #
    # Agent API — blocking
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        # Skip if router already produced an answer (small_talk / off_topic)
        if context.answer:
            return context

        prompt = self._build_prompt(context)
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self._options(),
                keep_alive=self.keep_alive,
            )
            context.answer = response["message"]["content"]
            context.answer_tokens = [context.answer]
        except Exception as e:
            context.error = f"Synthesis failed: {e}"
            context.answer = f"Error generating answer: {e}"

        return context

    # ------------------------------------------------------------------ #
    # Agent API — streaming
    # ------------------------------------------------------------------ #

    def stream(self, context: RAGContext) -> Generator[Dict[str, Any], None, None]:
        # Skip if router already produced an answer
        if context.answer:
            yield {"type": "token", "content": context.answer}
            return

        prompt = self._build_prompt(context)

        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self._options(),
                keep_alive=self.keep_alive,
                stream=True,
            )
            for part in stream:
                token = part.get("message", {}).get("content", "")
                if token:
                    context.answer_tokens.append(token)
                    yield {"type": "token", "content": token}
        except Exception as e:
            context.error = f"Synthesis failed: {e}"
            error_msg = f"Error generating answer: {e}"
            context.answer_tokens.append(error_msg)
            yield {"type": "token", "content": error_msg}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(self, context: RAGContext) -> str:
        context_str = context.metadata.get("context_string", "")
        history = context.metadata.get("history", "No previous conversation.")
        return self.prompt_template.format(
            context=context_str,
            history=history,
            question=context.question,
        )

    def _options(self) -> Dict[str, Any]:
        return {
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

    @staticmethod
    def _default_prompt() -> str:
        """Generic prompt — pipelines should override with a domain prompt."""
        return (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say you cannot find it.\n"
            "Cite the source document for every fact.\n\n"
            "CONTEXT:\n{context}\n\n"
            "HISTORY:\n{history}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER:"
        )