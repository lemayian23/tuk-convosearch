"""
SynthesizerAgent — generates the final answer from retrieved context.

Reads:  context.question, context.metadata["context_string"],
        context.metadata["history"]
Writes: context.answer, context.answer_tokens

Supports both blocking (run) and streaming (stream) execution.
Streaming yields {'type': 'token', 'content': ...} events.

Location: app/rag/agents/synthesizer_agent.py
"""

from typing import Generator, Dict, Any, Optional, Union

from app.rag.core.base_agent import GenerationAgent
from app.rag.core.context import RAGContext

NO_ANSWER = "I could not form an answer just now. Please ask again in a moment."


class SynthesizerAgent(GenerationAgent):
    """
    LLM-backed answer generator.

    Args:
        model_name:      Ollama model tag (e.g. "llama3.2:1b")
        prompt_template: str with {context}, {history}, {question} placeholders
        num_predict:     max tokens to generate
        temperature:     sampling temperature (0.0 = deterministic)
        num_ctx:         context window size
        keep_alive:      Ollama keep_alive — seconds as an int ("stay resident")
                       or a duration string like "30m"
        repeat_penalty:  discourage repetition (small models need this)
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        prompt_template: Optional[str] = None,
        num_predict: int = 300,
        temperature: float = 0.2,
        num_ctx: int = 2048,
        keep_alive: Union[int, str] = -1,
        repeat_penalty: float = 1.1,
        name: str = "Synthesizer",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template or self._default_prompt()
        self.num_predict = num_predict
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.repeat_penalty = repeat_penalty
        self.name = name

    # ------------------------------------------------------------------ #
    # Agent API — blocking
    # ------------------------------------------------------------------ #

    def run(self, context: RAGContext) -> RAGContext:
        # Skip if router already produced an answer (small_talk / off_topic)
        if context.answer:
            return context

        from app.services.ollama import client as ollama_client

        prompt = self._build_prompt(context)
        try:
            response = ollama_client().chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self._options(),
                keep_alive=self.keep_alive,
            )
            context.answer = response["message"]["content"]
            context.answer_tokens = [context.answer]
        except Exception as e:
            context.error = f"Synthesis failed: {e}"
            # The exception text names hosts and ports; the student only gets
            # a sentence, while /api/chat logs context.error for the operator.
            context.answer = NO_ANSWER
            context.answer_tokens = [NO_ANSWER]

        return context

    # ------------------------------------------------------------------ #
    # Agent API — streaming
    # ------------------------------------------------------------------ #

    def stream(self, context: RAGContext) -> Generator[Dict[str, Any], None, None]:
        # Skip if router already produced an answer
        if context.answer:
            yield {"type": "token", "content": context.answer}
            return

        from app.services.ollama import client as ollama_client

        prompt = self._build_prompt(context)

        try:
            stream = ollama_client().chat(
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
            context.answer_tokens.append(NO_ANSWER)
            yield {"type": "token", "content": NO_ANSWER}

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
            "repeat_penalty": self.repeat_penalty,
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