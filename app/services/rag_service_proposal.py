"""
RAG Service — modular pipeline wrapper.

Delegates all routing/retrieval/synthesis/critique to the modular
RAG pipeline in app.rag.configs.tuk_pipeline.

Public API (unchanged):
    answer_question(question, session_id, k) -> dict
    stream_answer(question, session_id, k)   -> generator of SSE events
    get_stats()                              -> dict
    clear_cache() / clear_history(session_id)

Location: app/services/rag_service_proposal.py
"""

import time
from typing import Dict, Any, Generator

from app.services import database
from app.rag import RAGContext
from app.rag.configs.tuk_pipeline import build_tuk_pipeline


# Cache for repeated questions
cache = {}
cache_ttl = 3600

# Conversation memory
conversation_memory = {}


class RAGServiceProposal:
    """
    Thin wrapper around the modular RAG pipeline.

    Keeps the original public surface (answer_question / stream_answer)
    while delegating all RAG logic to the composed Pipeline.
    """

    def __init__(self, model_name: str = "llama3.2:1b", critic_mode: str = "heuristic"):
        self.model_name = model_name
        self.critic_mode = critic_mode

        print(f"✓ Modular RAG Service initializing...")
        print(f"  LLM Model: {self.model_name}")
        print(f"  Critic mode: {self.critic_mode}")

        self.pipeline = build_tuk_pipeline(
            model_name=model_name,
            critic_mode=critic_mode,
        )

        # Keep a handle on the store for stats reporting
        self.vector_store = self.pipeline.agents[1].store  # RetrieverAgent.store
        stats = self.vector_store.get_stats()
        print(f"  FAISS Vector DB: {stats['total_chunks']} chunks, {stats['dimension']}-dim vectors")

    # ------------------------------------------------------------------ #
    # Conversation history helpers (unchanged)
    # ------------------------------------------------------------------ #

    def get_conversation_history(self, session_id: str, limit: int = 5) -> str:
        if session_id not in conversation_memory:
            return "No previous conversation."

        history = conversation_memory[session_id][-limit:]
        history_text = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text.append(f"{role}: {msg['content']}")
        return "\n".join(history_text)

    def add_to_history(self, session_id: str, role: str, content: str):
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []

        conversation_memory[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-20:]

    # ------------------------------------------------------------------ #
    # DB logging helper (unchanged)
    # ------------------------------------------------------------------ #

    def _log_query_safely(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list,
        chunks_found: int,
        response_time: float,
    ):
        try:
            database.log_query(
                session_id=session_id,
                question=question,
                answer=answer,
                sources=sources,
                chunks_found=chunks_found,
                response_time=response_time,
            )
        except Exception as e:
            print(f"  ⚠ Query logging failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Shared context builder
    # ------------------------------------------------------------------ #

    def _make_context(self, question: str, session_id: str, k: int) -> RAGContext:
        ctx = RAGContext(question=question, session_id=session_id, top_k=k)
        ctx.metadata["history"] = self.get_conversation_history(session_id, limit=5)
        return ctx

    def _sources_to_dicts(self, sources) -> list:
        """Convert typed Source objects to the dict shape the API expects."""
        return [
            {
                "source": s.source,
                "quote": s.quote,
                "relevance_score": s.relevance_score,
            }
            for s in sources
        ]

    # ------------------------------------------------------------------ #
    # Standard (non-streaming) answer
    # ------------------------------------------------------------------ #

    def answer_question(
        self, question: str, session_id: str = "default", k: int = 5
    ) -> Dict[str, Any]:
        print(f"\n🤔 Question: {question}")
        print(f"  Session: {session_id}")

        # 1. Cache check (skip pipeline entirely on hit)
        cache_key = f"{session_id}_{question}_{self.model_name}"
        if cache_key in cache:
            cache_time, cache_result = cache[cache_key]
            if time.time() - cache_time < cache_ttl:
                print(f"  ⚡ Returning cached answer")
                return cache_result

        # 2. Run the modular pipeline
        ctx = self._make_context(question, session_id, k)
        start_time = time.time()
        ctx = self.pipeline.run(ctx)
        elapsed = time.time() - start_time

        # 3. Build the API response
        result = {
            "question": question,
            "answer": ctx.answer,
            "sources": self._sources_to_dicts(ctx.sources),
            "chunks_found": len(ctx.retrieved_chunks),
            "response_time": float(elapsed),
            "vector_db": "FAISS",
            "grounded": ctx.grounded,
            "route": ctx.route,
        }

        # 4. Cache + history + log (only for non-trivial routes)
        if ctx.route not in ("small_talk", "off_topic"):
            cache[cache_key] = (time.time(), result)
            self.add_to_history(session_id, "user", question)
            self.add_to_history(session_id, "assistant", ctx.answer)

        self._log_query_safely(
            session_id=session_id,
            question=question,
            answer=ctx.answer,
            sources=result["sources"],
            chunks_found=len(ctx.retrieved_chunks),
            response_time=float(elapsed),
        )

        return result

    # ------------------------------------------------------------------ #
    # Streaming answer
    # ------------------------------------------------------------------ #

    def stream_answer(
        self, question: str, session_id: str = "default", k: int = 5
    ) -> Generator[Dict[str, Any], None, None]:
        print(f"\n🤔 [stream] Question: {question}")

        ctx = self._make_context(question, session_id, k)
        start_time = time.time()
        full_answer_parts = []
        sources_payload = []
        chunks_found = 0
        final_ctx_dict = None

        for event in self.pipeline.stream(ctx):
            etype = event.get("type")

            if etype == "sources":
                sources_payload = event.get("sources", [])
                chunks_found = event.get("chunks_found", 0)
                yield {
                    "type": "sources",
                    "sources": sources_payload,
                    "chunks_found": chunks_found,
                }

            elif etype == "token":
                content = event.get("content", "")
                full_answer_parts.append(content)
                yield {"type": "token", "content": content}

            elif etype == "retry":
                yield {
                    "type": "retry",
                    "attempt": event.get("attempt", 0),
                    "reason": event.get("reason", ""),
                }

            elif etype == "error":
                yield {"type": "error", "message": event.get("message", "")}

            elif etype == "done":
                final_ctx_dict = event.get("context", {})

        elapsed = time.time() - start_time
        full_answer = "".join(full_answer_parts)

        # Post-stream bookkeeping
        route = (final_ctx_dict or {}).get("route", "internal_docs")
        result = {
            "question": question,
            "answer": full_answer,
            "sources": sources_payload,
            "chunks_found": chunks_found,
            "response_time": float(elapsed),
            "vector_db": "FAISS",
            "grounded": (final_ctx_dict or {}).get("grounded", False),
            "route": route,
        }

        if route not in ("small_talk", "off_topic"):
            cache_key = f"{session_id}_{question}_{self.model_name}"
            cache[cache_key] = (time.time(), result)
            self.add_to_history(session_id, "user", question)
            self.add_to_history(session_id, "assistant", full_answer)

        self._log_query_safely(
            session_id=session_id,
            question=question,
            answer=full_answer,
            sources=sources_payload,
            chunks_found=chunks_found,
            response_time=float(elapsed),
        )

        yield {"type": "done", "response_time": float(elapsed)}

    # ------------------------------------------------------------------ #
    # Stats + admin helpers (unchanged)
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_db": self.vector_store.get_stats(),
            "model": self.model_name,
            "critic_mode": self.critic_mode,
            "cache_size": len(cache),
            "active_sessions": len(conversation_memory),
        }

    def clear_cache(self):
        global cache
        cache = {}
        print("✓ Cache cleared")

    def clear_history(self, session_id: str = None):
        global conversation_memory
        if session_id:
            conversation_memory.pop(session_id, None)
            print(f"✓ Cleared history for session: {session_id}")
        else:
            conversation_memory = {}
            print("✓ Cleared all conversation history")