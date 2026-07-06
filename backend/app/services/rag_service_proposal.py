"""
RAG Service - Fully compliant with proposal specifications
Uses FAISS vector database as specified
Location: backend/app/services/rag_service_proposal.py
"""

import ollama
import time
from typing import List, Dict, Any, Generator
from app.services.faiss_vector_store import FAISSVectorStore
from app.services import database

# Cache for repeated questions
cache = {}
cache_ttl = 3600

# Conversation memory
conversation_memory = {}


class RAGServiceProposal:
    """
    RAG Service implementing all proposal specifications:
    - FAISS vector database
    - Transformer-based embeddings
    - LLM response generation with source citations
    """

    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        self.vector_store = FAISSVectorStore()

        # TU-K keywords used to filter off-topic questions
        self.tuk_keywords = [
            'tuk', 'technical university', 'kenya', 'exam', 'registration',
            'fee', 'campus', 'library', 'student', 'course', 'department',
            'lecture', 'academic', 'calendar', 'deadline', 'semester',
            'project', 'guideline', 'proposal', 'timetable', 'computing', 'information technology',
            'graduation', 'degree', 'diploma', 'upgrade', 'evaluation'
        ]

        # Small-talk / meta questions handled directly, without hitting FAISS/LLM
        self.easter_eggs = {
            "tell me a joke": (
                "Why did the student bring a ladder to the library? "
                "Because they heard the books were on a higher level! 😄"
            ),
            "what can you do": (
                "I can help you with:\n"
                "• Find exam dates and schedules\n"
                "• Answer questions about project guidelines\n"
                "• Locate campus facilities\n"
                "• Explain registration procedures\n"
                "• Provide fee information\n"
                "• And much more about TU-K!"
            ),
            "your capabilities": (
                "I can help you with:\n"
                "• Find exam dates and schedules\n"
                "• Answer questions about project guidelines\n"
                "• Locate campus facilities\n"
                "• Explain registration procedures\n"
                "• Provide fee information\n"
                "• And much more about TU-K!"
            ),
        }

        # System prompt
        self.system_prompt = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

CRITICAL RULES:
1. ONLY answer using information from the context below
2. If answer not in context say: "I cannot find this information in the available TU-K documents."
3. ALWAYS cite your sources - mention which document provided the information
4. Use proper spelling and grammar

CONTEXT (from TU-K documents):
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

ANSWER (with source citations):"""

        print(f"✓ Proposal-Compliant RAG Service Initialized")
        print(f"  LLM Model: {self.model_name}")
        stats = self.vector_store.get_stats()
        print(f"  FAISS Vector DB: {stats['total_chunks']} chunks, {stats['dimension']}-dim vectors")

        self._warm_up_model()

    def _warm_up_model(self):
        """
        Send a tiny throwaway request to Ollama immediately on startup so the
        model gets loaded into RAM before the first real student/panelist
        question arrives. Without this, the very first question of a session
        pays an extra cold-start cost on top of normal generation time
        (Ollama loads model weights into memory on first use, then keeps
        them warm via keep_alive). For a live demo, this means: start the
        backend a minute before the call, and the first question is already
        as fast as every subsequent one.
        """
        try:
            print("  🔥 Warming up model (loading into RAM)...")
            start = time.time()
            ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Hello'}],
                options={'num_predict': 5},
                keep_alive=-1,
            )
            elapsed = time.time() - start
            print(f"  ✓ Model warm and resident in RAM ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ⚠ Warm-up failed (non-fatal, first real question will be slower): {e}")

    # ------------------------------------------------------------------ #
    # Conversation history helpers
    # ------------------------------------------------------------------ #

    def get_conversation_history(self, session_id: str, limit: int = 5) -> str:
        if session_id not in conversation_memory:
            return "No previous conversation."

        history = conversation_memory[session_id][-limit:]
        history_text = []
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text.append(f"{role}: {msg['content']}")

        return "\n".join(history_text)

    def add_to_history(self, session_id: str, role: str, content: str):
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []

        conversation_memory[session_id].append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })

        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-20:]

    # ------------------------------------------------------------------ #
    # Internal shared helpers (used by both answer_question and stream_answer)
    # ------------------------------------------------------------------ #

    def _check_easter_egg(self, question_lower: str) -> str | None:
        """Return a canned response for small-talk / meta questions, or None."""
        for trigger, response in self.easter_eggs.items():
            if trigger in question_lower:
                return response
        return None

    def _is_tuk_related(self, question_lower: str) -> bool:
        return any(kw in question_lower for kw in self.tuk_keywords)

    def _build_context_and_sources(self, relevant_chunks: List[Dict[str, Any]]):
        """Build the LLM context string and the source citation list from
        the top retrieved chunks. Shared by answer_question and stream_answer
        so both code paths produce identical citations."""
        context_parts = []
        sources = []

        for chunk in relevant_chunks[:3]:
            source = chunk['metadata'].get('source', 'unknown')
            chunk_text = chunk['text']

            quote = chunk_text[:250].strip()
            if len(chunk_text) > 250:
                quote = quote + "..."

            context_parts.append(f"[Source: {source}]\n{chunk_text[:600]}")
            sources.append({
                'source': str(source),
                'quote': str(quote),
                'relevance_score': float(chunk.get('relevance_score', 0))
            })

        context = "\n\n".join(context_parts)
        return context, sources

    def _log_query_safely(self, session_id: str, question: str, answer: str,
                           sources: list, chunks_found: int, response_time: float):
        """
        Persist the query to SQLite for the admin panel's query-log view.
        Wrapped in try/except so that a database problem (locked file,
        missing table, disk issue) never breaks a chat response — logging
        is a nice-to-have for the admin panel, not a dependency of the
        core chat feature.
        """
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
    # Standard (non-streaming) answer
    # ------------------------------------------------------------------ #

    def answer_question(self, question: str, session_id: str = "default", k: int = 5) -> Dict[str, Any]:
        print(f"\n🤔 Question: {question}")
        print(f"  Session: {session_id}")

        question_lower = question.lower()

        # 1. Easter eggs / small talk first
        easter_egg_reply = self._check_easter_egg(question_lower)
        if easter_egg_reply:
            self._log_query_safely(session_id, question, easter_egg_reply, [], 0, 0.0)
            return {
                "answer": easter_egg_reply,
                "sources": [],
                "chunks_found": 0,
                "response_time": 0.0,
                "vector_db": "FAISS"
            }

        # 2. Off-topic filter
        if not self._is_tuk_related(question_lower) and len(question_lower) > 5:
            off_topic_msg = "I'm TUK-ConvoSearch. I can only answer questions about TU-K related topics."
            self._log_query_safely(session_id, question, off_topic_msg, [], 0, 0.0)
            return {
                "answer": off_topic_msg,
                "sources": [],
                "chunks_found": 0,
                "response_time": 0.0,
                "vector_db": "FAISS"
            }

        # 3. Cache check
        cache_key = f"{session_id}_{question}_{self.model_name}"
        if cache_key in cache:
            cache_time, cache_result = cache[cache_key]
            if time.time() - cache_time < cache_ttl:
                print(f"  ⚡ Returning cached answer")
                return cache_result

        history = self.get_conversation_history(session_id, limit=5)

        print(f"  🔍 Searching FAISS...")
        relevant_chunks = self.vector_store.search(question, k=k)

        if not relevant_chunks:
            no_result_msg = "I cannot find this information in the available TU-K documents."
            self._log_query_safely(session_id, question, no_result_msg, [], 0, 0.0)
            return {
                "answer": no_result_msg,
                "sources": [],
                "chunks_found": 0,
                "response_time": 0.0,
                "vector_db": "FAISS"
            }

        context, sources = self._build_context_and_sources(relevant_chunks)

        print(f"  💭 Generating answer with {self.model_name}...")
        start_time = time.time()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': self.system_prompt.format(
                        context=context,
                        history=history,
                        question=question
                    )}
                ],
                options={
                    'num_predict': 300,    # cap response length - prevents rambling, keeps latency bounded
                    'temperature': 0.2,    # low temperature - factual, consistent answers
                    'num_ctx': 2048,       # context window sized to actual prompt needs, reduces memory overhead
                },
                keep_alive=-1,             # keep model resident in RAM between requests - eliminates reload delay
            )
            answer = response['message']['content']
            elapsed = time.time() - start_time
            print(f"  ✓ Answer generated in {elapsed:.1f} seconds")

        except Exception as e:
            answer = f"Error generating answer: {e}"
            elapsed = 0.0

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_found": len(relevant_chunks),
            "response_time": float(elapsed),
            "vector_db": "FAISS",
            "total_chunks": self.vector_store.get_stats()['total_chunks']
        }

        cache[cache_key] = (time.time(), result)

        self.add_to_history(session_id, "user", question)
        self.add_to_history(session_id, "assistant", answer)

        self._log_query_safely(session_id, question, answer, sources, len(relevant_chunks), float(elapsed))

        return result

    # ------------------------------------------------------------------ #
    # Streaming answer (word-by-word via Ollama's native streaming)
    # ------------------------------------------------------------------ #

    def stream_answer(self, question: str, session_id: str = "default", k: int = 5) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields small dicts as the answer is produced:
          {'type': 'sources', 'sources': [...], 'chunks_found': N}   -- sent once, first
          {'type': 'token', 'content': '...'}                        -- sent repeatedly
          {'type': 'done', 'response_time': float}                   -- sent once, last

        The chat_proposal.py endpoint turns these into Server-Sent Events.
        Caching and conversation history are updated exactly as in
        answer_question, so both code paths stay consistent.
        """
        print(f"\n🤔 [stream] Question: {question}")
        question_lower = question.lower()

        # 1. Easter eggs / small talk first
        easter_egg_reply = self._check_easter_egg(question_lower)
        if easter_egg_reply:
            self._log_query_safely(session_id, question, easter_egg_reply, [], 0, 0.0)
            yield {'type': 'sources', 'sources': [], 'chunks_found': 0}
            yield {'type': 'token', 'content': easter_egg_reply}
            yield {'type': 'done', 'response_time': 0.0}
            return

        # 2. Off-topic filter
        if not self._is_tuk_related(question_lower) and len(question_lower) > 5:
            msg = "I'm TUK-ConvoSearch. I can only answer questions about TU-K related topics."
            self._log_query_safely(session_id, question, msg, [], 0, 0.0)
            yield {'type': 'sources', 'sources': [], 'chunks_found': 0}
            yield {'type': 'token', 'content': msg}
            yield {'type': 'done', 'response_time': 0.0}
            return

        history = self.get_conversation_history(session_id, limit=5)

        print(f"  🔍 [stream] Searching FAISS...")
        relevant_chunks = self.vector_store.search(question, k=k)

        if not relevant_chunks:
            msg = "I cannot find this information in the available TU-K documents."
            self._log_query_safely(session_id, question, msg, [], 0, 0.0)
            yield {'type': 'sources', 'sources': [], 'chunks_found': 0}
            yield {'type': 'token', 'content': msg}
            yield {'type': 'done', 'response_time': 0.0}
            return

        context, sources = self._build_context_and_sources(relevant_chunks)

        # Tell the client what the sources are before streaming any tokens,
        # so the UI can render the "Sources" panel immediately.
        yield {'type': 'sources', 'sources': sources, 'chunks_found': len(relevant_chunks)}

        print(f"  💭 [stream] Generating answer with {self.model_name}...")
        start_time = time.time()
        full_answer_parts = []

        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': self.system_prompt.format(
                        context=context,
                        history=history,
                        question=question
                    )}
                ],
                options={
                    'num_predict': 300,    # cap response length - prevents rambling, keeps latency bounded
                    'temperature': 0.2,    # low temperature - factual, consistent answers
                    'num_ctx': 2048,       # context window sized to actual prompt needs, reduces memory overhead
                },
                keep_alive=-1,             # keep model resident in RAM between requests - eliminates reload delay
                stream=True
            )

            for part in stream:
                token = part.get('message', {}).get('content', '')
                if token:
                    full_answer_parts.append(token)
                    yield {'type': 'token', 'content': token}

        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            full_answer_parts.append(error_msg)
            yield {'type': 'token', 'content': error_msg}

        elapsed = time.time() - start_time
        full_answer = "".join(full_answer_parts)
        print(f"  ✓ [stream] Answer generated in {elapsed:.1f} seconds")

        # Keep cache + conversation history consistent with answer_question
        cache_key = f"{session_id}_{question}_{self.model_name}"
        result = {
            "question": question,
            "answer": full_answer,
            "sources": sources,
            "chunks_found": len(relevant_chunks),
            "response_time": float(elapsed),
            "vector_db": "FAISS",
            "total_chunks": self.vector_store.get_stats()['total_chunks']
        }
        cache[cache_key] = (time.time(), result)

        self.add_to_history(session_id, "user", question)
        self.add_to_history(session_id, "assistant", full_answer)

        self._log_query_safely(session_id, question, full_answer, sources, len(relevant_chunks), float(elapsed))

        yield {'type': 'done', 'response_time': float(elapsed)}

    # ------------------------------------------------------------------ #
    # Statisticss OR admin helpers
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            'vector_db': self.vector_store.get_stats(),
            'model': self.model_name,
            'cache_size': len(cache),
            'active_sessions': len(conversation_memory)
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