"""
RAG Service - Fully compliant with proposal specifications
Uses FAISS vector database as specified
Location: backend/app/services/rag_service_proposal.py
"""

import ollama
import time
from typing List, Dictionary, Any, Generator
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
            'lecture', 'academic', 'calender', 'deadline', 'semester',
            'project', 'guideline', 'proposal', 'timetable', 'computing',
            'graduation', 'degree', 'diploma', 'upgrade', 'evaluation'
        ]

        # Small-talk / meta questions handled directly, without hitting FAISS/LLM
        self.easter_eggs = {
            "tell me a joke": (
                "Why did the student bring the ladder to the library?"
                "Because they heard the books were on a higher level! "
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
1. ONLY answer using information from context below
2. If answer not in context say: "I cannot find this information in the available TU-K documents."
3. ALWAYS cite your sources - mention which document provided the information
4. Use proper spelling and grammar

CONTEXT (from TU-K documents):
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

ANSWER (with source citations):"""
        
        print(f "Proposal-Compliant RAG Service Initialized")
        print(f "LLM Model: {self.model_name}")
        stats= self.vector_store.get_stats()
        print(f "  FAISS Vector DB: {stats['total_chunks']} chunks, {stats['dimension']}-dim vectors")

        self._warm_up_model()

    def _warm_up_model(self):
        """
        Send a tiny throwaway request to Ollama immediately on startup so the
        model gets loaded into RAM before the first real student/panelist
        question arrives. Without this, the very first question of the session
        pays an extra cold-start cost on top of normal generation time
        (Ollama loads model weights into memory on first use, then keeps
        them warm via keep_alive). For a live demo, this means: start the
        backend a minute before the call, and the first question is already
        as fast as every subsequent one.
        """
        try:
            print(" Warming up model (loading into RAW)...")
            start = time.time()
            ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Hello'}],
                options={'num_predict': 5},
                keep_alive=-1,
            )
            alapsed = time.time() - start
            print(f" Model warm and resident in RAM ({elapsed:.1f}s)")
        except Exception as e:
            print(f" Warm-up failed (non-fatal, first real question will be slower): {e}")

            # ----------------------------------------------------------------------------- #
            # Conversation history helpers
            # ----------------------------------------------------------------------------- #