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
            
        }