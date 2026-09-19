"""
TU-K pipeline configuration.

Assembles Router → Retriever → Synthesizer → Critic into a Pipeline
using TU-K-specific keywords, prompts, and thresholds.

Location: backend/app/rag/configs/tuk_pipeline.py
"""

from app.rag.core.pipeline import Pipeline
from app.rag.agents.router_agent import RouterAgent
from app.rag.agents.retriever_agent import RetrieverAgent
from app.rag.agents.synthesizer_agent import SynthesizerAgent
from app.rag.agents.critic_agent import CriticAgent
from app.rag.stores.faiss_store import FaissStore


# ---------------------------------------------------------------------- #
# TU-K domain configuration
# ---------------------------------------------------------------------- #

TUK_KEYWORDS = [
    "tuk", "technical university", "kenya", "exam", "registration",
    "fee", "campus", "library", "student", "course", "department",
    "lecture", "academic", "calendar", "deadline", "semester",
    "project", "guideline", "proposal", "timetable", "computing",
    "information technology", "graduation", "degree", "diploma",
    "upgrade", "evaluation",
]

TUK_EASTER_EGGS = {
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

TUK_OFF_TOPIC_MESSAGE = (
    "I'm TUK-ConvoSearch. I can only answer questions about "
    "TU-K related topics."
)

TUK_SYSTEM_PROMPT = """You are TUK-ConvoSearch, an AI assistant for Technical University of Kenya.

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


# ---------------------------------------------------------------------- #
# Pipeline factory
# ---------------------------------------------------------------------- #

def build_tuk_pipeline(
    store: FaissStore = None,
    critic_mode: str = "heuristic",
    model_name: str = "llama3.2:1b",
) -> Pipeline:
    """
    Build the TU-K RAG pipeline.

    Args:
        store:       pre-built FaissStore (creates one if None)
        critic_mode: "heuristic" (fast) or "llm" (accurate, slower)
        model_name:  Ollama model for synthesis

    Returns:
        A fully assembled Pipeline ready to run or stream.
    """
    if store is None:
        store = FaissStore()

    agents = [
        RouterAgent(
            domain_keywords=TUK_KEYWORDS,
            easter_eggs=TUK_EASTER_EGGS,
            off_topic_message=TUK_OFF_TOPIC_MESSAGE,
        ),
        RetrieverAgent(store=store, name="Retriever"),
        SynthesizerAgent(
            model_name=model_name,
            prompt_template=TUK_SYSTEM_PROMPT,
            num_predict=300,
            temperature=0.2,
            num_ctx=2048,
            keep_alive=-1,
        ),
        CriticAgent(mode=critic_mode, overlap_threshold=0.4),
    ]

    return Pipeline(agents)