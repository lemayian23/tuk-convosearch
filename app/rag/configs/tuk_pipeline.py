"""
TU-K pipeline configuration.

Assembles Router → Retriever → Synthesizer → Critic into a Pipeline
using TU-K-specific keywords, prompts, and thresholds.

Location: app/rag/configs/tuk_pipeline.py
"""

from app.rag.core.pipeline import Pipeline
from app.rag.agents.router_agent import RouterAgent
from app.rag.agents.retriever_agent import RetrieverAgent
from app.rag.agents.synthesizer_agent import SynthesizerAgent
from app.rag.agents.critic_agent import CriticAgent


# ---------------------------------------------------------------------- #
# TU-K domain configuration
# ---------------------------------------------------------------------- #

TUK_KEYWORDS = [
    "tuk", "technical university", "university", "kenya", "exam",
    "registration", "fee", "campus", "library", "student", "course",
    "department", "lecture", "academic", "calendar", "date", "deadline",
    "semester", "session", "unit", "project", "guideline", "proposal",
    "timetable", "paper", "assessment", "cat", "marks", "result",
    "grade", "transcript", "graduation", "convocation", "degree",
    "diploma", "upgrade", "admission", "application", "intake",
    "attachment", "industrial learning", "logbook", "requirement",
    "address", "postal", "contact", "hostel", "clearance", "discipline",
    "computing", "information technology", "education", "evaluation",
]

TUK_GREETING = (
    "Hello! I am TUK-ConvoSearch. Ask me about exam dates, the academic "
    "calendar, the exam timetable, admissions, or anything else written in "
    "the TU-K documents I have been given."
)

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
2. Say "I cannot find this information in the available TU-K documents." ONLY when the context does not answer the question at all. If you have given an answer from the context, never add that sentence to it.
3. ALWAYS cite your sources - mention which document provided the information
4. Use proper spelling and grammar
5. Answer once. Never repeat or restate the same dates.

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
    store=None,
    critic_mode: str = "heuristic",
    model_name: str = "llama3.2:1b",
    critic_model: str = None,
    critic_overlap_threshold: float = 0.4,
    max_sources: int = 3,
    num_predict: int = 300,
    temperature: float = 0.2,
    num_ctx: int = 2048,
    repeat_penalty: float = 1.1,
    keep_alive=-1,
) -> Pipeline:
    """
    Build the TU-K RAG pipeline.

    Everything here is a plain value so the server can pass its settings
    straight through: the same config runs llama3.2:1b on a small VPS and a
    larger model on a big one without touching this file.

    Args:
        store:                    BaseVectorStore. Defaults to the shared
                                  CorpusIndex singleton — the same store the
                                  API serves from — so a hand-built pipeline
                                  measures what students actually hit.
        critic_mode:              "heuristic" (fast) or "llm" (accurate, slower)
        model_name:               Ollama model for synthesis
        critic_model:             Ollama model for the llm critic (defaults to model_name)
        critic_overlap_threshold: token-overlap bar for the heuristic critic
        max_sources:              sources shown to the student
        num_predict/num_ctx/temperature/repeat_penalty/keep_alive: generation options

    Returns:
        A fully assembled Pipeline ready to run or stream.
    """
    if store is None:
        from app.ingestion.index import get_index

        store = get_index()

    agents = [
        RouterAgent(
            domain_keywords=TUK_KEYWORDS,
            easter_eggs=TUK_EASTER_EGGS,
            off_topic_message=TUK_OFF_TOPIC_MESSAGE,
            small_talk_message=TUK_GREETING,
        ),
        RetrieverAgent(
            store=store,
            name="Retriever",
            max_sources=max_sources,
        ),
        SynthesizerAgent(
            model_name=model_name,
            prompt_template=TUK_SYSTEM_PROMPT,
            num_predict=num_predict,
            temperature=temperature,
            num_ctx=num_ctx,
            repeat_penalty=repeat_penalty,
            keep_alive=keep_alive,
        ),
        CriticAgent(
            mode=critic_mode,
            model_name=critic_model or model_name,
            overlap_threshold=critic_overlap_threshold,
        ),
    ]

    return Pipeline(agents)