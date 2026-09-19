"""
Cancer research pipeline configuration.

Assembles Router → Retriever(PubMed) → Synthesizer → Critic(llm)
using oncology-specific keywords, a medical prompt, and a larger
Critic model for higher grounding accuracy.

This file exists to prove the modular RAG framework is domain-agnostic:
same agents, same Pipeline, different configuration.

Location: app/rag/configs/cancer_pipeline.py
"""

from app.rag.core.pipeline import Pipeline
from app.rag.agents.router_agent import RouterAgent
from app.rag.agents.retriever_agent import RetrieverAgent
from app.rag.agents.synthesizer_agent import SynthesizerAgent
from app.rag.agents.critic_agent import CriticAgent
from app.rag.stores.pubmed_store import PubMedStore


# ---------------------------------------------------------------------- #
# Cancer research domain configuration
# ---------------------------------------------------------------------- #

CANCER_KEYWORDS = [
    # General oncology
    "cancer", "tumor", "tumour", "neoplasm", "malignancy", "oncology",
    "carcinoma", "sarcoma", "lymphoma", "leukemia", "melanoma", "glioma",
    # Diagnosis
    "biopsy", "histology", "staging", "metastasis", "metastatic",
    "imaging", "pet-ct", "mri", "ct scan", "biomarker", "mutation",
    "genomic", "sequencing", "pathology", "cytology",
    # Treatment
    "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy",
    "checkpoint inhibitor", "surgery", "resection", "adjuvant",
    "neoadjuvant", "palliative", "regimen", "dose",
    # Drugs / classes
    "cisplatin", "doxorubicin", "paclitaxel", "pembrolizumab",
    "nivolumab", "trastuzumab", "bevacizumab", "tamoxifen",
    "egfr", "her2", "brca", "kras", "pdl1", "pd-l1",
    # Outcomes
    "prognosis", "survival", "remission", "recurrence", "toxicity",
    "adverse event", "clinical trial", "efficacy",
]

# No easter eggs — this is a research tool, not a chatbot
CANCER_EASTER_EGGS = {}

CANCER_OFF_TOPIC_MESSAGE = (
    "This is a clinical research assistant. I can only answer questions "
    "about oncology diagnosis, treatment, and supporting literature."
)

CANCER_SYSTEM_PROMPT = """You are an AI research assistant supporting an oncology team.

You help clinicians and researchers find and interpret evidence from peer-reviewed literature.

CRITICAL RULES:
1. ONLY use information from the context below — never add prior knowledge
2. Cite the source (PMID, DOI, or title) for every clinical claim
3. If the context does not answer the question, say: "The retrieved literature does not address this."
4. Be explicit about uncertainty, sample sizes, and study limitations when present
5. NEVER provide direct patient-care advice — this is a decision-support tool, not a diagnosis

CONTEXT (from retrieved literature):
{context}

CONVERSATION HISTORY:
{history}

CLINICAL QUESTION: {question}

EVIDENCE-BASED ANSWER (with citations):"""


# ---------------------------------------------------------------------- #
# Pipeline factory
# ---------------------------------------------------------------------- #

def build_cancer_pipeline(
    store: PubMedStore = None,
    critic_mode: str = "llm",            # clinical claims need LLM-level grounding
    synthesizer_model: str = "llama3.2:1b",
    critic_model: str = "llama3.2:3b",   # bigger model for stricter verdicts
) -> Pipeline:
    """
    Build the cancer research RAG pipeline.

    Args:
        store:             PubMedStore (creates one if None)
        critic_mode:       "llm" (default) or "heuristic"
        synthesizer_model: Ollama model for answer generation
        critic_model:      Ollama model for grounding validation

    Returns:
        A fully assembled Pipeline ready to run or stream.
    """
    if store is None:
        store = PubMedStore()

    agents = [
        RouterAgent(
            domain_keywords=CANCER_KEYWORDS,
            easter_eggs=CANCER_EASTER_EGGS,
            off_topic_message=CANCER_OFF_TOPIC_MESSAGE,
        ),
        RetrieverAgent(store=store, name="PubMedRetriever"),
        SynthesizerAgent(
            model_name=synthesizer_model,
            prompt_template=CANCER_SYSTEM_PROMPT,
            num_predict=500,             # longer answers for clinical reasoning
            temperature=0.1,             # near-deterministic, evidence-focused
            num_ctx=4096,                # larger window for multi-chunk context
            keep_alive=-1,
        ),
        CriticAgent(
            mode=critic_mode,
            model_name=critic_model,
            overlap_threshold=0.5,       # stricter when falling back to heuristic
        ),
    ]

    return Pipeline(agents)