"""
Cancer research pipeline configuration.

Location: app/rag/configs/cancer_pipeline.py
"""

from app.rag.core.pipeline import Pipeline
from app.rag.agents.router_agent import RouterAgent
from app.rag.agents.query_expansion_agent import QueryExpansionAgent
from app.rag.agents.retriever_agent import RetrieverAgent
from app.rag.agents.synthesizer_agent import SynthesizerAgent
from app.rag.agents.critic_agent import CriticAgent
from app.rag.stores.pubmed_store import PubMedStore


# ---------------------------------------------------------------------- #
# Cancer research domain configuration
# ---------------------------------------------------------------------- #

CANCER_KEYWORDS = [
    "cancer", "tumor", "tumour", "neoplasm", "malignancy", "oncology",
    "carcinoma", "sarcoma", "lymphoma", "leukemia", "melanoma", "glioma",
    "biopsy", "histology", "staging", "metastasis", "metastatic",
    "imaging", "pet-ct", "mri", "ct scan", "biomarker", "mutation",
    "genomic", "sequencing", "pathology", "cytology",
    "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy",
    "checkpoint inhibitor", "surgery", "resection", "adjuvant",
    "neoadjuvant", "palliative", "regimen", "dose",
    "cisplatin", "doxorubicin", "paclitaxel", "pembrolizumab",
    "nivolumab", "trastuzumab", "bevacizumab", "tamoxifen",
    "egfr", "her2", "brca", "kras", "pdl1", "pd-l1",
    "prognosis", "survival", "remission", "recurrence", "toxicity",
    "adverse event", "clinical trial", "efficacy",
]

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

CANCER_QUERY_EXPANSION_PROMPT = """Rewrite the clinical question below into a short PubMed search query.

Rules:
- Output ONLY the query keywords. No explanation. No quotes. No boolean operators.
- Use standard medical terminology (drug names, target names, disease names).
- Extract only concepts that appear in the question. Do NOT add related terms.
- Keep it under 15 words.

Question: {question}

Query:"""


# ---------------------------------------------------------------------- #
# Pipeline factory
# ---------------------------------------------------------------------- #

def build_cancer_pipeline(
    store: PubMedStore = None,
    critic_mode: str = "llm",
    synthesizer_model: str = "llama3.2:1b",
    critic_model: str = "llama3.2:3b",
    expander_model: str = "llama3.2:3b",
    enable_expansion: bool = True,
) -> Pipeline:
    """
    Build the cancer research RAG pipeline.

    Args:
        store:              PubMedStore (creates one if None)
        critic_mode:        "llm" (default) or "heuristic"
        synthesizer_model:  Ollama model for answer generation
        critic_model:       Ollama model for grounding validation
        expander_model:     Ollama model for query expansion
        enable_expansion:   set False to skip query expansion entirely

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
    ]

    if enable_expansion:
        agents.append(
            QueryExpansionAgent(
                model_name=expander_model,
                prompt_template=CANCER_QUERY_EXPANSION_PROMPT,
            )
        )

    agents.extend([
        RetrieverAgent(store=store, name="PubMedRetriever"),
        SynthesizerAgent(
            model_name=synthesizer_model,
            prompt_template=CANCER_SYSTEM_PROMPT,
            num_predict=200,
            temperature=0.1,
            num_ctx=4096,
            keep_alive=-1,
        ),
        CriticAgent(
            mode=critic_mode,
            model_name=critic_model,
            overlap_threshold=0.5,
        ),
    ])

    return Pipeline(agents)