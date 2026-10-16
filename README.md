# TUK-ConvoSearch

**Build domain-specific RAG pipelines from composable agents.**
Ships with two working examples: a cancer research literature assistant and a university Q&A system.

[![Tests](https://github.com/lemayian23/tuk-convosearch/actions/workflows/tests.yml/badge.svg)](https://github.com/lemayian23/tuk-convosearch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## Why this exists

Most RAG projects are one-off scripts: a retriever, an LLM call, and a prompt. They work for a demo. They break the moment you need to:

- Add a second domain (cancer research vs. student Q&A)
- Change the retrieval backend (FAISS → PubMed → pgvector)
- Catch and correct hallucinations
- Measure whether a change actually improved anything

TUK-ConvoSearch is what you get when you take a working RAG pipeline and ask: **"what would this look like if it had to support a second domain without rewriting everything?"**

The answer is a small framework of composable agents, swappable stores, and domain-specific pipeline configs.

---

## Architecture

Every question flows through a **Pipeline** of **Agents**. Each agent has one job. Each can be swapped, tested, or extended without touching the others.


Question
│
▼
┌─────────────────────────────────────────────────────────────┐
│ RouterAgent │
│ Classifies: in-domain / off-topic / small-talk │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ QueryExpansionAgent (optional, per-pipeline) │
│ Rewrites user question into domain-optimized search query │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ RetrieverAgent │
│ Searches any BaseVectorStore for the top-k chunks │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ SynthesizerAgent │
│ Generates a cited answer constrained to retrieved context │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ CriticAgent │
│ Validates grounding. Requests retry if ungrounded. │
└─────────────────────────────────────────────────────────────┘
│
▼
Answer + Sources

text

**Design principles:**

- **Single responsibility** — each agent does one thing
- **Swappable** — replace any agent or store without touching the pipeline
- **Testable** — 26 unit tests, all isolated (no LLM calls, no I/O)
- **Measurable** — an eval harness with a fixed question set and per-question scoring

---

## Two working examples

The framework ships with two domain pipelines. Same agents. Same pipeline class. Different config.

### 1. Cancer research literature assistant

- **Source:** PubMed abstracts, fetched live via NCBI E-utilities
- **Embeddings:** `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical)
- **Query expansion:** LLM rewrites clinical questions into MeSH-style queries
- **Critic:** LLM-based faithfulness check with citation enforcement
- **Refusal handling:** Deterministically detects and accepts "no literature" answers

**Eval results** (10 fixed questions):

| Metric | Score |
|--------|-------|
| Route accuracy | 10/10 (100%) |
| Retrieval hits | 9/10 (90%) |
| Grounded answers | 9/10 (90%) |
| Citation present | 9/10 (90%) |
| Average latency | 37s per question |

The one failing question (BRCA1) is a known corpus gap, not a pipeline failure. The system correctly refuses to answer it.

### 2. University Q&A assistant (TU-K)

- **Source:** Official TU-K documents (PDFs, DOCX, TXT)
- **Embeddings:** `all-MiniLM-L6-v2` (general purpose)
- **Query expansion:** None — the corpus uses plain English
- **Critic:** Heuristic token overlap (fast, no LLM call)
- **Easter eggs:** Handles "tell me a joke" without invoking the pipeline

Both pipelines use the same four core agents. The differences are entirely in configuration.

---

## Quickstart

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed.

```bash
git clone https://github.com/lemayian23/tuk-convosearch.git
cd tuk-convosearch

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Pull the models
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# Run the tests
python -m pytest
Try the cancer pipeline
python
from app.rag.configs.cancer_pipeline import build_cancer_pipeline
from app.rag import RAGContext

pipeline = build_cancer_pipeline()
ctx = RAGContext(
    question="What is the mechanism of trastuzumab in HER2-positive breast cancer?",
    session_id="demo",
)
ctx = pipeline.run(ctx)

print(ctx.answer)
for source in ctx.sources:
    print(f"  - {source.source}")
Output:

text
The mechanism of trastuzumab involves binding to the extracellular
domain of the HER2 protein, inhibiting downstream signaling pathways...

  - PMID:39821040 — HER2-Positive Breast Cancer Treatment and Resistance
  - PMID:32721042 — Novel HER2-targeted therapies for HER2-positive...
  - PMID:32896428 — Systemic therapy for metastatic HER2-positive...
Try the TU-K pipeline
python
from app.rag.configs.tuk_pipeline import build_tuk_pipeline
from app.rag import RAGContext

pipeline = build_tuk_pipeline()
ctx = RAGContext(question="When do the second semester exams start?", session_id="demo")
ctx = pipeline.run(ctx)
print(ctx.answer)
Project structure
text
tuk-convosearch/
├── app/
│   ├── api/                              # FastAPI endpoints
│   │   └── chat_proposal.py              # Chat + SSE streaming
│   ├── services/
│   │   ├── rag_service_proposal.py       # Service layer (delegates to pipeline)
│   │   └── database.py                   # SQLite query logging
│   └── rag/                              # ← the framework
│       ├── core/
│       │   ├── base_agent.py             # Agent interface
│       │   ├── context.py                # RAGContext (shared state)
│       │   └── pipeline.py               # Orchestrator (retries, short-circuit)
│       ├── agents/
│       │   ├── router_agent.py
│       │   ├── query_expansion_agent.py
│       │   ├── retriever_agent.py
│       │   ├── synthesizer_agent.py
│       │   └── critic_agent.py
│       ├── stores/
│       │   ├── base_store.py             # VectorStore interface
│       │   ├── faiss_store.py            # Local FAISS implementation
│       │   └── pubmed_store.py           # NCBI E-utilities + FAISS
│       └── configs/
│           ├── tuk_pipeline.py           # University Q&A config
│           └── cancer_pipeline.py        # Cancer research config
│
├── evals/                                # Evaluation harness
│   ├── questions.py                      # Fixed question set
│   └── runner.py                         # Scoring + reporting
│
├── tests/                                # Unit tests (pytest)
├── docs/                                 # TU-K knowledge base
├── pubmed_index/                         # Cancer corpus (FAISS)
├── faiss_index/                          # TU-K corpus (FAISS)
└── pytest.ini
Extending the framework
The whole point of this project is that adding a new domain is one config file, not a rewrite.

Example: adding a legal research pipeline
python
# app/rag/configs/legal_pipeline.py

from app.rag.core.pipeline import Pipeline
from app.rag.agents import RouterAgent, RetrieverAgent, SynthesizerAgent, CriticAgent
from app.rag.stores.faiss_store import FaissStore

LEGAL_KEYWORDS = ["contract", "statute", "precedent", "plaintiff", "jurisdiction", ...]

LEGAL_PROMPT = """You are a legal research assistant...

RULES:
1. Cite the case or statute after every claim
2. If the context does not contain the answer, say: "..."
"""

def build_legal_pipeline(store=None):
    if store is None:
        store = FaissStore(persist_directory="./legal_index")

    return Pipeline([
        RouterAgent(domain_keywords=LEGAL_KEYWORDS, ...),
        RetrieverAgent(store=store),
        SynthesizerAgent(prompt_template=LEGAL_PROMPT, ...),
        CriticAgent(mode="llm", ...),
    ])
Three things need to change:

Domain keywords for the router

System prompt for the synthesizer

Vector store for the corpus

Everything else — the pipeline, the retries, the critic logic, the short-circuit behavior — stays identical.

Available extension points
Component	What you can swap	Where to plug in
Vector store	FAISS, PubMed, Chroma, pgvector	Subclass BaseVectorStore
Router logic	Keyword, LLM, hybrid	Subclass Agent
Query expansion	LLM, MeSH dictionary, none	Add a QueryExpansionAgent or skip
Synthesizer	Ollama, OpenAI, Anthropic	Pass a different model_name or agent
Critic	Heuristic, LLM, hybrid	Swap mode or subclass ValidationAgent
Testing
Unit tests (fast, no I/O):

bash
python -m pytest
Evaluation harness (slow, runs full pipelines):

bash
python -m evals.runner > evals/baseline.txt
The harness measures per-question:

Route accuracy — did the router classify correctly?

Retrieval hit — did the top-k chunks contain the expected keywords?

Grounded — did the critic accept the answer?

Citation present — did the synthesizer cite sources?

Latency — how long did the full pipeline take?

Numbers are committed to evals/ after each significant change, so you can diff improvements.

Roadmap
What's done, what's next, what's not planned.

✅ Done
Modular agent pipeline with retries and short-circuit routing

Two working domain pipelines (cancer research, TU-K)

26 unit tests, all isolated from LLMs and I/O

Evaluation harness with real numbers

Live PubMed ingestion via NCBI E-utilities

Citation enforcement in the synthesizer

Deterministic refusal handling in the critic

🚧 Next
GitHub Actions CI on all supported Python versions

Published PyPI package (pip install tuk-convosearch)

Docker image for one-command deployment

Documentation site (MkDocs or Docusaurus)

Additional store backends (Chroma, pgvector)

Streaming support for the critic stage

❌ Not planned
A hosted SaaS version

Clinical decision support (this is a research tool, not a medical device)

Multi-tenant authentication

Fine-tuning custom models

If you want any of the "not planned" items, fork the project — the architecture supports all of them.

Contributing
Contributions are welcome. The best place to start:

good first issue labels on GitHub

Add a new pipeline config — pick a domain you know, follow the cancer pipeline as a template

Write an eval question set for an existing domain

Improve the docs — especially the extension guide

Before opening a PR:

bash
python -m pytest              # all tests must pass
python -m evals.runner        # baseline must not regress
If you're adding a new agent or store, include tests. If you're adding a pipeline config, include a couple of eval questions.

For larger changes, open an issue first to discuss the approach.

Citation
If you use this framework in academic work:

bibtex
@software{tuk_convosearch_2026,
  author = {Kirionki, Denis Lemayian},
  title = {TUK-ConvoSearch: A Modular RAG Framework for High-Stakes Domains},
  year = {2026},
  url = {https://github.com/lemayian23/tuk-convosearch}
}
License
MIT — see LICENSE.

Author
Denis Lemayian Kirionki
School of Computing and Information Technology
Technical University of Kenya

Built with support from the TU-K School of Computing and Information Technology. Supervised by Mr. Shadrack Ngumbau and Mr. Dalmas Owira.

Open an issue for questions, bugs, or feature requests.
