# TUK-ConvoSearch

**A production RAG assistant for TU-K students, built from composable agents.**

The shipped product is the student assistant: a FastAPI service that answers TUK questions over official university documents, with a React student UI, an admin panel, per-IP rate limiting, sessions persisted in SQLite, and streaming answers. Underneath it is still the framework this project started as — a pipeline of small agents and swappable stores — and the second pipeline (a cancer research literature assistant over PubMed) proves the point: same agents, same pipeline class, one config file.

Both pipelines are served by the same UI and the same `POST /api/chat` endpoint, selected with a `pipeline` field (`tuk` | `cancer`). The cancer one is opt-in on a deployed box: it needs the local `pubmed_index/` snapshot mounted (see `deploy/docker-compose.yml`), and `GET /api/config` only advertises it when those files are actually present, so students never pick an assistant that has no corpus behind it.

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

```text
Question
│
▼
┌─────────────────────────────────────────────────────────────┐
│ RouterAgent                                                 │
│ Classifies: in-domain / off-topic / small-talk              │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ QueryExpansionAgent (optional, per-pipeline)                │
│ Rewrites user question into domain-optimized search query   │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ RetrieverAgent                                              │
│ Searches any BaseVectorStore for the top-k chunks           │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ SynthesizerAgent                                            │
│ Generates a cited answer constrained to retrieved context   │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ CriticAgent                                                 │
│ Validates grounding. Requests retry if ungrounded.          │
└─────────────────────────────────────────────────────────────┘
│
▼
Answer + Sources
```

**Design principles:**

- **Single responsibility** — each agent does one thing
- **Swappable** — replace any agent or store without touching the pipeline
- **Testable** — 124 unit tests, all isolated (no LLM calls, no I/O; Ollama is faked)
- **Measurable** — an eval harness with a fixed question set and per-question scoring

---

## The two pipelines

Two domain pipelines run on the same server and the same endpoint. Same agents. Same pipeline class. Different config.

### 1. TU-K student assistant (the shipped product)

- **Source:** Official TU-K documents (PDFs, DOCX, TXT, MD) in `corpus/tuk/`, extracted and chunked at ingestion
- **Embeddings:** `all-MiniLM-L6-v2` (general purpose), stored as blobs in SQLite and searched with FAISS
- **Query expansion:** None — the corpus uses plain English
- **Critic:** Heuristic token overlap (fast, no LLM call)
- **Easter eggs:** Handles "tell me a joke" without invoking the pipeline
- **Around the pipeline:** session history, document upload, rate limiting, and an admin panel (`app/api/`, `admin/`)

### 2. Cancer research literature assistant

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

Both pipelines are assembled from the same agent classes; the TU-K config simply leaves out the optional query-expansion step. The differences are entirely in configuration.

---

## Quickstart

Two ways to run it. **Local development** (pip + Ollama on your machine) is for hacking on the framework. **Production / VPS** (Docker) is how the student assistant is actually deployed.

### Local development

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
```

Run the server itself (it also serves the built UIs, so `web/dist` and `admin/dist` must exist for `/` and `/admin`):

```bash
uvicorn app.main:app --reload --port 8000
```

For front-end work, `npm --prefix web run dev` starts Vite on 5173 and proxies `/api` to `127.0.0.1:8000`; that origin is the default in `TUK_CORS_ORIGINS`, so nothing else needs configuring.

#### Try the cancer pipeline

```python
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
```

**Output:**

```text
The mechanism of trastuzumab involves binding to the extracellular
domain of the HER2 protein, inhibiting downstream signaling pathways...

  - PMID:39821040 — HER2-Positive Breast Cancer Treatment and Resistance
  - PMID:32721042 — Novel HER2-targeted therapies for HER2-positive...
  - PMID:32896428 — Systemic therapy for metastatic HER2-positive...
```

#### Try the TU-K pipeline

```python
from app.rag.configs.tuk_pipeline import build_tuk_pipeline
from app.rag import RAGContext

pipeline = build_tuk_pipeline()
ctx = RAGContext(question="When do the second semester exams start?", session_id="demo")
ctx = pipeline.run(ctx)
print(ctx.answer)
```

### Production / VPS

The whole stack (Ollama + FastAPI + Caddy) is three containers and one command. The step-by-step guide, including DNS, TLS, backups and the first deploy, is [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) — start at step 1.

```bash
make env     # copies deploy/.env.example to .env with a freshly generated TUK_SECRET_KEY
# edit .env: SITE_ADDRESS, TUK_OLLAMA_MODEL, TUK_BOOTSTRAP_ADMIN_EMAIL / _PASSWORD
make up      # build if needed, start, and wait until /api/health is ready
```

Then:

- Student UI: `https://your-host/`
- Admin panel: `https://your-host/admin`
- API docs: `https://your-host/docs`
- Liveness: `GET /health` (is the process up?) vs readiness: `GET /api/health` (is the corpus indexed, does the in-memory index match the stored chunks, can Ollama serve the configured model?)

The other operator shortcuts are in the root `Makefile` (`make help` lists them): `make status`, `make logs`, `make restart`, `make backup`, `make restore`, `make reindex`, `make admin`, `make sqlite`, `make bench`.

### Asking the API

`POST /api/chat` takes a `pipeline` field so the student assistant and the cancer assistant share one deployment:

```bash
curl -sS -X POST https://your-host/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"When do the second semester exams start?","pipeline":"tuk"}'

curl -sS -X POST https://your-host/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the mechanism of trastuzumab?","pipeline":"cancer"}'
```

`POST /api/chat/stream` is the same request as Server-Sent Events, which is what the UI uses. Omit `pipeline` and you get `tuk`.

---

## Project structure

```text
tuk-convosearch/
├── app/                                    # the FastAPI service
│   ├── main.py                             # app factory; lifespan warms index + model, serves the built SPAs
│   ├── config.py                           # pydantic-settings (env prefix TUK_), production validators
│   ├── api/
│   │   ├── chat.py                         # POST /api/chat + /api/chat/stream (SSE), history by id
│   │   ├── admin.py                        # documents, index stats, logs (JWT-protected)
│   │   ├── auth_routes.py                  # POST /api/auth/login, GET /api/auth/me
│   │   ├── health.py                       # /health (liveness) vs /api/health (readiness)
│   │   ├── deps.py                         # shared dependencies: auth, rate limit, settings
│   │   └── schemas.py                      # request/response models, incl. the pipeline field
│   ├── services/
│   │   ├── chat.py                         # async boundary + concurrency gate around the pipelines
│   │   ├── pipelines.py                    # builds each pipeline once and reuses it per name
│   │   ├── database.py                     # SQLite: sessions, messages, chunks, documents, query_log
│   │   ├── ollama.py                       # the only place that talks to Ollama
│   │   ├── auth.py                         # password hashing + JWT for the admin API
│   │   └── rate_limit.py                   # in-process fixed-window limiter
│   ├── ingestion/
│   │   ├── documents.py                    # text extraction, chunking, filename sanitising
│   │   └── index.py                        # CorpusIndex: FAISS over vectors stored as SQLite BLOBs
│   └── rag/                                # ← the framework
│       ├── core/
│       │   ├── base_agent.py               # Agent interface
│       │   ├── context.py                  # RAGContext (shared state)
│       │   └── pipeline.py                 # Orchestrator (retries, short-circuit)
│       ├── agents/
│       │   ├── router_agent.py
│       │   ├── query_expansion_agent.py
│       │   ├── retriever_agent.py
│       │   ├── synthesizer_agent.py
│       │   └── critic_agent.py
│       ├── stores/
│       │   ├── base_store.py               # BaseVectorStore interface
│       │   ├── faiss_store.py              # Local FAISS implementation
│       │   └── pubmed_store.py             # NCBI E-utilities + FAISS
│       └── configs/
│           ├── tuk_pipeline.py             # student assistant config
│           └── cancer_pipeline.py          # cancer research config
│
├── web/                                    # React 18 + Vite student SPA (served at /)
│   └── src/                                # App.jsx, api.js, components/Message|Composer|Sources|Drawer.jsx
├── admin/                                  # React admin panel (served at /admin)
│
├── corpus/tuk/                             # the TU-K source documents that get indexed
├── data/                                   # the SQLite database (single file, the whole store)
├── pubmed_index/                           # cancer corpus snapshot (FAISS + metadata)
│
├── scripts/
│   ├── create_admin.py                     # make the first admin account
│   └── ingest_corpus.py                    # out-of-band ingestion (--force re-embeds everything)
├── evals/                                  # Evaluation harness
│   ├── questions.py                        # Fixed question set
│   └── runner.py                           # Scoring + reporting
├── deploy/                                 # Dockerfile, docker-compose.yml, Caddyfile, *.sh, RUNBOOK.md
├── tests/                                  # Unit tests (pytest)
└── pytest.ini
```

The split worth noticing: `app/rag/` is the framework and has no knowledge of HTTP or SQLite. `app/` above it is the product — FastAPI routes, auth, rate limiting, the ingestion pipeline, and the single-file database that holds documents, chunks, vectors and chat history.

---

## The front ends

Two small React apps, both built ahead of time and served as static files by the FastAPI process. There is no Node runtime in production and no second origin to configure CORS for.

- **`web/`** — the student SPA, mounted at `/`. React 18 + Vite; `src/App.jsx`, `src/api.js` and four components (`Message`, `Composer`, `Sources`, `Drawer`). About 50 KB of gzipped JS, no webfonts, no icon library. It streams answers over SSE using `POST /api/chat/stream` (POST, not EventSource, because the question goes in the body).
- **`admin/`** — the admin panel, mounted at `/admin`. Document upload and re-ingestion, index stats, query logs. Its API calls require a JWT from `/api/auth/login`.

```bash
npm --prefix web   ci && npm --prefix web   run build   # → web/dist, served at /
npm --prefix admin ci && npm --prefix admin run build   # → admin/dist, served at /admin
```

`admin/vite.config.js` sets `base: '/admin/'`; if that is reverted the panel loads a blank page because its bundle asks for the wrong asset paths.

## Deployment

Everything deployment-related lives in `deploy/`, and **[`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) is the operator guide** — read it rather than reverse-engineering the scripts.

- `Dockerfile` — installs CPU-only torch (the default wheel drags ~2 GB of CUDA into an image that never uses a GPU), downloads `all-MiniLM-L6-v2` at build time so a container start does not depend on Hugging Face being reachable, and runs as the non-root uid 10001.
- `docker-compose.yml` — `ollama`, `app`, `caddy` on a private network. Only Caddy publishes ports (443, plus 80 for redirects and ACME challenges).
- `Caddyfile` — automatic HTTPS, SSE buffering disabled, security headers.
- `.env.example` — every `TUK_` variable with its default and why it exists.
- `bootstrap.sh`, `deploy.sh`, `backup.sh`, `restore.sh`, `pull-models.sh`, `tuk-backup.cron`.

Two production guards live in `app/config.py` rather than in a checklist, because checklists get skipped: a short `TUK_SECRET_KEY` and a `*` CORS origin both raise at startup when `TUK_ENVIRONMENT=production`. The data that matters is `data/tuk_convosearch.db` (documents, chunks, vectors, sessions, query log) and `corpus/` — those two paths are what `make backup` archives.

Students have no login, so a session id is also the only access token a conversation has. That rules out one obvious-looking feature: there is no `GET /api/sessions`. A list of every conversation would publish those ids along with every title, and `GET /api/sessions/{id}/messages` trusts the id. The operator view is `GET /api/admin/sessions`, behind the same JWT as the rest of `/api/admin`, and the drawer in the student UI reads its list from `localStorage` (`web/src/history.js`) rather than from the server.

---

## Extending the framework

The whole point of this project is that adding a new domain is one config file, not a rewrite.

### Example: adding a legal research pipeline

```python
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
```

Three things need to change:

- Domain keywords for the router
- System prompt for the synthesizer
- Vector store for the corpus

Everything else — the pipeline, the retries, the critic logic, the short-circuit behavior — stays identical.

### Available extension points

| Component | What you can swap | Where to plug in |
|-----------|-------------------|------------------|
| Vector store | FAISS, PubMed, Chroma, pgvector | Subclass BaseVectorStore |
| Router logic | Keyword, LLM, hybrid | Subclass Agent |
| Query expansion | LLM, MeSH dictionary, none | Add a QueryExpansionAgent or skip |
| Synthesizer | Ollama, OpenAI, Anthropic | Pass a different model_name or agent |
| Critic | Heuristic, LLM, hybrid | Swap mode or subclass ValidationAgent |

---

## Testing

Unit tests (fast, no I/O):

```bash
python -m pytest
```

Plain pytest, not `make test` — the root `Makefile` only wraps operations on a deployed box.

124 unit tests, and none of them need Ollama: the pipeline, the embedding model and the corpus index are faked (`tests/conftest.py` plus per-test monkeypatching), so the suite runs offline in CI and on a laptop. `pytest.ini` excludes anything marked `integration` by design; `tests/test_integration_pipeline.py` is that mark — a real end-to-end run over your own `data/` and `corpus/`, skipped automatically when Ollama or the model is missing. Run it on a box that has both:

```bash
pytest -m integration
```

Evaluation harness (slow, runs full pipelines):

```bash
python -m evals.runner tuk heuristic > evals/baseline_tuk_1b.txt
python -m evals.runner cancer llm     > evals/baseline_cancer_llm.txt
```

Arguments are `<pipeline> <critic_mode> <enable_expansion>`; defaults are `tuk heuristic true`.

The harness measures per-question:

- Route accuracy — did the router classify correctly?
- Retrieval hit — did the top-k chunks contain the expected keywords?
- Refusal correctness — for a question the corpus cannot answer, did the assistant say so instead of inventing a number?
- Citation present — did the synthesizer cite sources?
- Latency — how long did the full pipeline take?

Current TU-K numbers, llama3.2:1b with the heuristic critic and 25 indexed chunks of text (five readable documents):

| metric | score |
|---|---|
| Questions passed | 15/17 (88%) |
| Route accuracy | 17/17 (100%) |
| Retrieval hit | 17/17 (100%) |
| Citation present | 15/17 (88%) |
| Average latency | 12.6 s |

The two red lines are known and named: one scanned PDF that extracts to zero characters until someone OCRs it (`needs_ocr`), and one question whose answer does not exist in the corpus, where a 1-billion-parameter model invents a page reference instead of refusing. That is the strongest argument for `critic_mode=llm` or a larger model on the VPS.

Numbers are committed to evals/ after each significant change, so you can diff improvements.

---

## Roadmap

What's done, what's next, what's not planned.

### ✅ Done

- Modular agent pipeline with retries and short-circuit routing
- Two domain pipelines (TU-K student assistant, cancer research) behind one endpoint
- 124 unit tests, all isolated from LLMs and I/O, running in GitHub Actions CI (Python 3.11 and 3.12), plus a marked end-to-end integration test for a box that has Ollama
- Evaluation harness with real numbers
- Live PubMed ingestion via NCBI E-utilities
- Citation enforcement in the synthesizer
- Deterministic refusal handling in the critic
- FastAPI service: SSE streaming, sessions and query log in a single SQLite file
- Corpus ingestion for PDF/DOCX/TXT/MD with filename sanitising, FAISS over vectors stored as blobs
- Admin panel with JWT auth, document upload and re-ingestion; per-IP rate limiting
- Docker deployment (Ollama + app + Caddy) with automatic HTTPS, backups and a rollback path — `deploy/`, driven by the root `Makefile`
- Liveness (`/health`) separated from readiness (`/api/health`)

### 🚧 Next

- Published PyPI package (pip install tuk-convosearch)
- Documentation site (MkDocs or Docusaurus)
- Additional store backends (Chroma, pgvector)
- Streaming support for the critic stage

### ❌ Not planned

- A hosted SaaS version
- Clinical decision support (this is a research tool, not a medical device)
- Fine-tuning custom models

If you want any of the "not planned" items, fork the project — the architecture supports all of them.

## Contributing

Contributions are welcome. The best place to start:

- good first issue labels on GitHub
- Add a new pipeline config — pick a domain you know, follow the cancer pipeline as a template
- Write an eval question set for an existing domain
- Improve the docs — especially the extension guide

Before opening a PR:

```bash
python -m pytest              # all tests must pass
python -m evals.runner tuk    # baseline must not regress
```

If you're adding a new agent or store, include tests. If you're adding a pipeline config, include a couple of eval questions.

For larger changes, open an issue first to discuss the approach.

## Citation

If you use this framework in academic work:

```bibtex
@software{tuk_convosearch_2026,
  author = {Kirionki, Denis Lemayian},
  title = {TUK-ConvoSearch: A Modular RAG Framework for High-Stakes Domains},
  year = {2026},
  url = {https://github.com/lemayian23/tuk-convosearch}
}
```

## License

MIT — see LICENSE.

## Author

Denis Lemayian Kirionki
School of Computing and Information Technology
Technical University of Kenya

Built with support from the TU-K School of Computing and Information Technology. Supervised by Mr. Shadrack Ngumbau and Mr. Dalmas Owira.

Open an issue for questions, bugs, or feature requests.
