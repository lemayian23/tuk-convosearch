# 🎓 TUK-ConvoSearch

An AI-powered student assistant for **The Technical University of Kenya (TU-K)** using **Retrieval-Augmented Generation (RAG)**.

TUK-ConvoSearch helps students access accurate, document-grounded information from official university documents — 24/7, with source citations.

---

## 📖 Overview

### The Problem

Students at TU-K struggle to find accurate institutional information because it is scattered across:

- The official university website (poor search relevance)
- The student portal (limited to fee payment and e-learning)
- Physical notice boards (outdated, require physical presence)
- WhatsApp groups (misinformation, peer rumours)
- Administrative offices (long queues, restricted hours)

Students spend **3-4 hours per week** searching for basic information like exam dates, fee structures, and registration procedures.

### The Solution

**TUK-ConvoSearch** is a RAG-powered chatbot that:

1. Ingests official TU-K documents (academic calendars, exam timetables, fee structures, student handbooks)
2. Indexes them into a FAISS vector database
3. Processes student questions using natural language
4. Retrieves the most relevant document chunks
5. Generates grounded answers using a local LLM (Ollama + llama3.2:1b)
6. Provides **source citations** with every answer

**No hallucination. No misinformation. Just accurate, verifiable answers.**

---

## ✨ Features

### Student Features

| Feature | Description |
|---------|-------------|
| **Natural Language Q&A** | Ask questions in plain English |
| **Source Citations** | Every answer shows which document it came from |
| **24/7 Availability** | No office hours required |
| **No Installation** | Open in any browser, no login required |
| **Streaming Responses** | Word-by-word display as the answer is generated |
| **Off-Topic Rejection** | Politely rejects non-TU-K questions |
| **Dark Mode** | Toggle between light and dark themes |
| **Conversation History** | Contextual follow-up questions |

### Admin Features

| Feature | Description |
|---------|-------------|
| **JWT Authentication** | Secure admin login |
| **Document Upload** | PDF, DOCX, TXT support |
| **Automatic Re-indexing** | FAISS index updates automatically |
| **Document Removal** | Archive documents without permanent deletion |
| **Query Logs** | View all student queries with responses |
| **System Statistics** | Active documents, total queries, average response time |

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.12, FastAPI | API server and orchestration |
| **Student Frontend** | Vanilla HTML/CSS/JS | Chat interface (no build tools) |
| **Admin Frontend** | React + Vite | Document management panel |
| **Vector Database** | FAISS (IndexFlatL2) | Fast semantic similarity search |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dimensional text embeddings |
| **LLM** | Ollama + llama3.2:1b | Local answer generation |
| **Database** | SQLite | Query logs and document metadata |
| **Authentication** | JWT + bcrypt | Secure admin access |
| **Streaming** | Server-Sent Events (SSE) | Word-by-word response delivery |

---

## 📋 Prerequisites

- **Python** 3.10+
- **Ollama** 0.19+ (with llama3.2:1b model pulled)
- **Node.js** 18+ (for admin panel only)
- **Git** (for cloning the repository)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/lemayian23/tuk-convosearch.git
cd tuk-convosearch

### Step 2: Create Virtual Environment
bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/Mac
### Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Install Ollama and Pull Model
bash
###Step 4: Download Ollama from https://ollama.com/download
ollama pull llama3.2:1b


### Step 5: Configure Environment
bash
copy .env.example .env
notepad .env
Edit .env with your settings:

env
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./faiss_index/faiss_index.bin
METADATA_PATH=./faiss_index/metadata.pkl
OLLAMA_MODEL=llama3.2:1b
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./data/tuk_convosearch.db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.5
MAX_CRITIC_RETRIES=2
JWT_SECRET_KEY=your_secret_key_here


###Step 6: Add Documents to Knowledge Base
Place official TU-K documents in the docs/ folder:
docs/
├── e-Booklet March 2026.pdf
├── SEMESTER TWO MAY 2026 EXAMINATION TIMETABLE.pdf
├── SATUK-DELEGATES-25-3-26.pdf
├── university_info.txt
└── Upgrade-From-Diploma-to-Degree-at-TU-K-2026.pdf
Supported formats: PDF, DOCX, TXT




Step 7: Build the FAISS Index
bash
python rebuild_faiss.py
Expected output:
Loading documents from ./docs/...
Loaded 7 documents
Created 66 chunks
Embeddings generated
FAISS index saved to ./faiss_index/faiss_index.bin
Database records created: 7 documents
Step 8: Start the Backend Server
bash
python -m uvicorn app.main:app --reload
Server will start at: http://localhost:8000



### Step 9: Start the Admin Panel (Optional)
Open a new terminal:
bash
cd admin
npm install
npm run dev
Admin panel will start at: http://localhost:5173

### Step 10: Access the System
Component	URL
Student Chat	Open frontend/index.html in any browser
Admin Panel	http://localhost:5173
API Docs	http://localhost:8000/docs
📁 Project Structure



tuk-convosearch/
│
├── app/
│   ├── main.py                      # FastAPI entrypoint
│   ├── api/
│   │   ├── chat_proposal.py         # Student chat endpoints (SSE streaming)
│   │   └── admin.py                 # Protected admin endpoints
│   │
│   ├── services/
│   │   ├── auth.py                  # JWT authentication
│   │   ├── database.py              # SQLite connection manager
│   │   ├── rag_service_proposal.py  # RAG pipeline orchestration
│   │   ├── faiss_vector_store.py    # FAISS index management
│   │   ├── document_loader.py       # PDF/DOCX/TXT loading
│   │   └── chunking.py              # Paragraph-aware text chunking
│   │
│   └── utils/
│       ├── prompts.py               # Router, Synthesizer, Critic prompts
│       └── logger.py                # Logging setup
│
├── admin/                           # React admin panel
│   ├── src/
│   │   └── App.jsx                  # Login, documents, logs, stats
│   └── package.json
│
├── frontend/
│   └── index.html                   # Student chat interface
│
├── docs/                            # Knowledge base (TU-K documents)
│   ├── e-Booklet March 2026.pdf
│   ├── SEMESTER TWO MAY 2026 EXAMINATION TIMETABLE.pdf
│   └── ...
│
├── docs_archive/                    # Archived documents (soft-deleted)
│
├── faiss_index/
│   ├── faiss_index.bin              # FAISS vector index
│   └── metadata.pkl                 # Chunk text and source metadata
│
├── data/
│   └── tuk_convosearch.db           # SQLite database
│
├── rebuild_faiss.py                 # Document ingestion script
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
🔌 API Endpoints
Health Check
http
GET /health
Response:

json
{
  "status": "ok",
  "vector_count": 66,
  "model_loaded": true,
  "version": "1.0.0"
}
Ask a Question
http
POST /api/chat
Content-Type: application/json

{
  "question": "When do the second semester exams start?"
}
Response:

json
{
  "answer": "The second semester examination period begins on 15th May 2026...",
  "sources": [
    {
      "source": "SEMESTER TWO MAY 2026 EXAMINATION TIMETABLE.pdf",
      "text": "Examinations for the second semester will commence on 15th May 2026..."
    }
  ],
  "grounded": true,
  "route_used": "internal_docs"
}
Streaming Chat (SSE)
http
POST /api/chat/stream
Content-Type: application/json

{
  "question": "When do the second semester exams start?",
  "session_id": "abc123"
}
Response: Server-Sent Events stream with tokens.





Upload Document (Admin)
POST /api/admin/documents
Authorization: Bearer <token>
Content-Type: multipart/form-data



API Documentation


http://localhost:8000/docs
🎯 Multi-Agent Architecture




Student Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ROUTER AGENT                                               │
│  Decides: internal_docs / web_search / both                 │
│  Uses keyword matching for off-topic detection              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVER AGENT                                            │
│  Searches FAISS vector store for relevant document chunks   │
│  Returns top-5 chunks by similarity score                   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  SYNTHESIZER AGENT                                          │
│  Generates answer using Ollama (llama3.2:1b)                │
│  Constrained to retrieved context only                      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  CRITIC AGENT                                               │
│  Validates answer is grounded in retrieved context          │
│  If not grounded → triggers retry                           │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Final Answer + Source Citations


🖥 Frontend Features
Student Chat Interface
Feature	Description
Chat UI	DeepSeek-style dark theme
Message Bubbles	User (right) and Assistant (left)
Streaming	Word-by-word response display
Source Panel	Expandable citations
Grounded Badge	✅ Grounded or ⚠️ Not Grounded
Dark Mode Toggle	Switch between light and dark
Suggestion Chips	Common questions for quick access
No Login Required	Open in any browser
Admin Panel
Feature	Description
Secure Login	JWT authentication
Document Upload	PDF, DOCX, TXT support
Document Library	View all active documents
Remove Documents	Archive without deletion
Query Logs	View all student queries
Statistics	Active documents, total queries, average response time




🧪 Testing
Test with Swagger UI
Open http://localhost:8000/docs

Click POST /api/chat

Click "Try it out"

Enter:

json
{"question": "When do the second semester exams start?"}
Click "Execute"

Test with CLI
bash
python scripts/test_query.py
Sample Queries
Query	Expected Behavior
"When do exams start?"	Retrieves from examination timetable
"How do I upgrade from diploma to degree?"	Retrieves from upgrade guidelines
"Where is TU-K located?"	Retrieves from university info
"What is the weather in Nairobi?"	Off-topic rejection




📊 Key Metrics
Metric	Value
Documents Ingested	7
Text Chunks	66
Embedding Dimension	384
Average Response Time	18.65 seconds
Cache Hit Response Time	Under 1 second
Citation Rate	100%
Test Cases Passed	20/20
Student Satisfaction	11/12 easy to use
Preference over Office	10/12 students
🚢 Deployment
Local Deployment
Runs on any machine with Python 3.10+ and Ollama.

Raspberry Pi 5 Deployment
RAM: 8GB recommended

Storage: 64GB+ microSD (A2 class) or NVMe SSD

OS: Raspberry Pi OS (64-bit) Bookworm

Power: Official 27W USB-C power supply

Production Considerations
Add HTTPS with a reverse proxy (Nginx/Caddy)

Use PostgreSQL instead of SQLite

Set up monitoring and logging

Add rate limiting

Configure CORS appropriately



🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📄 License
This project is licensed under the MIT License.

👨‍💻 Developer
Denis Lemayian Kirionki

0799801096

School of Computing and Information Technology

Technical University of Kenya



🙏 Acknowledgments
Technical University of Kenya (TU-K)

School of Computing and Information Technology

Supervisors: Mr. Shadrack Ngumbau and Mr. Dalmas Owira

Open-source community (FAISS, Hugging Face, Ollama, FastAPI)

📞 Contact
For questions or support, please open an issue on GitHub.

Built with ❤️ for TU-K students.


