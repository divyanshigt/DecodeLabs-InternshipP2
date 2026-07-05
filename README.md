# 📄 AI PDF Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and have a grounded, hallucination-resistant conversation with it — powered by **Groq's Llama 3.3 70B Versatile**, **FAISS** vector search, and **Sentence-Transformers** embeddings, wrapped in a premium, ChatGPT-inspired dark UI built entirely in Streamlit.

> Upload a document. Ask a question. Get an answer sourced only from the document — never a guess.

---

## ✨ Features

- **Strict, grounded answers** — the model is instructed to answer *only* from retrieved PDF context. If the answer isn't in the document, it says so instead of hallucinating.
- **Semantic search, not keyword search** — questions are embedded and matched against document chunks using cosine similarity in FAISS.
- **Streamed responses** — answers render token-by-token for a natural, ChatGPT-like typing experience.
- **Smart chunking** — recursive character splitting (1000 chars, 200 overlap) preserves semantic boundaries better than naive fixed-size splitting.
- **Full conversation memory** — chat history persists for the session, with timestamps and downloadable transcripts.
- **Live system diagnostics** — sidebar shows real-time status of the Groq connection, embedding model, and FAISS index.
- **Document metrics** — page count, character/word counts, chunk count, and embedding dimensionality, all surfaced after processing.
- **Resilient by design** — every I/O and network boundary (PDF parsing, embedding, API calls) is wrapped in error handling with user-facing, non-technical messages.
- **Cached model & client loading** — the embedding model and Groq client are loaded once per session via `st.cache_resource`, keeping the app fast after the first run.
- **Fully custom dark UI** — glassmorphism, gradient backgrounds, and hover-animated cards; zero default Streamlit styling.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  PDF Upload  │ --> │  Text Extraction  │ --> │  Chunking (LangChain) │
└─────────────┘     └──────────────────┘     └────────────────────┘
                                                        │
                                                        v
                                          ┌────────────────────────────┐
                                          │  Embedding (MiniLM-L6-v2)   │
                                          └────────────────────────────┘
                                                        │
                                                        v
                                              ┌───────────────────┐
                                              │   FAISS Index      │
                                              │  (cosine / IP)      │
                                              └───────────────────┘
                                                        │
                       ┌────────────────────────────────┘
                       v
        ┌───────────────────────┐        ┌─────────────────────────┐
        │  User Question         │ -----> │  Top-K Chunk Retrieval    │
        └───────────────────────┘        └─────────────────────────┘
                                                        │
                                                        v
                                       ┌───────────────────────────────┐
                                       │  Grounded Prompt Construction   │
                                       └───────────────────────────────┘
                                                        │
                                                        v
                                        ┌─────────────────────────────┐
                                        │  Groq — Llama 3.3 70B (stream) │
                                        └─────────────────────────────┘
                                                        │
                                                        v
                                              ┌───────────────────┐
                                              │  Streamed Answer UI │
                                              └───────────────────┘
```

**Why FAISS `IndexFlatIP` with normalized vectors?** Normalizing embeddings and using inner product is mathematically equivalent to cosine similarity, which is the standard, well-behaved metric for sentence-embedding retrieval — and `IndexFlatIP` gives exact (non-approximate) search, which is appropriate at the scale of a single document.

---

## 🧰 Tech Stack

| Layer                | Technology                          |
|----------------------|--------------------------------------|
| UI Framework         | Streamlit                           |
| LLM Inference        | Groq API — Llama 3.3 70B Versatile  |
| Embeddings           | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Store         | FAISS (CPU)                         |
| Text Chunking        | LangChain Text Splitters             |
| PDF Parsing          | pypdf                                |
| Numerics             | NumPy                                |
| Config               | python-dotenv                        |

---

## 📦 Requirements

- Python 3.11+
- A free [Groq API key](https://console.groq.com/keys)

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-pdf-assistant.git
cd ai-pdf-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then open .env and paste in your own GROQ_API_KEY
```

> ⚠️ **Security note:** Never commit your `.env` file or paste your real API key anywhere public (chat logs, GitHub issues, screenshots). If a key is ever exposed, revoke and regenerate it immediately from the Groq console.

---

## ▶️ How to Run

```bash
streamlit run chat.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📁 Folder Structure

```
ai-pdf-assistant/
├── chat.py              # Main application (UI + RAG pipeline)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Files/folders excluded from version control
└── README.md            # Project documentation
```

---

## 📸 Screenshots

> _Add screenshots of the hero section, chat interface, and sidebar here once deployed locally, e.g.:_
>
> `docs/screenshot-hero.png`, `docs/screenshot-chat.png`, `docs/screenshot-sidebar.png`

---

## 🔭 Future Improvements

- Multi-PDF sessions with per-document source citations
- Support for `.docx` and `.txt` uploads alongside PDF
- Persistent vector store (e.g., FAISS index saved to disk / Chroma) across sessions
- User authentication and per-user chat history storage
- Highlighted source-chunk preview under each answer
- Streaming token usage / cost tracking in the sidebar
- Dockerfile + one-click deployment guide (Streamlit Community Cloud / Render)

---

## 👤 Developer

Built as a portfolio / internship-showcase project demonstrating end-to-end RAG system design: document ingestion, semantic retrieval, grounded generation, and production-grade UI engineering.

---

## 📄 License

This project is provided as-is for educational and portfolio purposes.