
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Third-party ML / retrieval stack
try:
    import faiss
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from groq import Groq
except ImportError as missing_dep:  # pragma: no cover - environment guard
    # Streamlit hasn't rendered a page yet at import time in some execution
    # contexts, so we raise a clear message that will surface in the terminal.
    raise SystemExit(
        f"Missing dependency: {missing_dep}. "
        "Run `pip install -r requirements.txt` before starting the app."
    )

load_dotenv()

APP_NAME = "AI PDF Assistant"
APP_TAGLINE = "Upload a PDF and ask intelligent questions powered by Llama 3."
APP_ICON = "📄"

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 4
LLM_TEMPERATURE = 0.2
MAX_RESPONSE_TOKENS = 1024

NO_ANSWER_MESSAGE = "I couldn't find that information in the uploaded PDF."

SYSTEM_PROMPT_TEMPLATE = """You are a precise document-analysis assistant. You answer \
questions strictly and exclusively using the CONTEXT extracted from a PDF document \
provided below. You never use outside knowledge, never speculate, and never \
fabricate information that is not explicitly present in the context.

Rules you must always follow:
1. Only answer using facts found in the CONTEXT section.
2. If the CONTEXT does not contain enough information to answer the question, \
respond with exactly: "{no_answer}"
3. Do not mention that you are an AI, do not mention "the context" explicitly in \
your answer — just answer naturally as if you had read the document.
4. Be concise, accurate, and well-structured. Use bullet points or short \
paragraphs when helpful.
5. Never invent page numbers, statistics, or facts not present in the context.

CONTEXT:
{context}
"""


st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css() -> None:
    """Inject the complete custom CSS theme, overriding all Streamlit defaults."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ---------- Global background ---------- */
        .stApp {
            background: radial-gradient(circle at 15% 10%, #1b1035 0%, transparent 45%),
                        radial-gradient(circle at 85% 0%, #0f2a3f 0%, transparent 40%),
                        linear-gradient(160deg, #0b0f19 0%, #0d1120 45%, #100a1c 100%);
            background-attachment: fixed;
            color: #e6e8ef;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        /* ---------- Scrollbar ---------- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #7c3aed, #2563eb);
            border-radius: 8px;
        }

        /* ---------- Hero header ---------- */
        .hero-wrap {
            text-align: center;
            padding: 2.2rem 1rem 1.6rem 1rem;
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-family: 'Sora', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: #9aa3b8;
            font-size: 1.02rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        .hero-pill {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.14);
            border: 1px solid rgba(167, 139, 250, 0.35);
            color: #c4b5fd;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* ---------- Glass card base ---------- */
        .glass-card {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass-card:hover {
            box-shadow: 0 12px 40px rgba(124, 58, 237, 0.18);
        }

        /* ---------- Upload dropzone card ---------- */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1.5px dashed rgba(167, 139, 250, 0.45) !important;
            border-radius: 16px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(167, 139, 250, 0.8) !important;
        }

        /* ---------- Metric cards ---------- */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 0.85rem 1rem 0.6rem 1rem;
            backdrop-filter: blur(10px);
        }
        [data-testid="stMetricLabel"] { color: #9aa3b8 !important; }
        [data-testid="stMetricValue"] {
            color: #f1f2f8 !important;
            font-family: 'Sora', sans-serif;
        }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
            font-size: 0.88rem;
            letter-spacing: 0.01em;
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
            box-shadow: 0 4px 18px rgba(124, 58, 237, 0.28);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.08);
            box-shadow: 0 6px 22px rgba(124, 58, 237, 0.4);
        }
        .stButton > button:active { transform: translateY(0px); }

        /* Secondary / ghost buttons (remove, clear) via key-based container */
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.06);
            box-shadow: none;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(167, 139, 250, 0.5);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0a1a 0%, #0a0e17 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }
        .sidebar-brand-icon {
            font-size: 1.6rem;
            filter: drop-shadow(0 0 10px rgba(124, 58, 237, 0.6));
        }
        .sidebar-brand-name {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            font-size: 1.08rem;
            color: #f1f2f8;
        }
        .sidebar-section-title {
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            color: #6b7280;
            font-weight: 700;
            margin: 1.1rem 0 0.5rem 0;
        }
        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.4rem 0.65rem;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 0.4rem;
            font-size: 0.83rem;
        }
        .status-label { color: #b8bfd1; }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .dot-green { background: #34d399; box-shadow: 0 0 8px #34d399aa; }
        .dot-red { background: #f87171; box-shadow: 0 0 8px #f87171aa; }
        .dot-amber { background: #fbbf24; box-shadow: 0 0 8px #fbbf24aa; }

        .pdf-info-card {
            background: rgba(124, 58, 237, 0.08);
            border: 1px solid rgba(167, 139, 250, 0.25);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.6rem;
        }
        .pdf-info-title {
            font-weight: 600;
            font-size: 0.86rem;
            color: #e9e7fd;
            word-break: break-word;
            margin-bottom: 0.3rem;
        }
        .pdf-info-meta {
            font-size: 0.78rem;
            color: #9aa3b8;
        }

        /* ---------- Chat message bubbles (styling native st.chat_message) ---------- */
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 0.4rem 0.2rem;
            margin-bottom: 0.7rem;
            backdrop-filter: blur(10px);
        }

        .chat-timestamp {
            font-size: 0.71rem;
            color: #6b7280;
            margin-top: 0.15rem;
            margin-left: 0.2rem;
        }

        /* ---------- Chat input ---------- */
        [data-testid="stChatInput"] textarea {
            background: rgba(255, 255, 255, 0.045) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
            color: #f1f2f8 !important;
        }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            border-radius: 12px;
            backdrop-filter: blur(8px);
        }

        /* ---------- Divider ---------- */
        hr { border-color: rgba(255, 255, 255, 0.08); }

        /* ---------- About / footer text in sidebar ---------- */
        .about-text {
            font-size: 0.8rem;
            color: #8b93a7;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-embedding model for the lifetime of the process."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Optional[Groq]:
    """Instantiate and cache the Groq client. Returns None if no API key is set."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def init_session_state() -> None:
    """Initialize all session-state keys used across the app, exactly once."""
    defaults = {
        "messages": [],            # list[{"role", "content", "timestamp"}]
        "pdf_processed": False,
        "pdf_name": None,
        "pdf_pages": 0,
        "pdf_chars": 0,
        "pdf_words": 0,
        "chunks": [],
        "faiss_index": None,
        "embedding_dim": None,
        "processing_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pdf_state() -> None:
    """Clear everything related to the currently loaded PDF."""
    st.session_state.pdf_processed = False
    st.session_state.pdf_name = None
    st.session_state.pdf_pages = 0
    st.session_state.pdf_chars = 0
    st.session_state.pdf_words = 0
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.embedding_dim = None
    st.session_state.processing_error = None


def reset_chat_state() -> None:
    """Clear the conversation history only, keeping the loaded PDF intact."""
    st.session_state.messages = []


def extract_text_from_pdf(uploaded_file) -> Tuple[str, int]:
    """
    Extract raw text from an uploaded PDF file object.

    Returns:
        (full_text, page_count)

    Raises:
        ValueError if the PDF cannot be parsed or contains no extractable text.
    """
    try:
        reader = PdfReader(uploaded_file)
    except Exception as exc:
        raise ValueError(f"This file could not be read as a valid PDF ({exc}).")

    if len(reader.pages) == 0:
        raise ValueError("The uploaded PDF appears to have no pages.")

    page_texts = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            # Skip unreadable pages rather than failing the whole document.
            page_texts.append("")

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without an OCR text layer."
        )

    return full_text, len(reader.pages)


def split_text_into_chunks(text: str) -> List[str]:
    """Split document text into overlapping, semantically-aware chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]


def build_faiss_index(
    chunks: List[str], model: SentenceTransformer
) -> Tuple["faiss.Index", int]:
    """Embed all chunks and build a cosine-similarity FAISS index over them."""
    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, dimension


def process_uploaded_pdf(uploaded_file) -> None:
    """
    Full ingestion pipeline: extract -> chunk -> embed -> index.
    Updates session_state in place and renders progress feedback.
    """
    progress_bar = st.progress(0, text="Reading PDF...")
    try:
        # Step 1: extract text
        full_text, page_count = extract_text_from_pdf(uploaded_file)
        progress_bar.progress(30, text="Splitting into chunks...")

        # Step 2: chunk
        chunks = split_text_into_chunks(full_text)
        if not chunks:
            raise ValueError("Document text could not be split into usable chunks.")
        progress_bar.progress(55, text="Loading embedding model...")

        # Step 3: embed + index
        model = load_embedding_model()
        progress_bar.progress(75, text="Generating embeddings and building index...")
        index, dimension = build_faiss_index(chunks, model)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)

        # Persist results into session state
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_pages = page_count
        st.session_state.pdf_chars = len(full_text)
        st.session_state.pdf_words = len(full_text.split())
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.session_state.embedding_dim = dimension
        st.session_state.pdf_processed = True
        st.session_state.processing_error = None

    except ValueError as known_error:
        st.session_state.processing_error = str(known_error)
        st.session_state.pdf_processed = False
    except Exception as unexpected_error:
        st.session_state.processing_error = (
            f"An unexpected error occurred while processing the PDF: {unexpected_error}"
        )
        st.session_state.pdf_processed = False
    finally:
        progress_bar.empty()

def retrieve_relevant_chunks(question: str, k: int = TOP_K_CHUNKS) -> List[str]:
    """Embed the question and return the top-k most similar document chunks."""
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return []

    model = load_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    k = min(k, len(st.session_state.chunks))
    _, indices = st.session_state.faiss_index.search(query_embedding, k)

    chunks = st.session_state.chunks
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def build_messages(question: str, context_chunks: List[str]) -> List[dict]:
    """Construct the message payload sent to the Groq chat completion endpoint."""
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        no_answer=NO_ANSWER_MESSAGE,
        context=context_text,
    )

    # Include the last few turns for conversational continuity, without
    # letting history override the document-grounding rules above.
    history_messages = []
    for msg in st.session_state.messages[-6:]:
        history_messages.append({"role": msg["role"], "content": msg["content"]})

    return (
        [{"role": "system", "content": system_prompt}]
        + history_messages
        + [{"role": "user", "content": question}]
    )


def stream_answer(question: str, context_chunks: List[str]):
    """
    Generator that yields answer tokens as they arrive from Groq, for use
    with st.write_stream(). Falls back to a single informative message on
    any failure (missing key, network error, empty context, etc.).
    """
    client = get_groq_client()

    if client is None:
        yield (
            "⚠️ No Groq API key configured. Please set `GROQ_API_KEY` in your "
            "`.env` file and restart the app."
        )
        return

    if not context_chunks:
        yield NO_ANSWER_MESSAGE
        return

    messages = build_messages(question, context_chunks)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
            stream=True,
        )
        received_any = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                received_any = True
                yield delta
        if not received_any:
            yield NO_ANSWER_MESSAGE
    except Exception as exc:
        yield f"⚠️ Something went wrong while contacting the model: {exc}"


def render_hero_header() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">{APP_ICON} {APP_NAME}</h1>
            <p class="hero-subtitle">{APP_TAGLINE}</p>
            <span class="hero-pill">⚡ RAG-powered · Llama 3.3 70B · FAISS Vector Search</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(label: str, ok: bool, ok_text: str, bad_text: str, warn: bool = False) -> str:
    dot_class = "dot-amber" if warn else ("dot-green" if ok else "dot-red")
    value_text = ok_text if ok else bad_text
    return (
        f'<div class="status-row">'
        f'<span class="status-label"><span class="status-dot {dot_class}"></span>{label}</span>'
        f'<span>{value_text}</span>'
        f'</div>'
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <span class="sidebar-brand-icon">{APP_ICON}</span>
                <span class="sidebar-brand-name">{APP_NAME}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="about-text">Document Q&A, grounded and hallucination-free.</p>', unsafe_allow_html=True)

        # ---------------- System status ----------------
        st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)

        groq_ok = get_groq_client() is not None
        status_html = render_status_row(
            "Groq Connection", groq_ok, "Connected", "Not configured"
        )
        status_html += render_status_row(
            "Embedding Model", True, EMBEDDING_MODEL_NAME, "", warn=False
        )
        faiss_ready = st.session_state.faiss_index is not None
        status_html += render_status_row(
            "FAISS Index", faiss_ready, "Ready", "Empty"
        )
        st.markdown(status_html, unsafe_allow_html=True)

        # ---------------- PDF info ----------------
        st.markdown('<div class="sidebar-section-title">Document</div>', unsafe_allow_html=True)
        if st.session_state.pdf_processed:
            st.markdown(
                f"""
                <div class="pdf-info-card">
                    <div class="pdf-info-title">📎 {st.session_state.pdf_name}</div>
                    <div class="pdf-info-meta">Pages: {st.session_state.pdf_pages} &nbsp;·&nbsp;
                    Chunks: {len(st.session_state.chunks)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="about-text">No document uploaded yet.</p>',
                unsafe_allow_html=True,
            )

        # ---------------- Actions ----------------
        st.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Remove PDF", use_container_width=True, disabled=not st.session_state.pdf_processed):
                reset_pdf_state()
                reset_chat_state()
                st.rerun()
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True, disabled=not st.session_state.messages):
                reset_chat_state()
                st.rerun()

        if st.session_state.messages:
            transcript = build_conversation_transcript()
            st.download_button(
                "⬇️ Download Conversation",
                data=transcript,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # ---------------- Theme / about ----------------
        st.markdown('<div class="sidebar-section-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p class="about-text">
            <b>Theme:</b> Dark · Glassmorphism<br/>
            <b>Retrieval:</b> FAISS cosine similarity, top-4 chunks<br/>
            <b>Model:</b> Llama 3.3 70B Versatile via Groq<br/><br/>
            This assistant answers exclusively from the content of your
            uploaded PDF. If information isn't present in the document,
            it will tell you honestly rather than guessing.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Developer</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="about-text">Built as an end-to-end RAG portfolio project — '
            'document ingestion, semantic retrieval, and grounded generation, '
            'with production-grade error handling and UI engineering.</p>',
            unsafe_allow_html=True,
        )


def render_metrics() -> None:
    st.markdown('<div class="sidebar-section-title"></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Pages", st.session_state.pdf_pages)
    cols[1].metric("Characters", f"{st.session_state.pdf_chars:,}")
    cols[2].metric("Words", f"{st.session_state.pdf_words:,}")
    cols[3].metric("Chunks", len(st.session_state.chunks))
    cols[4].metric("Embedding Dim", st.session_state.embedding_dim or "-")


def render_upload_zone() -> None:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 1.2rem;">
        <h3 style="margin-top:0;">📤 Upload a document to get started</h3>
        <p class="about-text">
        Drop in a PDF and I'll read it, index it, and answer questions using
        only what's actually written inside — no guessing, no outside facts.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None and not st.session_state.pdf_processed:
        process_uploaded_pdf(uploaded_file)
        if st.session_state.pdf_processed:
            st.success(f"✅ '{uploaded_file.name}' processed successfully!")
            st.balloons()
            st.rerun()
        elif st.session_state.processing_error:
            st.error(st.session_state.processing_error)


def build_conversation_transcript() -> str:
    """Format the full chat history as a plain-text transcript for download."""
    lines = [f"Conversation Export — {APP_NAME}", f"Document: {st.session_state.pdf_name}", "=" * 60, ""]
    for msg in st.session_state.messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{msg['timestamp']}] {speaker}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def render_chat_interface() -> None:
    render_metrics()
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # ---- Render existing conversation ----
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            st.markdown(f'<div class="chat-timestamp">{msg["timestamp"]}</div>', unsafe_allow_html=True)

    # ---- New user input ----
    question = st.chat_input("Ask a question about your PDF...")
    if not question:
        return

    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": question, "timestamp": now})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
        st.markdown(f'<div class="chat-timestamp">{now}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching the document..."):
            context_chunks = retrieve_relevant_chunks(question)
        response_text = st.write_stream(stream_answer(question, context_chunks))
        response_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f'<div class="chat-timestamp">{response_time}</div>', unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "timestamp": response_time}
    )

def main() -> None:
    init_session_state()
    inject_custom_css()
    render_sidebar()
    render_hero_header()

    if not st.session_state.pdf_processed:
        render_upload_zone()
        return

    render_chat_interface()


if __name__ == "__main__":
    main()