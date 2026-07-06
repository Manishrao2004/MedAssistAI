"""
Medical NLP Chatbot — Streamlit Application.

A production-quality medical question answering interface powered by
RAG (Retrieval-Augmented Generation) with Medical NER, Semantic Search,
and LLM generation via Groq.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medical_chatbot.preprocessing.text_processor import TextProcessor
from medical_chatbot.ner.medical_ner import MedicalNER
from medical_chatbot.embeddings.encoder import EmbeddingEncoder
from medical_chatbot.vector_store.faiss_store import FAISSStore
from medical_chatbot.retriever.semantic_retriever import SemanticRetriever
from medical_chatbot.rag.pipeline import RAGPipeline
from medical_chatbot.llm.groq_client import GroqClient
from medical_chatbot.utils.config import load_config, get_env


# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedAssist AI — Medical NLP Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS — Dark Theme Optimized
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    /* Main header */
    .main-header {
        background: #0f172a;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #1e293b;
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Entity badges */
    .entity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 4px 4px;
        border: 1px solid transparent;
    }
    .entity-disease { background: rgba(220, 38, 38, 0.1); color: #fca5a5; border-color: rgba(248, 113, 113, 0.2); }
    .entity-symptom { background: rgba(234, 88, 12, 0.1); color: #fdba74; border-color: rgba(251, 146, 60, 0.2); }
    .entity-drug { background: rgba(37, 99, 235, 0.1); color: #93c5fd; border-color: rgba(96, 165, 250, 0.2); }
    .entity-anatomy { background: rgba(22, 163, 74, 0.1); color: #86efac; border-color: rgba(74, 222, 128, 0.2); }

    /* Clean Cards */
    .glass-card, .chunk-card, .metric-card, .entities-section {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
    }

    /* Chunk card */
    .chunk-card {
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .chunk-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    .chunk-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #cbd5e1;
    }

    /* Score indicator */
    .score-high { color: #10b981; font-weight: 600; }
    .score-medium { color: #f59e0b; font-weight: 600; }
    .score-low { color: #ef4444; font-weight: 600; }

    /* Answer box */
    .answer-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.7;
        font-size: 1rem;
        color: #f1f5f9 !important;
    }
    .answer-box p, .answer-box li, .answer-box span {
        color: #f1f5f9 !important;
    }

    /* Metric cards */
    .metric-card {
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }

    /* Sidebar info card */
    .sidebar-info {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        line-height: 1.8;
        color: #94a3b8 !important;
    }
    .sidebar-info strong {
        color: #e2e8f0;
    }

    /* Entities section */
    .entities-section {
        padding: 1.2rem;
        height: 100%;
    }
    .entities-section .entity-count {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 1rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }

    /* Clean buttons */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #94a3b8 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        border-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    button[kind="primary"] {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
    }
    button[kind="primary"]:hover {
        background: #f8fafc !important;
    }

    /* Hide ONLY main menu and footer, keep header so sidebar toggle works */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fix Streamlit dark expander text */
    .streamlit-expanderHeader p {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Initialize Components (cached)
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading NLP models and FAISS index...")
def load_components():
    """Load and cache all pipeline components."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    # Text Processor
    text_processor = TextProcessor(config["preprocessing"]["spacy_model"])

    # Medical NER
    medical_ner = MedicalNER(config["ner"]["model"])

    # Embedding Encoder
    encoder = EmbeddingEncoder(
        model_name=config["embeddings"]["model_name"],
        normalize=config["embeddings"]["normalize"],
    )

    # FAISS Store
    faiss_store = FAISSStore(dimension=config["embeddings"]["dimension"])
    index_path = str(PROJECT_ROOT / config["vector_store"]["index_path"])
    metadata_path = str(PROJECT_ROOT / config["vector_store"]["metadata_path"])

    if not Path(index_path).exists() or not Path(metadata_path).exists():
        st.error(
            "⚠️ FAISS index not found. Please run the indexing pipeline first:\n\n"
            "```bash\n"
            "uv run python scripts/prepare_dataset.py\n"
            "uv run python scripts/build_index.py\n"
            "```"
        )
        st.stop()

    faiss_store.load(index_path, metadata_path)

    # Semantic Retriever
    retriever = SemanticRetriever(
        encoder=encoder,
        faiss_store=faiss_store,
        text_processor=text_processor,
        min_similarity=config["retriever"]["min_similarity"],
    )

    # RAG Pipeline
    rag_pipeline = RAGPipeline()

    # Groq LLM Client
    try:
        api_key = get_env("GROQ_API_KEY")
        llm_client = GroqClient(
            api_key=api_key,
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
        )
    except (EnvironmentError, ValueError) as e:
        st.error(f"⚠️ LLM Configuration Error: {e}")
        st.stop()

    return {
        "config": config,
        "text_processor": text_processor,
        "medical_ner": medical_ner,
        "encoder": encoder,
        "faiss_store": faiss_store,
        "retriever": retriever,
        "rag_pipeline": rag_pipeline,
        "llm_client": llm_client,
    }


def get_score_class(score: float) -> str:
    """Return CSS class based on similarity score."""
    if score >= 0.65:
        return "score-high"
    elif score >= 0.40:
        return "score-medium"
    return "score-low"


def render_entities_html(entities: dict[str, list[str]]) -> str:
    """Render entities as styled HTML badges."""
    html_parts: list[str] = []
    category_class = {
        "DISEASE": "entity-disease",
        "SYMPTOM": "entity-symptom",
        "DRUG": "entity-drug",
        "ANATOMY": "entity-anatomy",
    }
    category_icon = {
        "DISEASE": "🦠",
        "SYMPTOM": "🤒",
        "DRUG": "💊",
        "ANATOMY": "🫁",
    }

    for category, items in entities.items():
        for item in items:
            cls = category_class.get(category, "entity-disease")
            icon = category_icon.get(category, "•")
            html_parts.append(
                f'<span class="entity-badge {cls}">{icon} {item}</span>'
            )

    return "".join(html_parts) if html_parts else '<span style="color:#94a3b8;">No medical entities detected</span>'


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

def render_sidebar(config: dict) -> int:
    """Render the sidebar with system info and settings."""
    with st.sidebar:
        st.markdown("## ⚙️ System Info")

        st.markdown(f"""
        <div class="sidebar-info">
            <strong>🧠 LLM:</strong> {config['llm']['model']}<br>
            <strong>🔗 Provider:</strong> Groq (LPU)<br>
            <strong>📊 Embeddings:</strong> MiniLM-L6-v2<br>
            <strong>🗄️ Vector DB:</strong> FAISS<br>
            <strong>🔬 NER:</strong> SciSpacy<br>
            <strong>🌡️ Temperature:</strong> {config['llm']['temperature']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📚 About")
        st.markdown("""
        **MedAssist AI** is a medical question-answering system that uses:

        - **Medical NER** to detect diseases, symptoms, drugs & anatomy
        - **Semantic Search** over the MedQuAD knowledge base
        - **RAG** to ground LLM responses in real medical literature
        - **Groq LPU** for ultra-fast inference

        ⚠️ *This tool is for educational purposes only and does not replace professional medical advice.*
        """)

        st.markdown("---")
        st.markdown("## 🎛️ Settings")
        top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=config["retriever"]["top_k"])
        return top_k


# ─────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────

def main() -> None:
    """Run the Streamlit application."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedAssist AI</h1>
        <p>Intelligent Medical Q&A powered by NLP, Semantic Search, RAG & Large Language Models</p>
    </div>
    """, unsafe_allow_html=True)

    # Load components
    components = load_components()
    config = components["config"]
    retriever = components["retriever"]
    medical_ner = components["medical_ner"]
    rag_pipeline = components["rag_pipeline"]
    llm_client = components["llm_client"]

    # Sidebar
    top_k = render_sidebar(config)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Query input
    st.markdown("### 💬 Ask a Medical Question")

    with st.form("search_form", clear_on_submit=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Enter your medical question",
                placeholder="e.g., What are the symptoms of diabetes?",
                label_visibility="collapsed",
            )
        with col_btn:
            submit = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")

    # Example queries
    st.markdown("**💡 Try these examples:**")
    example_cols = st.columns(4)
    examples = [
        "What are the symptoms of diabetes?",
        "How is glaucoma treated?",
        "What causes high blood pressure?",
        "What is Alzheimer's disease?",
    ]
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                query = example
                submit = True

    # Process query
    if submit and query and query.strip():
        st.markdown("---")

        total_start = time.perf_counter()

        search_query = query
        # ── Step 0: Query Reformulation ──
        if st.session_state.chat_history:
            with st.spinner("🧠 Understanding context..."):
                last_turn = st.session_state.chat_history[-1]
                rewrite_prompt = (
                    f"Given the following conversation history, rewrite the user's latest "
                    f"follow-up question to be a standalone search query that contains all "
                    f"necessary context (e.g. replace pronouns like 'it' with the actual medical condition). "
                    f"If the question is already standalone, return it exactly as is.\n\n"
                    f"Previous Question: {last_turn['query']}\n"
                    f"Previous Answer: {last_turn['answer'][:200]}...\n"
                    f"Latest Question: {query}\n\n"
                    f"Standalone Search Query:"
                )
                try:
                    rewritten = llm_client.generate(
                        prompt=rewrite_prompt,
                        system_prompt="You are a query rewriting assistant. Output ONLY the rewritten query, nothing else. Do not use quotes."
                    )
                    search_query = rewritten.strip(' "\'\n')
                    if search_query.lower() != query.lower():
                        st.caption(f"*(Contextualized search: {search_query})*")
                except Exception:
                    pass

        # ── Step 1: Medical NER ──
        with st.spinner("🔬 Extracting medical entities..."):
            entities = medical_ner.extract_entities(search_query)

        # ── Step 2: Semantic Retrieval ──
        with st.spinner("📚 Searching knowledge base..."):
            retrieval_start = time.perf_counter()
            results = retriever.retrieve(search_query, top_k=top_k)
            retrieval_time = time.perf_counter() - retrieval_start

        # ── Step 3: RAG + LLM ──
        with st.spinner("🤖 Generating answer with AI..."):
            if results:
                prompt = rag_pipeline.build_prompt(query, results, entities)
                llm_start = time.perf_counter()
                
                # Build conversation history for the LLM
                history_msgs = []
                for entry in st.session_state.chat_history[-3:]:
                    history_msgs.append({"role": "user", "content": entry["query"]})
                    history_msgs.append({"role": "assistant", "content": entry["answer"]})

                answer = llm_client.generate(
                    prompt=prompt,
                    system_prompt=rag_pipeline.get_system_prompt(),
                    history=history_msgs
                )
                llm_time = time.perf_counter() - llm_start
            else:
                answer = rag_pipeline.build_no_context_response(query)
                llm_time = 0.0

        total_time = time.perf_counter() - total_start
        avg_sim = sum(r.similarity_score for r in results) / len(results) if results else 0

        # ── Display Results ──

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card glass-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label">Results Found</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card glass-card">
                <div class="metric-value">{avg_sim:.3f}</div>
                <div class="metric-label">Avg Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card glass-card">
                <div class="metric-value">{retrieval_time:.2f}s</div>
                <div class="metric-label">Retrieval Time</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card glass-card">
                <div class="metric-value">{total_time:.2f}s</div>
                <div class="metric-label">Total Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Two-column layout: Answer + Entities
        col_answer, col_entities = st.columns([3, 1])

        with col_answer:
            st.markdown('<div class="section-header">🤖 Generated Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with col_entities:
            st.markdown('<div class="section-header">🔬 Medical Entities</div>', unsafe_allow_html=True)
            entities_html = render_entities_html(entities)
            total_ent = sum(len(v) for v in entities.values())
            st.markdown(f"""
            <div class="entities-section glass-card">
                {entities_html}
                <div class="entity-count">{total_ent} entities detected</div>
            </div>
            """, unsafe_allow_html=True)

        # Retrieved Sources
        st.markdown("---")
        st.markdown('<div class="section-header">📚 Retrieved Sources</div>', unsafe_allow_html=True)

        if results:
            for i, result in enumerate(results, 1):
                score_class = get_score_class(result.similarity_score)
                with st.expander(
                    f"📄 Source {i}  —  {result.focus_area}  |  "
                    f"Score: {result.similarity_score:.3f} ({result.confidence})",
                    expanded=(i <= 2),
                ):
                    st.markdown(f"""
                    <div class="chunk-card glass-card">
                        <div class="chunk-meta">
                            📌 <strong>Topic:</strong> {result.focus_area} &nbsp;&bull;&nbsp;
                            📂 <strong>Source:</strong> {result.source} &nbsp;&bull;&nbsp;
                            🎯 <strong>Similarity:</strong>
                            <span class="{score_class}">{result.similarity_score:.4f}</span> &nbsp;&bull;&nbsp;
                            📊 <strong>Confidence:</strong> {result.confidence}
                        </div>
                        <div class="chunk-text">{result.chunk_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption(f"Original question: _{result.question}_")
        else:
            st.warning("No relevant sources found for this query.")

        # Similarity Score Chart
        if results:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Similarity Scores</div>', unsafe_allow_html=True)
            chart_data = {
                f"#{i+1} {r.focus_area[:25]}": r.similarity_score
                for i, r in enumerate(results)
            }
            st.bar_chart(chart_data, horizontal=True, color="#14b8a6")

        # Add to chat history
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "entities": entities,
            "num_results": len(results),
            "avg_similarity": avg_sim,
            "total_time": total_time,
        })

    elif submit and (not query or not query.strip()):
        st.warning("⚠️ Please enter a medical question.")

    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown('<div class="section-header">🕒 Query History</div>', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(
                f"Q: {entry['query'][:80]}  |  ⏱️ {entry['total_time']:.2f}s",
                expanded=False,
            ):
                st.markdown(f"**Answer:**\n\n{entry['answer']}")
                st.caption(
                    f"Results: {entry['num_results']} | "
                    f"Avg Similarity: {entry['avg_similarity']:.3f} | "
                    f"Time: {entry['total_time']:.2f}s"
                )


if __name__ == "__main__":
    main()
