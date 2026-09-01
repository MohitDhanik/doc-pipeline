import sys
import os
import time
import glob
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.ingestion.document_parser import parse_files
from src.ingestion.chunker import RecursiveChunker
from src.transformation.quality_checks import QualityChecker
from src.transformation.embedder import Embedder
from src.store.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.retrieval.evaluator import evaluate, mean_metrics, BUILTIN_TEST_SET

STORE_PATH = str(ROOT / "data" / "processed" / "faiss")
RAW_DIR    = ROOT / "data" / "raw"

# ─── dark matplotlib style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
})

# ─── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DWH · Doc Pipeline",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS overrides ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace !important; }

  .header-bar {
    background: linear-gradient(90deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid #00ff88;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
  }
  .header-bar h1 { color: #00ff88; font-size: 1.6rem; margin: 0; letter-spacing: 2px; }
  .header-bar p  { color: #8b949e; font-size: 0.75rem; margin: 0; }

  .stage-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #00ff88;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
  }
  .stage-card.pending { border-left-color: #30363d; }
  .stage-card.running { border-left-color: #f0a500; }
  .stage-card.done    { border-left-color: #00ff88; }

  .metric-pill {
    display: inline-block;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #00ff88;
    margin-right: 0.4rem;
  }

  .result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    position: relative;
  }
  .rank-badge {
    position: absolute;
    top: 0.7rem; right: 0.8rem;
    background: #0d1117;
    border: 1px solid #00ff88;
    color: #00ff88;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
  }
  .source-tag {
    display: inline-block;
    background: #21262d;
    color: #58a6ff;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-size: 0.68rem;
    margin-bottom: 0.4rem;
  }
  .sim-bar { height: 4px; background: #21262d; border-radius: 2px; margin-top: 0.5rem; }
  .sim-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #00ff88, #00bcd4); }

  .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #21262d; gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px 6px 0 0;
    color: #8b949e;
    padding: 0.4rem 1.2rem;
    font-size: 0.8rem;
    letter-spacing: 1px;
  }
  .stTabs [aria-selected="true"] {
    background: #0d1117;
    border-bottom: 2px solid #00ff88 !important;
    color: #00ff88 !important;
  }

  div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 1rem;
  }
  div[data-testid="stMetricLabel"] { font-size: 0.68rem !important; color: #8b949e !important; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)

# ─── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>⬡ DWH · DOCUMENT PIPELINE</h1>
  <p>ETL · Vector Store · Semantic Retrieval · Evaluation Framework</p>
</div>
""", unsafe_allow_html=True)

# ─── session state ─────────────────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "store_loaded" not in st.session_state:
    st.session_state.store_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ─── lazy embedder ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()

# ─── try auto-load pre-built index ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pre-built index…")
def load_prebuilt():
    try:
        store = FAISSStore.load(STORE_PATH)
        emb   = get_embedder()
        return Retriever(store=store, embedder=emb), store, emb
    except Exception:
        return None, None, None

_ret, _store, _emb = load_prebuilt()
if _ret and not st.session_state.store_loaded:
    st.session_state.retriever    = _ret
    st.session_state.embedder     = _emb
    st.session_state.store_loaded = True

# ═══════════════════════════════════════════════════════════════════════════════
TAB_PIPELINE, TAB_QUERY, TAB_EVAL = st.tabs([
    "⬡  PIPELINE", "⟳  QUERY", "◈  EVALUATION"
])

# ──────────────────────────────── PIPELINE TAB ─────────────────────────────────
with TAB_PIPELINE:
    st.markdown("### Data Ingestion Pipeline")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Source Selection**")
        source_mode = st.radio("Input source", ["Sample data (pre-loaded)", "Upload documents"],
                               horizontal=True, label_visibility="collapsed")

        uploaded_paths = []
        if source_mode == "Upload documents":
            uploads = st.file_uploader(
                "Drop files here (.txt, .md, .pdf, .csv)",
                type=["txt", "md", "pdf", "csv"],
                accept_multiple_files=True,
            )
            if uploads:
                tmp_dir = tempfile.mkdtemp()
                for u in uploads:
                    p = os.path.join(tmp_dir, u.name)
                    with open(p, "wb") as f:
                        f.write(u.read())
                    uploaded_paths.append(p)
        else:
            sample_files = sorted(glob.glob(str(RAW_DIR / "*.txt")) +
                                  glob.glob(str(RAW_DIR / "*.md")) +
                                  glob.glob(str(RAW_DIR / "*.csv")))
            st.markdown(f"<span class='metric-pill'>{len(sample_files)} files found</span>", unsafe_allow_html=True)
            for f in sample_files:
                st.markdown(f"<span style='color:#58a6ff;font-size:0.75rem'>› {Path(f).name}</span>",
                            unsafe_allow_html=True)

        st.markdown("**Pipeline Configuration**")
        chunk_size    = st.slider("Chunk size (chars)", 128, 1024, 512, 64)
        chunk_overlap = st.slider("Overlap (chars)", 0, 256, 64, 16)
        min_chars     = st.slider("Min quality length", 10, 200, 50, 10)

        run_btn = st.button("▶  RUN PIPELINE", use_container_width=True, type="primary")

    with col_right:
        st.markdown("**Stage Monitor**")

        stages = ["PARSE", "CHUNK", "EMBED", "QA", "STORE"]
        stage_placeholders = [st.empty() for _ in stages]

        def render_stage(ph, name, state, detail=""):
            icons = {"pending": "○", "running": "◌", "done": "●"}
            colors= {"pending": "#30363d", "running": "#f0a500", "done": "#00ff88"}
            ic = icons.get(state, "○")
            cl = colors.get(state, "#30363d")
            ph.markdown(f"""
            <div class="stage-card {state}">
              <span style="color:{cl};font-size:1rem">{ic}</span>
              <span style="color:#c9d1d9;margin-left:0.5rem;font-size:0.85rem">{name}</span>
              <span style="color:#8b949e;font-size:0.72rem;float:right">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

        for i, (ph, name) in enumerate(zip(stage_placeholders, stages)):
            render_stage(ph, name, "pending")

        metrics_ph = st.empty()
        log_ph     = st.empty()

    # ── run ──
    if run_btn:
        file_paths = uploaded_paths if source_mode == "Upload documents" else \
                     glob.glob(str(RAW_DIR / "*.txt")) + \
                     glob.glob(str(RAW_DIR / "*.md")) + \
                     glob.glob(str(RAW_DIR / "*.csv"))

        if not file_paths:
            st.error("No files to process.")
        else:
            emb = get_embedder()

            from src.ingestion.document_parser import parse_files
            from src.ingestion.chunker import RecursiveChunker
            from src.transformation.quality_checks import QualityChecker
            from src.store.faiss_store import FAISSStore

            # Stage 1: Parse
            render_stage(stage_placeholders[0], "PARSE", "running")
            t0 = time.time()
            docs = parse_files(file_paths)
            t_parse = time.time() - t0
            render_stage(stage_placeholders[0], "PARSE", "done", f"{len(docs)} docs · {t_parse:.3f}s")

            # Stage 2: Chunk
            render_stage(stage_placeholders[1], "CHUNK", "running")
            t0 = time.time()
            chunker = RecursiveChunker(chunk_size, chunk_overlap)
            chunks = chunker.split_documents(docs)
            t_chunk = time.time() - t0
            render_stage(stage_placeholders[1], "CHUNK", "done", f"{len(chunks)} chunks · {t_chunk:.3f}s")

            # Stage 3: Embed
            render_stage(stage_placeholders[2], "EMBED", "running")
            t0 = time.time()
            chunks, embed_time = emb.embed_chunks(chunks)
            t_embed = time.time() - t0
            render_stage(stage_placeholders[2], "EMBED", "done", f"{t_embed:.2f}s")

            # Stage 4: QA
            render_stage(stage_placeholders[3], "QA", "running")
            t0 = time.time()
            checker = QualityChecker()
            report  = checker.run_all(chunks)
            good    = checker.filter_passing(chunks, report)
            t_qa    = time.time() - t0
            render_stage(stage_placeholders[3], "QA", "done",
                         f"{report.pass_rate:.0%} pass · {report.fail_count} flagged · {t_qa:.3f}s")

            # Stage 5: Store
            render_stage(stage_placeholders[4], "STORE", "running")
            t0 = time.time()
            dim   = len(good[0].embedding)
            store = FAISSStore(dim=dim)
            store.add(good)
            store.save(STORE_PATH)
            t_store = time.time() - t0
            render_stage(stage_placeholders[4], "STORE", "done", f"{len(good)} vectors · {t_store:.3f}s")

            # Update retriever
            st.session_state.retriever = Retriever(store=store, embedder=emb)
            st.session_state.store_loaded = True

            # Metrics row
            m1, m2, m3, m4 = metrics_ph.columns(4)
            m1.metric("Docs parsed", len(docs))
            m2.metric("Chunks stored", len(good))
            m3.metric("Quality pass", f"{report.pass_rate:.0%}")
            m4.metric("Embed time", f"{embed_time:.1f}s")

            log_ph.code(
                f"[pipeline] parse={t_parse:.3f}s  chunk={t_chunk:.3f}s  "
                f"embed={t_embed:.3f}s  qa={t_qa:.3f}s  store={t_store:.3f}s\n"
                f"[pipeline] issues: {report.issues[:3]}{'…' if len(report.issues)>3 else ''}",
                language="text",
            )

    # Show pre-built index status
    if st.session_state.store_loaded and not run_btn:
        st.success("Pre-built index loaded. Switch to QUERY or EVALUATION tabs.")
        m1, m2 = st.columns(2)
        m1.metric("Indexed chunks", len(st.session_state.retriever.store))
        m2.metric("Status", "READY")


# ──────────────────────────────── QUERY TAB ────────────────────────────────────
with TAB_QUERY:
    st.markdown("### Semantic Search")

    if not st.session_state.store_loaded:
        st.warning("Run the pipeline first (PIPELINE tab) or wait for pre-built index to load.")
    else:
        q_col, cfg_col = st.columns([3, 1], gap="large")
        with q_col:
            query_text = st.text_input(
                "Query",
                placeholder="e.g.  How does gradient descent work?",
                label_visibility="collapsed",
            )
        with cfg_col:
            top_k = st.number_input("top-k", min_value=1, max_value=10, value=5)

        search_btn = st.button("⟳  SEARCH", type="primary")

        if search_btn and query_text.strip():
            retriever = st.session_state.retriever
            with st.spinner("Embedding query…"):
                results = retriever.query(query_text.strip(), top_k=int(top_k))

            st.markdown(f"<p style='color:#8b949e;font-size:0.75rem'>→ {len(results)} results for: <em>{query_text}</em></p>",
                        unsafe_allow_html=True)

            for r in results:
                sim_pct = int(r.similarity * 100)
                st.markdown(f"""
                <div class="result-card">
                  <span class="rank-badge">#{r.rank}</span>
                  <span class="source-tag">{r.chunk.source}</span>
                  <span class="source-tag" style="color:#3fb950">sim {r.similarity:.3f}</span>
                  <span class="source-tag" style="color:#8b949e">{r.chunk.token_count} tokens</span>
                  <p style="font-size:0.82rem;color:#c9d1d9;margin-top:0.5rem;margin-bottom:0.3rem">{r.chunk.content[:420]}{'…' if len(r.chunk.content)>420 else ''}</p>
                  <div class="sim-bar"><div class="sim-fill" style="width:{sim_pct}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

        elif search_btn:
            st.warning("Enter a query first.")


# ──────────────────────────────── EVALUATION TAB ───────────────────────────────
with TAB_EVAL:
    st.markdown("### Retrieval Quality Evaluation")

    if not st.session_state.store_loaded:
        st.warning("Run or load the pipeline before evaluating.")
    else:
        st.markdown(f"<p style='color:#8b949e;font-size:0.75rem'>Test set: {len(BUILTIN_TEST_SET)} queries · metrics: precision@k · recall@k</p>",
                    unsafe_allow_html=True)

        eval_btn = st.button("◈  RUN EVALUATION", type="primary")

        if eval_btn:
            retriever = st.session_state.retriever
            k_values  = [1, 3, 5, 10]

            with st.spinner("Running evaluation…"):
                df      = evaluate(retriever, k_values=k_values)
                summary = mean_metrics(df, k_values=k_values)

            # ── summary metrics ──
            m_cols = st.columns(len(k_values))
            for col, row in zip(m_cols, summary.itertuples()):
                col.metric(f"P@{row.k}", f"{row.mean_precision:.2f}")

            m_cols2 = st.columns(len(k_values))
            for col, row in zip(m_cols2, summary.itertuples()):
                col.metric(f"R@{row.k}", f"{row.mean_recall:.2f}")

            # ── bar chart ──
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            x     = np.arange(len(k_values))
            width = 0.35

            for ax, metric, color, label in [
                (axes[0], "mean_precision", "#00ff88", "Mean Precision@k"),
                (axes[1], "mean_recall",    "#00bcd4", "Mean Recall@k"),
            ]:
                ax.bar(x, summary[metric], width=0.6, color=color, alpha=0.85, edgecolor="#21262d")
                ax.set_xticks(x)
                ax.set_xticklabels([f"k={k}" for k in k_values])
                ax.set_ylim(0, 1.05)
                ax.set_title(label, fontsize=10, pad=8)
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
                ax.grid(axis="y", alpha=0.4)
                for i, v in enumerate(summary[metric]):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#c9d1d9")

            fig.tight_layout(pad=2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── per-query table ──
            st.markdown("**Per-Query Results**")
            display_cols = ["query"] + [f"precision@{k}" for k in k_values] + [f"recall@{k}" for k in k_values]
            st.dataframe(
                df[display_cols].style.format({
                    **{f"precision@{k}": "{:.2f}" for k in k_values},
                    **{f"recall@{k}": "{:.2f}" for k in k_values},
                }).background_gradient(
                    subset=[f"precision@{k}" for k in k_values],
                    cmap="Greens", vmin=0, vmax=1,
                ),
                use_container_width=True,
            )
