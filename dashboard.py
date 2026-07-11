import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from app.chunking import clean_text
from app.loaders import load_file


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TEXT = """# Refund Policy

Customers may request a refund within 30 days of purchase. Refunds are processed to the original payment method. Enterprise customers should contact account support for exceptions.

# Shipping Policy

Standard shipping usually takes 5 to 7 business days. Expedited shipping takes 2 to 3 business days.

# Account Security

Users should enable multi-factor authentication. Password reset links expire after 15 minutes.

# Support Escalation

Priority support is available for enterprise customers. Critical incidents should be acknowledged within one hour.
"""


st.set_page_config(
    page_title="RAG Playground Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def call_api(method: str, path: str, **kwargs) -> tuple[bool, Any]:
    try:
        response = requests.request(method, api_url(path), timeout=180, **kwargs)
        if response.ok:
            return True, response.json()
        try:
            return False, response.json()
        except Exception:
            return False, response.text
    except Exception as exc:
        return False, str(exc)


def source_card(source: dict, index: int, title_prefix: str = "Source") -> None:
    title = source.get("title") or source.get("source") or "unknown"
    score = source.get("score")
    metadata = source.get("metadata") or {}
    chunk_index = metadata.get("chunk_index", "unknown")

    score_text = f" | score: {score:.4f}" if isinstance(score, (int, float)) else ""
    with st.expander(f"{title_prefix} {index}: {title} | chunk {chunk_index}{score_text}", expanded=index == 1):
        st.write(source.get("text", ""))
        st.caption("Metadata")
        st.json(metadata)


def load_uploaded_file_preview(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    return load_file(tmp_path)


with st.sidebar:
    st.title("RAG Playground")
    api_base = st.text_input("FastAPI base URL", API_BASE_URL)
    API_BASE_URL = api_base.rstrip("/")

    ok, health = call_api("GET", "/health")
    if ok:
        st.success("API connected")
    else:
        st.error("API not reachable")
        st.caption(str(health))

    ok, options = call_api("GET", "/options")
    if ok:
        current_config = options.get("current_config", {})
        st.caption(f"Collection: {current_config.get('qdrant_collection')}")
        st.caption(f"Embedding: {current_config.get('embedding_model')}")
        st.caption(f"Default LLM: {current_config.get('llm_provider')} / {current_config.get('llm_model')}")
    else:
        options = {}


st.title("Interactive RAG Dashboard")
st.caption("Explore the full flow: Loader → Cleaner → Chunker → Embedding → Qdrant → Reranker → Prompt → LLM → Answer + Sources")

chunk_options = [item["name"] for item in options.get("chunk_strategies", [])] or [
    "fixed", "recursive", "sentence", "markdown", "semantic", "parent_child", "code", "table"
]
llm_providers = options.get("llm_providers", ["echo", "openai", "anthropic", "gemini", "ollama"])

pipeline_tab, ingest_tab, retrieval_tab, chat_tab, compare_tab, config_tab = st.tabs(
    [
        "1. Pipeline Map",
        "2. Ingest Lab",
        "3. Retrieval Lab",
        "4. Chat Lab",
        "5. Chunk Strategy Compare",
        "6. Config",
    ]
)

with pipeline_tab:
    st.header("How this RAG system works")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("/ingest pipeline")
        ingest_steps = [
            ("Loader", "Reads raw files or text. Supported examples: PDF, DOCX, TXT, MD, HTML, CSV, source code."),
            ("Cleaner", "Normalizes whitespace, removes invalid characters, and prepares text for chunking."),
            ("Chunker", "Splits text into retrieval-sized units using fixed, recursive, sentence, markdown, semantic, parent_child, code, or table strategies."),
            ("Embedding Model", "Converts every chunk into a numeric vector that captures meaning."),
            ("Qdrant", "Stores vectors plus metadata so chunks can be searched later by similarity."),
        ]
        for stage, explanation in ingest_steps:
            with st.expander(stage, expanded=True):
                st.write(explanation)
                if stage == "Chunker":
                    st.write("Use the Ingest Lab and Compare tab to see exactly how each strategy changes chunk count, size, and metadata.")

    with col2:
        st.subheader("/chat pipeline")
        chat_steps = [
            ("Embed Query", "Converts the user question into the same vector space as document chunks."),
            ("Qdrant Search", "Finds the top_k chunks most similar to the question vector."),
            ("Optional Reranker", "Re-scores retrieved chunks using the query and chunk text together. This often improves precision."),
            ("Prompt Builder", "Formats the chosen chunks into a prompt with source, title, and chunk metadata."),
            ("LLM Provider", "Sends the grounded prompt to echo, OpenAI, Anthropic, Gemini, or Ollama."),
            ("Answer + Sources", "Returns the generated answer and the chunks used as evidence."),
        ]
        for stage, explanation in chat_steps:
            with st.expander(stage, expanded=True):
                st.write(explanation)

    st.subheader("What to try first")
    st.markdown(
        """
        1. Go to **Ingest Lab** and preview chunks with `recursive`.
        2. Click **Ingest text into Qdrant**.
        3. Go to **Retrieval Lab** and ask: `What is the refund policy?`.
        4. Go to **Chat Lab**, choose `echo`, and run the full answer.
        5. Compare `recursive`, `markdown`, `sentence`, and `parent_child` in the compare tab.
        """
    )

with ingest_tab:
    st.header("Ingest Lab: Loader → Cleaner → Chunker → Embedding → Qdrant")

    input_mode = st.radio("Input mode", ["Paste text", "Upload file"], horizontal=True)
    text = DEFAULT_TEXT
    uploaded_file = None

    if input_mode == "Paste text":
        text = st.text_area("Document text", DEFAULT_TEXT, height=260)
        source = st.text_input("Source", "dashboard_manual.md")
        title = st.text_input("Title", "Dashboard Manual Test")
    else:
        uploaded_file = st.file_uploader("Upload PDF, TXT, MD, DOCX, HTML, CSV, or code file")
        source = uploaded_file.name if uploaded_file else "uploaded_file"
        title = uploaded_file.name if uploaded_file else "Uploaded File"
        if uploaded_file:
            try:
                text = load_uploaded_file_preview(uploaded_file)
                st.success(f"Loaded preview from {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not preview file: {exc}")
                text = ""

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        chunk_strategy = st.selectbox("Chunk strategy", chunk_options, index=chunk_options.index("recursive") if "recursive" in chunk_options else 0)
    with col_b:
        chunk_size = st.slider("Chunk size", 100, 3000, 900, step=50)
    with col_c:
        chunk_overlap = st.slider("Chunk overlap", 0, 800, 150, step=25)

    cleaned = clean_text(text or "")
    m1, m2, m3 = st.columns(3)
    m1.metric("Original characters", len(text or ""))
    m2.metric("Cleaned characters", len(cleaned))
    m3.metric("Cleaned words", len(cleaned.split()))

    with st.expander("Cleaner preview"):
        st.text_area("Cleaned text", cleaned[:10000], height=220)

    preview_clicked = st.button("Preview chunks", type="secondary")
    ingest_clicked = st.button("Ingest into Qdrant", type="primary")

    if preview_clicked:
        if not text.strip():
            st.warning("Add text or upload a supported file first.")
        else:
            payload = {
                "text": text,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            ok, result = call_api("POST", "/debug/chunk", json=payload)
            if not ok:
                st.error("Chunk preview failed")
                st.write(result)
            else:
                st.success(f"Generated {result['chunk_count']} chunks")
                cols = st.columns(4)
                cols[0].metric("Chunks", result["chunk_count"])
                cols[1].metric("Min chars", result.get("min_chunk_chars") or 0)
                cols[2].metric("Max chars", result.get("max_chunk_chars") or 0)
                cols[3].metric("Avg chars", round(result.get("avg_chunk_chars") or 0, 1))

                df = pd.DataFrame(
                    [
                        {
                            "index": c["index"],
                            "char_count": c["char_count"],
                            "word_count": c["word_count"],
                            "metadata": json.dumps(c["metadata"]),
                        }
                        for c in result["chunks"]
                    ]
                )
                st.bar_chart(df.set_index("index")[["char_count"]])
                st.dataframe(df, use_container_width=True)

                for c in result["chunks"][:20]:
                    with st.expander(f"Chunk {c['index']} | {c['char_count']} chars | {c['word_count']} words"):
                        st.write(c["text"])
                        st.json(c["metadata"])

    if ingest_clicked:
        if input_mode == "Upload file" and uploaded_file is not None:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
            params = {
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            ok, result = call_api("POST", "/ingest/file", files=files, params=params)
        else:
            payload = {
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "documents": [
                    {
                        "text": text,
                        "source": source,
                        "title": title,
                        "metadata": {"created_from": "streamlit_dashboard"},
                    }
                ],
            }
            ok, result = call_api("POST", "/ingest", json=payload)

        if ok:
            st.success("Ingested successfully")
            st.json(result)
        else:
            st.error("Ingestion failed")
            st.write(result)

with retrieval_tab:
    st.header("Retrieval Lab: Embed Query → Qdrant Search → Optional Reranker → Prompt Builder")

    question = st.text_input("Question", "What is the refund policy?")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Qdrant top_k", 1, 30, 8)
    with col2:
        rerank_top_k = st.slider("Rerank top_k", 1, 15, 4)
    with col3:
        run_llm_for_trace = st.checkbox("Also run LLM", value=False)

    filters_raw = st.text_input("Optional filters as JSON", "")
    filters = None
    if filters_raw.strip():
        try:
            filters = json.loads(filters_raw)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid filter JSON: {exc}")

    col_a, col_b = st.columns(2)
    with col_a:
        retrieve_clicked = st.button("Run retrieve only")
    with col_b:
        trace_clicked = st.button("Run full pipeline trace", type="primary")

    if retrieve_clicked:
        payload = {"question": question, "top_k": top_k, "filters": filters}
        ok, result = call_api("POST", "/debug/retrieve", json=payload)
        if ok:
            st.success("Retrieved chunks from Qdrant")
            c1, c2 = st.columns(2)
            c1.metric("Embedding dimension", result["embedding_dimension"])
            c2.metric("Retrieved chunks", len(result["retrieved"]))
            st.caption("Query vector preview")
            st.code(result["query_vector_preview"])
            for i, source in enumerate(result["retrieved"], start=1):
                source_card(source, i, "Retrieved")
        else:
            st.error("Retrieve failed")
            st.write(result)

    if trace_clicked:
        payload = {
            "question": question,
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
            "filters": filters,
            "llm_provider": "echo",
            "llm_model": "echo",
            "temperature": 0.1,
            "run_llm": run_llm_for_trace,
        }
        ok, result = call_api("POST", "/debug/pipeline", json=payload)
        if ok:
            st.success("Pipeline trace complete")
            st.subheader("Stage summary")
            st.dataframe(pd.DataFrame(result["stage_summary"]), use_container_width=True)

            st.subheader("Retrieved before reranking")
            for i, source in enumerate(result["retrieved"], start=1):
                source_card(source, i, "Retrieved")

            st.subheader("Reranked final context chunks")
            for i, source in enumerate(result["reranked"], start=1):
                source_card(source, i, "Reranked")

            with st.expander("Prompt context sent to the LLM"):
                st.text_area("Context", result["context"], height=260)
            with st.expander("System prompt"):
                st.text_area("System prompt", result["system_prompt"], height=160)
            with st.expander("User prompt"):
                st.text_area("User prompt", result["user_prompt"], height=260)
            if result.get("answer"):
                st.subheader("LLM answer")
                st.write(result["answer"])
        else:
            st.error("Pipeline trace failed")
            st.write(result)

with chat_tab:
    st.header("Chat Lab: Full /chat endpoint")

    question = st.text_input("Ask a question", "What is the refund policy?", key="chat_question")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_k = st.slider("top_k", 1, 30, 8, key="chat_topk")
    with col2:
        rerank_top_k = st.slider("rerank_top_k", 1, 15, 4, key="chat_rerank")
    with col3:
        provider = st.selectbox("LLM provider", llm_providers, index=llm_providers.index("echo") if "echo" in llm_providers else 0)
    with col4:
        temperature = st.slider("Temperature", 0.0, 1.5, 0.1, step=0.1)

    model_defaults = {
        "echo": "echo",
        "openai": "gpt-4.1-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "gemini": "gemini-2.5-flash",
        "ollama": "llama3.1",
    }
    model = st.text_input("Model", model_defaults.get(provider, ""))

    if st.button("Ask /chat", type="primary"):
        payload = {
            "question": question,
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
            "filters": None,
            "llm_provider": provider,
            "llm_model": model,
            "temperature": temperature,
        }
        ok, result = call_api("POST", "/chat", json=payload)
        if ok:
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Sources")
            for i, source in enumerate(result["sources"], start=1):
                source_card(source, i)
        else:
            st.error("Chat failed")
            st.write(result)

with compare_tab:
    st.header("Chunk Strategy Compare")
    st.write("Use this tab to run all chunkers on the same text and compare their behavior before indexing anything.")

    compare_text = st.text_area("Comparison text", DEFAULT_TEXT, height=260, key="compare_text")
    col1, col2 = st.columns(2)
    with col1:
        compare_chunk_size = st.slider("Compare chunk size", 100, 3000, 900, step=50)
    with col2:
        compare_overlap = st.slider("Compare overlap", 0, 800, 150, step=25)

    selected_strategies = st.multiselect("Strategies", chunk_options, default=[s for s in ["fixed", "recursive", "sentence", "markdown", "parent_child"] if s in chunk_options])

    if st.button("Compare selected strategies", type="primary"):
        rows = []
        results_by_strategy = {}
        for strategy in selected_strategies:
            payload = {
                "text": compare_text,
                "chunk_strategy": strategy,
                "chunk_size": compare_chunk_size,
                "chunk_overlap": compare_overlap,
            }
            ok, result = call_api("POST", "/debug/chunk", json=payload)
            if ok:
                rows.append(
                    {
                        "strategy": strategy,
                        "chunks": result["chunk_count"],
                        "min_chars": result.get("min_chunk_chars"),
                        "max_chars": result.get("max_chunk_chars"),
                        "avg_chars": round(result.get("avg_chunk_chars") or 0, 1),
                    }
                )
                results_by_strategy[strategy] = result
            else:
                rows.append({"strategy": strategy, "error": str(result)})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        if "chunks" in df.columns:
            st.bar_chart(df.set_index("strategy")[["chunks"]])

        for strategy, result in results_by_strategy.items():
            with st.expander(f"Preview chunks for {strategy}"):
                for c in result["chunks"][:10]:
                    st.markdown(f"**Chunk {c['index']} — {c['char_count']} chars**")
                    st.write(c["text"])
                    st.json(c["metadata"])

with config_tab:
    st.header("Current API configuration")
    ok, result = call_api("GET", "/options")
    if ok:
        st.json(result.get("current_config", {}))
        st.subheader("Chunk strategy guide")
        st.dataframe(pd.DataFrame(result.get("chunk_strategies", [])), use_container_width=True)
    else:
        st.error("Could not load config")
        st.write(result)

    st.subheader("Important environment variables")
    st.code(
        """APP_ENV=dev
QDRANT_URL=http://qdrant:6333       # inside docker compose
QDRANT_COLLECTION=rag_documents
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_RERANKING=true
LLM_PROVIDER=echo                  # no API key needed for testing
OPENAI_API_KEY=...                 # required only for OpenAI
ANTHROPIC_API_KEY=...              # required only for Anthropic
GOOGLE_API_KEY=...                 # required only for Gemini
OLLAMA_BASE_URL=http://ollama:11434 # with docker-compose.ollama.yml
"""
    )
