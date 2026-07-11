from pathlib import Path
import tempfile

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.chunking import chunk_document, clean_text
from app.config import get_settings
from app.embeddings import get_embedding_model
from app.ingest import ingest_documents
from app.llms import get_llm
from app.loaders import load_file
from app.rag import SYSTEM_PROMPT, answer_question, format_context
from app.reranker import Reranker
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkDebugChunk,
    ChunkDebugRequest,
    ChunkDebugResponse,
    DocumentInput,
    IngestRequest,
    IngestResponse,
    PipelineDebugRequest,
    PipelineDebugResponse,
    RetrieveDebugRequest,
    RetrieveDebugResponse,
    SourceChunk,
)
from app.vectorstore import QdrantVectorStore


app = FastAPI(
    title="Production RAG API",
    version="1.1.0",
    description="RAG API with ingestion, chunking strategies, Qdrant retrieval, reranking, debug pipeline endpoints, and pluggable LLMs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.app_env == "dev":
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _source_chunks_from_results(chunks: list[dict]) -> list[SourceChunk]:
    sources: list[SourceChunk] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        sources.append(
            SourceChunk(
                text=chunk.get("text", ""),
                source=metadata.get("source", "unknown"),
                title=metadata.get("title"),
                score=chunk.get("rerank_score") or chunk.get("score"),
                metadata=metadata,
            )
        )
    return sources


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    settings = get_settings()
    return {
        "chunk_strategies": [
            {
                "name": "fixed",
                "best_for": "Logs, simple flat text, first baseline.",
                "tradeoff": "May cut sentences or sections in the middle.",
            },
            {
                "name": "recursive",
                "best_for": "Default for PDFs, DOCX, web pages, and general text.",
                "tradeoff": "Good balance, but not aware of all document-specific structure.",
            },
            {
                "name": "sentence",
                "best_for": "FAQs, policies, support articles, short explanatory content.",
                "tradeoff": "Chunk sizes can be uneven.",
            },
            {
                "name": "markdown",
                "best_for": "Markdown docs with # headings.",
                "tradeoff": "Works best when headings are meaningful.",
            },
            {
                "name": "semantic",
                "best_for": "Research-style prose where topic boundaries matter.",
                "tradeoff": "Slower because it embeds sentences during chunking.",
            },
            {
                "name": "parent_child",
                "best_for": "Long legal, policy, architecture, or research documents.",
                "tradeoff": "Stores more metadata; may increase storage size.",
            },
            {
                "name": "code",
                "best_for": "Python, JavaScript, TypeScript, and function/class-like code.",
                "tradeoff": "Lightweight parser; not a full AST parser.",
            },
            {
                "name": "table",
                "best_for": "CSV-like or markdown-table-like text.",
                "tradeoff": "Basic table detection only.",
            },
        ],
        "llm_providers": ["echo", "openai", "anthropic", "gemini", "ollama"],
        "current_config": {
            "qdrant_collection": settings.qdrant_collection,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "enable_reranking": settings.enable_reranking,
            "reranker_model": settings.reranker_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        },
    }


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(request: IngestRequest):
    chunks_indexed = ingest_documents(
        documents=request.documents,
        chunk_strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return IngestResponse(
        collection=get_settings().qdrant_collection,
        documents_received=len(request.documents),
        chunks_indexed=chunks_indexed,
    )


@app.post("/ingest/file", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_file(
    file: UploadFile = File(...),
    chunk_strategy: str = "recursive",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
):
    suffix = Path(file.filename or "upload.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    text = load_file(tmp_path)
    doc = DocumentInput(
        text=text,
        source=file.filename or "uploaded_file",
        title=file.filename,
        metadata={"ingestion_type": "file_upload"},
    )
    chunks_indexed = ingest_documents(
        documents=[doc],
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return IngestResponse(
        collection=get_settings().qdrant_collection,
        documents_received=1,
        chunks_indexed=chunks_indexed,
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest):
    answer, chunks = answer_question(
        question=request.question,
        top_k=request.top_k,
        rerank_top_k=request.rerank_top_k,
        filters=request.filters,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        temperature=request.temperature,
    )
    return ChatResponse(answer=answer, sources=_source_chunks_from_results(chunks))


@app.post("/debug/chunk", response_model=ChunkDebugResponse, dependencies=[Depends(require_api_key)])
def debug_chunk(request: ChunkDebugRequest):
    embedding_model = None

    def embed_for_semantic(texts: list[str]) -> list[list[float]]:
        nonlocal embedding_model
        if embedding_model is None:
            embedding_model = get_embedding_model()
        return embedding_model.embed_texts(texts)

    original_char_count = len(request.text)
    cleaned = clean_text(request.text)

    chunks = chunk_document(
        text=request.text,
        strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        embed_fn=embed_for_semantic,
    )

    debug_chunks = []
    lengths = []
    for idx, chunk in enumerate(chunks):
        char_count = len(chunk.text)
        lengths.append(char_count)
        debug_chunks.append(
            ChunkDebugChunk(
                index=idx,
                text=chunk.text,
                char_count=char_count,
                word_count=len(chunk.text.split()),
                metadata=chunk.metadata,
            )
        )

    return ChunkDebugResponse(
        strategy=request.chunk_strategy,
        original_char_count=original_char_count,
        cleaned_char_count=len(cleaned),
        chunk_count=len(debug_chunks),
        min_chunk_chars=min(lengths) if lengths else None,
        max_chunk_chars=max(lengths) if lengths else None,
        avg_chunk_chars=(sum(lengths) / len(lengths)) if lengths else None,
        chunks=debug_chunks,
    )


@app.post("/debug/retrieve", response_model=RetrieveDebugResponse, dependencies=[Depends(require_api_key)])
def debug_retrieve(request: RetrieveDebugRequest):
    settings = get_settings()
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(request.question)

    vectorstore = QdrantVectorStore(vector_size=embedding_model.dimension())
    retrieved = vectorstore.search(
        query_vector=query_vector,
        top_k=request.top_k,
        filters=request.filters,
    )

    return RetrieveDebugResponse(
        question=request.question,
        embedding_model=settings.embedding_model,
        embedding_dimension=embedding_model.dimension(),
        query_vector_preview=[round(float(x), 6) for x in query_vector[:12]],
        retrieved=_source_chunks_from_results(retrieved),
    )


@app.post("/debug/pipeline", response_model=PipelineDebugResponse, dependencies=[Depends(require_api_key)])
def debug_pipeline(request: PipelineDebugRequest):
    settings = get_settings()

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(request.question)

    vectorstore = QdrantVectorStore(vector_size=embedding_model.dimension())
    retrieved = vectorstore.search(
        query_vector=query_vector,
        top_k=request.top_k,
        filters=request.filters,
    )

    reranked = Reranker().rerank(
        query=request.question,
        chunks=retrieved,
        top_k=request.rerank_top_k,
    )

    context = format_context(chunks=reranked, max_chars=settings.max_context_chars)
    user_prompt = f"""
Question:
{request.question}

Context:
{context}

Answer:
"""

    answer = None
    if request.run_llm:
        llm = get_llm(provider=request.llm_provider, model=request.llm_model)
        answer = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=request.temperature,
        )

    stage_summary = [
        {
            "stage": "Embed Query",
            "what_happened": "The user question was converted into a numeric vector.",
            "details": {
                "embedding_model": settings.embedding_model,
                "embedding_dimension": embedding_model.dimension(),
                "query_vector_preview": [round(float(x), 6) for x in query_vector[:12]],
            },
        },
        {
            "stage": "Qdrant Search",
            "what_happened": "Qdrant searched for chunks whose vectors are closest to the query vector.",
            "details": {"requested_top_k": request.top_k, "retrieved_count": len(retrieved)},
        },
        {
            "stage": "Optional Reranker",
            "what_happened": "The reranker rescored the retrieved chunks using the query and chunk text together.",
            "details": {
                "enabled": settings.enable_reranking,
                "reranker_model": settings.reranker_model,
                "rerank_top_k": request.rerank_top_k,
                "reranked_count": len(reranked),
            },
        },
        {
            "stage": "Prompt Builder",
            "what_happened": "The selected chunks were formatted into a grounded prompt with citation metadata.",
            "details": {"context_chars": len(context), "prompt_chars": len(user_prompt)},
        },
        {
            "stage": "LLM Provider",
            "what_happened": "The prompt can be sent to echo, OpenAI, Anthropic, Gemini, or Ollama.",
            "details": {
                "provider": request.llm_provider or settings.llm_provider,
                "model": request.llm_model or settings.llm_model,
                "run_llm": request.run_llm,
            },
        },
    ]

    return PipelineDebugResponse(
        question=request.question,
        stage_summary=stage_summary,
        retrieved=_source_chunks_from_results(retrieved),
        reranked=_source_chunks_from_results(reranked),
        context=context,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        answer=answer,
    )
