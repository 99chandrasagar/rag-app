from typing import Any, Literal
from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    text: str
    source: str = "manual"
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ChunkStrategy = Literal[
    "fixed",
    "recursive",
    "sentence",
    "markdown",
    "semantic",
    "parent_child",
    "code",
    "table",
]


class IngestRequest(BaseModel):
    documents: list[DocumentInput]
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = 900
    chunk_overlap: int = 150


class IngestResponse(BaseModel):
    collection: str
    documents_received: int
    chunks_indexed: int


class ChatRequest(BaseModel):
    question: str
    top_k: int = 8
    rerank_top_k: int = 4
    filters: dict[str, Any] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float = 0.1


class SourceChunk(BaseModel):
    text: str
    source: str
    title: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ChunkDebugRequest(BaseModel):
    text: str
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = 900
    chunk_overlap: int = 150


class ChunkDebugChunk(BaseModel):
    index: int
    text: str
    char_count: int
    word_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkDebugResponse(BaseModel):
    strategy: str
    original_char_count: int
    cleaned_char_count: int
    chunk_count: int
    min_chunk_chars: int | None = None
    max_chunk_chars: int | None = None
    avg_chunk_chars: float | None = None
    chunks: list[ChunkDebugChunk]


class RetrieveDebugRequest(BaseModel):
    question: str
    top_k: int = 8
    filters: dict[str, Any] | None = None


class RetrieveDebugResponse(BaseModel):
    question: str
    embedding_model: str
    embedding_dimension: int
    query_vector_preview: list[float]
    retrieved: list[SourceChunk]


class PipelineDebugRequest(BaseModel):
    question: str
    top_k: int = 8
    rerank_top_k: int = 4
    filters: dict[str, Any] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float = 0.1
    run_llm: bool = False


class PipelineDebugResponse(BaseModel):
    question: str
    stage_summary: list[dict[str, Any]]
    retrieved: list[SourceChunk]
    reranked: list[SourceChunk]
    context: str
    system_prompt: str
    user_prompt: str
    answer: str | None = None
