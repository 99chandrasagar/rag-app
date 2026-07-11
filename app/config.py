from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    api_key: str = "change-me"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_documents"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranking: bool = True

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    default_top_k: int = 8
    default_rerank_top_k: int = 4
    max_context_chars: int = 18_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
