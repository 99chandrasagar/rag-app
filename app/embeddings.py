from abc import ABC, abstractmethod
from typing import Sequence

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def dimension(self) -> int:
        return int(self._dimension)


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, model_name: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = 1536 if "small" in model_name else 3072

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def dimension(self) -> int:
        return self._dimension


def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedding(settings.embedding_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbedding(settings.embedding_model, settings.openai_api_key)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
