from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.config import get_settings


class QdrantVectorStore:
    def __init__(self, vector_size: int):
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict[str, Any]]) -> int:
        self.ensure_collection()
        points = []
        for text, vector, metadata in zip(texts, vectors, metadatas):
            payload = dict(metadata)
            payload["text"] = text
            points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 8, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)
        # Compatibility with older and newer qdrant-client versions.
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "id": hit.id,
                "score": hit.score,
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            })
        return results

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        conditions = [FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()]
        return Filter(must=conditions)
