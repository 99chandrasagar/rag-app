from sentence_transformers import CrossEncoder
from app.config import get_settings


class Reranker:
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.enable_reranking
        self.model = CrossEncoder(settings.reranker_model) if self.enabled else None

    def rerank(self, query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
        if not self.enabled or not self.model or not chunks:
            return chunks[:top_k]
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        enriched = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            enriched.append(item)
        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        return enriched[:top_k]
