from typing import Any

from app.chunking import chunk_document, stable_hash
from app.embeddings import get_embedding_model
from app.vectorstore import QdrantVectorStore
from app.schemas import DocumentInput


def ingest_documents(documents: list[DocumentInput], chunk_strategy: str = "recursive", chunk_size: int = 900, chunk_overlap: int = 150) -> int:
    embedding_model = get_embedding_model()
    vectorstore = QdrantVectorStore(vector_size=embedding_model.dimension())
    all_texts: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    def embed_for_semantic(texts: list[str]) -> list[list[float]]:
        return embedding_model.embed_texts(texts)

    for doc_idx, doc in enumerate(documents):
        doc_hash = stable_hash(doc.text)
        base_metadata = {
            "source": doc.source,
            "title": doc.title,
            "document_index": doc_idx,
            "document_hash": doc_hash,
            **doc.metadata,
        }
        chunks = chunk_document(
            text=doc.text,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_fn=embed_for_semantic,
        )
        for chunk in chunks:
            chunk_hash = stable_hash(chunk.text)
            metadata = {**base_metadata, **chunk.metadata, "chunk_hash": chunk_hash}
            all_texts.append(chunk.text)
            all_metadatas.append(metadata)

    if not all_texts:
        return 0
    vectors = embedding_model.embed_texts(all_texts)
    return vectorstore.upsert_chunks(texts=all_texts, vectors=vectors, metadatas=all_metadatas)
