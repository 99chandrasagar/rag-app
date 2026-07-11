from app.config import get_settings
from app.embeddings import get_embedding_model
from app.llms import get_llm
from app.reranker import Reranker
from app.vectorstore import QdrantVectorStore


SYSTEM_PROMPT = """
You are a precise RAG assistant.

Rules:
1. Answer only from the provided context.
2. If the context is insufficient, say you do not have enough information.
3. Do not invent facts.
4. Cite sources inline using [source: title or source, chunk N].
5. Prefer concise, direct answers.
"""


def format_context(chunks: list[dict], max_chars: int) -> str:
    blocks = []
    used_chars = 0
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        title = metadata.get("title") or source
        chunk_index = metadata.get("chunk_index", "unknown")
        text = chunk["text"]
        block = f"[CONTEXT {i}]\nTitle: {title}\nSource: {source}\nChunk: {chunk_index}\nText:\n{text}\n"
        if used_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = 8, rerank_top_k: int = 4, filters: dict | None = None, llm_provider: str | None = None, llm_model: str | None = None, temperature: float = 0.1) -> tuple[str, list[dict]]:
    settings = get_settings()
    embedding_model = get_embedding_model()
    vectorstore = QdrantVectorStore(vector_size=embedding_model.dimension())
    query_vector = embedding_model.embed_query(question)
    retrieved = vectorstore.search(query_vector=query_vector, top_k=top_k, filters=filters)
    final_chunks = Reranker().rerank(query=question, chunks=retrieved, top_k=rerank_top_k)
    context = format_context(chunks=final_chunks, max_chars=settings.max_context_chars)
    user_prompt = f"""
Question:
{question}

Context:
{context}

Answer:
"""
    llm = get_llm(provider=llm_provider, model=llm_model)
    answer = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, temperature=temperature)
    return answer, final_chunks
