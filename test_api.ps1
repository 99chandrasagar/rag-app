curl.exe http://localhost:8000/health
curl.exe -X POST "http://localhost:8000/ingest/file?chunk_strategy=recursive&chunk_size=900&chunk_overlap=150" -F "file=@sample_doc.txt"
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"Which chunking strategy is recommended for production?\",\"top_k\":8,\"rerank_top_k\":4,\"llm_provider\":\"echo\"}"
