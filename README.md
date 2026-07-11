<<<<<<< HEAD
# rag-app
=======
# Production RAG API - Local Setup

This ZIP contains a complete Retrieval Augmented Generation API built with FastAPI, Qdrant, SentenceTransformers, optional reranking, and multiple LLM backends.

It is designed to run locally from Visual Studio Code or Visual Studio terminal.

## Features

- Text and file ingestion
- PDF, DOCX, HTML, TXT, Markdown, CSV loading
- Chunking strategies:
  - fixed
  - recursive
  - sentence
  - markdown
  - semantic
  - parent-child
  - code
  - table
- SentenceTransformers or OpenAI embeddings
- Qdrant vector database
- CrossEncoder reranking
- OpenAI, Anthropic, Gemini, Ollama, or echo test LLM
- FastAPI endpoints
- Docker Compose setup

## Option A: Run with Docker Compose - recommended

### 1. Requirements

Install:

- Docker Desktop
- Visual Studio Code or Visual Studio

### 2. Open the project

Unzip this file, then open the `rag_app` folder in VS Code.

```bash
cd rag_app
```

### 3. Create your environment file

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 4. For no-key testing, edit `.env`

Set:

```env
LLM_PROVIDER=echo
ENABLE_RERANKING=false
```

This lets you test ingestion and retrieval without an OpenAI/Anthropic/Gemini key.

### 5. Start the services

```bash
docker compose up --build
```

API docs:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

## Option B: Run directly with Python

### 1. Start Qdrant only

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.15.1
```

### 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Run the API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

## Test ingestion

### Ingest the included sample text file

```bash
curl -X POST "http://localhost:8000/ingest/file?chunk_strategy=recursive&chunk_size=900&chunk_overlap=150" \
  -F "file=@sample_doc.txt"
```

Windows PowerShell equivalent:

```powershell
curl.exe -X POST "http://localhost:8000/ingest/file?chunk_strategy=recursive&chunk_size=900&chunk_overlap=150" -F "file=@sample_doc.txt"
```

## Test chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which chunking strategy is recommended for production?",
    "top_k": 8,
    "rerank_top_k": 4,
    "llm_provider": "echo"
  }'
```

Windows PowerShell equivalent:

```powershell
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"Which chunking strategy is recommended for production?\",\"top_k\":8,\"rerank_top_k\":4,\"llm_provider\":\"echo\"}"
```

## Use OpenAI

Edit `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_key_here
```

Optional OpenAI embeddings:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_key_here
```

Then restart the API.

## Use Anthropic Claude

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=your_key_here
```

## Use Google Gemini

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_key_here
```

## Use local Ollama

Start the main stack plus Ollama:

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
```

Pull a model:

```bash
docker exec -it ollama ollama pull llama3.1
```

Edit `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1
```

## API endpoints

### GET `/health`

Returns API status.

### POST `/ingest`

Ingest JSON text documents.

### POST `/ingest/file`

Upload and ingest a file.

Supported extensions:

```text
.txt, .md, .csv, .json, .py, .js, .ts, .pdf, .docx, .html, .htm
```

### POST `/chat`

Ask a question over indexed documents.

## Recommended production defaults

```env
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_RERANKING=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
```

For higher quality embeddings, use OpenAI embeddings or a larger local embedding model.

## Notes

- The first run may take time because SentenceTransformers downloads models.
- For no-key local testing, use `LLM_PROVIDER=echo`.
- For production, restrict CORS in `app/api.py`.
- For large file ingestion, move ingestion to a background worker such as Celery or RQ.

---

## Interactive dashboard

This ZIP includes a Streamlit dashboard for playing with the full RAG pipeline.

Run everything with Docker:

```bash
cp .env.example .env

docker compose up --build
```

Open:

```text
http://localhost:8501
```

The dashboard lets you test:

- Loader: paste text or upload files.
- Cleaner: preview normalized text.
- Chunker: compare fixed, recursive, sentence, markdown, semantic, parent_child, code, and table strategies.
- Embedding Model: inspect embedding dimension and vector previews.
- Qdrant: view raw retrieved chunks.
- Reranker: compare retrieved chunks versus reranked chunks.
- Prompt Builder: inspect the exact context and prompt sent to the LLM.
- LLM Provider: run with `echo`, OpenAI, Anthropic, Gemini, or Ollama.
- Answer + Sources: view final answer and cited source chunks.

Dashboard URL:

```text
http://localhost:8501
```

API Swagger URL:

```text
http://localhost:8000/docs
```

For detailed instructions, read:

```text
DASHBOARD_GUIDE.md
```

### No-key first test

The default `.env.example` uses:

```env
LLM_PROVIDER=echo
LLM_MODEL=echo
```

This lets you test retrieval and prompt construction without API keys. For real answers, change the LLM provider and add the correct key.
>>>>>>> b598466 (initial rag implemetnation)
