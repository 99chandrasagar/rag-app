# Interactive RAG Dashboard Guide

This project includes a Streamlit dashboard that lets you play with every stage of the RAG pipeline.

## URLs

After running Docker Compose:

- FastAPI Swagger: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Qdrant: http://localhost:6333

## Fast start

```bash
cp .env.example .env

docker compose up --build
```

Then open:

```text
http://localhost:8501
```

## Dashboard tabs

### 1. Pipeline Map

Explains the two main flows:

```text
/ingest
  Loader
  Cleaner
  Chunker
  Embedding Model
  Qdrant

/chat
  Embed Query
  Qdrant Search
  Optional Reranker
  Prompt Builder
  LLM Provider
  Answer + Sources
```

Use this tab as your mental model.

### 2. Ingest Lab

Use this to test ingestion.

You can:

- Paste text.
- Upload PDF, DOCX, TXT, MD, HTML, CSV, or code files.
- Preview cleaned text.
- Preview chunks before indexing.
- Ingest into Qdrant.

Recommended first test:

1. Keep the sample text.
2. Choose `recursive`.
3. Click **Preview chunks**.
4. Click **Ingest into Qdrant**.

### 3. Retrieval Lab

Use this to inspect the retrieval pipeline without hiding the intermediate steps.

You can:

- Embed a query.
- See the query vector dimension and preview values.
- Search Qdrant.
- Compare raw retrieved chunks and reranked chunks.
- View the exact context and prompt that would be sent to the LLM.

Try:

```text
What is the refund policy?
```

### 4. Chat Lab

This calls the real `/chat` endpoint.

Use `echo` first. It does not require an API key and shows that retrieval and prompt construction worked.

To use a real model, update `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_key
```

Then restart Docker Compose.

### 5. Chunk Strategy Compare

Use this to compare all chunking strategies on the same text before ingestion.

Strategies:

| Strategy | Best for | Notes |
|---|---|---|
| fixed | Logs, simple flat text | Simple baseline, can cut meaning mid-sentence |
| recursive | Most PDFs, DOCX, HTML, plain text | Best default |
| sentence | FAQs, policy text, support articles | Preserves sentence boundaries |
| markdown | Markdown docs with headings | Keeps heading path metadata |
| semantic | Research prose, dense text | Slower because it embeds sentences |
| parent_child | Long legal, compliance, architecture docs | Retrieves small child chunks but stores larger parent context |
| code | Python/JS/TS-like source files | Tries to preserve functions and classes |
| table | CSV or markdown-table-like text | Keeps headers and rows together |

### 6. Config

Shows the active runtime configuration:

- Qdrant collection
- Embedding provider and model
- Reranker status
- Default LLM provider and model

## New debug endpoints

The dashboard uses these endpoints:

### GET /options

Returns available chunking strategies, LLM providers, and current config.

### POST /debug/chunk

Preview chunking without saving to Qdrant.

Sample:

```json
{
  "text": "Your document text",
  "chunk_strategy": "recursive",
  "chunk_size": 900,
  "chunk_overlap": 150
}
```

### POST /debug/retrieve

Runs query embedding and Qdrant search only.

Sample:

```json
{
  "question": "What is the refund policy?",
  "top_k": 8,
  "filters": null
}
```

### POST /debug/pipeline

Runs the full explainable retrieval pipeline. You can choose whether to call the LLM.

Sample:

```json
{
  "question": "What is the refund policy?",
  "top_k": 8,
  "rerank_top_k": 4,
  "filters": null,
  "llm_provider": "echo",
  "llm_model": "echo",
  "temperature": 0.1,
  "run_llm": true
}
```

## Running without Docker

Terminal 1: start Qdrant.

```bash
docker compose up qdrant
```

Terminal 2: start FastAPI.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Terminal 3: start dashboard.

```bash
.venv\Scripts\activate
set API_BASE_URL=http://localhost:8000
streamlit run dashboard.py --server.port 8501
```

PowerShell users can run:

```powershell
.\run_dashboard.ps1
```
