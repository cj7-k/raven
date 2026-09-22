# RAVEN — Retrieval-Augmented Vectorized Engine for websites

A RAG-powered chatbot that ingests any given website URL (via recursive crawling)
and answers questions grounded in that site's content — reducing hallucination
and improving response latency vs. plain LLM chat or keyword search.

## Architecture

```
Input URL -> Recursive Scraper -> Clean & Chunk -> Embed & Vector Store -> Retriever -> LLM (RAG) -> Chat Response
```

## Project Structure

```
src/
  scraper/        # recursive crawling + HTML/text/table extraction
  preprocessing/  # cleaning, deduplication, chunking
  embeddings/     # embedding generation + vector store wrapper
  rag/            # retrieval + grounded answer generation
  api/            # FastAPI app (/ingest, /chat)
  cache/          # Redis client for low-latency serving
frontend/         # React or Streamlit chat UI
data/             # raw/processed content + vector index (gitignored)
scripts/          # CLI utilities (run_ingest.py, benchmark_latency.py)
tests/            # pytest test suite
```

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```
5. Run the API locally:
   ```
   uvicorn src.api.main:app --reload
   ```
   Visit `http://localhost:8000/docs` for the interactive API docs.

## Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Scraping | Playwright |
| Chunking & Orchestration | LangChain |
| Vector Store | ChromaDB |
| LLM / Embeddings | OpenAI |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment & Caching | Docker + Redis |
