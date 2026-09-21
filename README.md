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

## Team Ownership

| Area | Owner |
|---|---|
| RAG core (retriever, generator, prompt design) + system integration | Leader |
| Scraper & ingestion (`src/scraper`) | Member 1 |
| Preprocessing & vector store (`src/preprocessing`, `src/embeddings`) | Member 2 |
| API, frontend & deployment (`src/api`, `frontend/`, Docker/Redis) | Member 3 |

## Roadmap

- Week 1-2: Requirement study, literature survey, tool finalization
- Week 3-4: System architecture design; recursive scraper module
- Week 5-6: Chunking, embeddings & vector store integration
- Week 7-8: RAG pipeline + LLM answer generation
- Week 9-10: Chat interface; backend/frontend integration
- Week 11-12: Testing, latency optimization, documentation, final review

## Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Scraping | Scrapy / Playwright / BeautifulSoup |
| Chunking & Orchestration | LangChain |
| Vector Store | Chroma (dev) -> FAISS / Pinecone (prod) |
| LLM / Embeddings | OpenAI / Anthropic |
| Backend | FastAPI |
| Frontend | Streamlit / React |
| Deployment & Caching | Docker, Redis |
