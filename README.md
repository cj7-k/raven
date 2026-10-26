# RAVEN: Retrieval-Augmented Vectorized Engine for websites

A RAG chatbot that crawls a website, indexes its content by meaning, and answers
questions **only from what it retrieved**, with the source pages shown under every
answer. Repeat questions are served from a Redis cache in about 0 ms, and the whole
app runs with one Docker command.

The demo knowledge base is Wikipedia's *Skin care* topic (20 pages, 898 chunks), but
the pipeline works on any website you point it at.

## How it works

```
Website URL -> Scraper -> Clean & chunk -> Embed & store -> Retriever -> LLM -> Answer + sources
 (Playwright)  (LangChain)    (MiniLM + ChromaDB)   (top-4 by meaning)  (Groq)
                                        served through a GraphQL API (FastAPI + Strawberry),
                                        shown in a Streamlit chat UI, cached in Redis
```

| Stage | What it does |
|---|---|
| Scraper | Breadth-first crawl of same-domain links with a headless browser (renders JavaScript pages), strips navigation, footers and citation markup |
| Preprocessing | Removes boilerplate and duplicate pages, then splits text into ~500-character chunks (~125 tokens) with 50 characters of overlap |
| Embeddings | Turns each chunk into a 384-number vector with a free local model (all-MiniLM-L6-v2) and stores it in ChromaDB |
| RAG | Finds the 4 closest chunks, builds a prompt that says "answer only from this context", and asks Groq (`openai/gpt-oss-120b`) |
| API | One GraphQL endpoint, `/graphql`, with `chat`, `collectionInfo`, `cacheInfo` and `health` queries |
| Cache | Redis, with an automatic in-memory fallback if Redis isn't running |
| UI | Streamlit chat with collapsible sources, per-answer timing and cache statistics |

## Project structure

```
src/
  scraper/scraper.py          recursive crawler (Playwright + BeautifulSoup)
  preprocessing/preprocess.py cleaning, de-duplication, chunking
  embeddings/embed.py         embeddings + ChromaDB (index and search)
  rag/rag.py                  retrieval + grounded answer generation (Groq)
  api/main.py                 GraphQL API (FastAPI + Strawberry)
  cache/cache.py              answer cache (Redis, in-memory fallback)
frontend/app.py               Streamlit chat UI
data/                         raw pages, chunks, vector store   (gitignored)
models/                       embedding model for Docker        (gitignored)
Dockerfile, docker-compose.yml, requirements-docker.txt, .dockerignore
requirements.txt              full local environment (includes the scraping tools)
.env.example                  copy to .env and add your key
```

## Quick start with Docker

**Prerequisites:** Docker Desktop (running), a free [Groq API key](https://console.groq.com/keys),
and `data/processed/chunks.json` (created by steps 1 and 2 of the data pipeline below).

1. **Add your key.** Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
2. **Put the embedding model in the project** so the container never downloads it.
   Once you have run `embed.py` locally, the model sits in your home folder:
   ```
   # Windows
   xcopy "%USERPROFILE%\.cache\chroma\onnx_models\all-MiniLM-L6-v2" "models\all-MiniLM-L6-v2" /E /I /Y
   # macOS / Linux
   mkdir -p models && cp -r ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2 models/
   ```
3. **Build the image:**
   ```
   docker compose build
   ```
4. **Build the search index inside Docker (first time only):**
   ```
   docker compose run --rm backend python src/embeddings/embed.py
   ```
   Expect `Done. Collection 'raven_local' now holds 898 chunks.`
5. **Start everything:**
   ```
   docker compose up -d
   ```
   Open **http://localhost:8501** (chat UI). The GraphQL playground is at
   http://localhost:8000/graphql. The sidebar should say `Backend: redis`.

Stop with `docker compose down` (the index is kept) or `docker compose down -v` (deletes the index too).

## Data pipeline (run locally)

Create a virtual environment and install the full dependencies:

```
python -m venv venv
venv\Scripts\activate            # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

1. **Scrape** a site into `data/raw/scraped_pages.json`:
   ```
   python src/scraper/scraper.py https://en.wikipedia.org/wiki/Skin_care --max-depth 1 --max-pages 20
   ```
2. **Clean and chunk** into `data/processed/chunks.json`:
   ```
   python src/preprocessing/preprocess.py
   ```
3. **Embed and index** into ChromaDB (`--reset` rebuilds from scratch):
   ```
   python src/embeddings/embed.py
   python src/embeddings/embed.py --query "How does sunscreen protect skin?"
   ```
4. **Ask a question** from the command line:
   ```
   python src/rag/rag.py --question "What causes eczema?"
   ```

**Run without Docker** (two terminals, both with the venv activated):

```
uvicorn src.api.main:app --reload          # terminal 1: backend
streamlit run frontend/app.py              # terminal 2: chat UI at http://localhost:8501
```

**After scraping new data in Docker,** rebuild the index and clear old cached answers:

```
docker compose run --rm backend python src/embeddings/embed.py --reset
docker compose restart backend
docker compose exec backend python src/cache/cache.py --clear
```

## GraphQL API

```graphql
query {
  chat(question: "What causes eczema?") {
    answer
    cached        # true when served from the cache
    latencyMs
    sources { title url }
  }
}
```

Other queries: `health`, `collectionInfo { collection chunkCount }`,
`cacheInfo { backend hits misses entries ttlSeconds }`.

If the answer isn't in the retrieved text, the model says so instead of guessing, and no sources are shown.

## Answer cache

- Questions are normalised first, so "What causes eczema?" and "what causes  eczema" share one entry.
- Entries expire after one hour (`CACHE_TTL_SECONDS`).
- Failed answers are never cached.
- If Redis is down, requests keep working using an in-memory cache.
- Inspect or clear it: `python src/cache/cache.py --info` / `--clear`.

## Configuration

| Variable | Used by | Purpose |
|---|---|---|
| `GROQ_API_KEY` | backend | Groq API key (required) |
| `REDIS_URL` | backend | Redis address, default `redis://localhost:6379/0` (set automatically by Docker) |
| `CACHE_TTL_SECONDS` | backend | Cache lifetime, default 3600 |
| `RAVEN_API_URL` | frontend | GraphQL address, default `http://localhost:8000/graphql` (set automatically by Docker) |

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python |
| Scraping | Playwright + BeautifulSoup |
| Chunking | LangChain text splitter |
| Vector store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2, local and free (runs through ChromaDB's ONNX runtime) |
| LLM | Groq, `openai/gpt-oss-120b` |
| Backend | FastAPI + GraphQL (Strawberry) |
| Frontend | Streamlit |
| Cache | Redis, with in-memory fallback |
| Deployment | Docker Compose (Redis + backend + frontend) |

## Troubleshooting

- **Sidebar shows 0 indexed chunks:** the index hasn't been built. Run the embed step (Docker quick start, step 4).
- **"Connection error" when asking a question:** the backend can't reach Groq. Check your internet, and on campus or café WiFi check that the network sign-in page hasn't expired.
- **`pip install` fails on Windows with Python 3.12 or 3.13:** `chroma-hnswlib` and `numpy 1.26.4` have no prebuilt Windows packages for those versions, so pip tries to compile them and needs the Visual Studio C++ build tools. Use **Python 3.11** (checked: every runtime package has a ready-made Windows install there).
- **Build context is huge or the build is very slow:** make sure the file is named `.dockerignore` with the leading dot. Windows sometimes drops it.
- **`Failed to send telemetry event ... capture()` warnings:** caused by a newer `posthog` package. `pip install "posthog==5.4.0"` removes them (already pinned in `requirements-docker.txt`).
- **Embedding model download times out:** download `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz` in a browser and place it in `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`.

## Status and next steps

Done: scraper, preprocessing, embeddings, RAG pipeline, GraphQL API, Streamlit UI, answer cache, Docker deployment.

Planned: a pytest suite, a latency benchmark script, and ingesting any URL from the UI.
