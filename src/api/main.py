"""
src/api/main.py

GraphQL backend for RAVEN, built with Strawberry on top of FastAPI.

Exposes a single GraphQL endpoint (default: /graphql) with:
    Query.health           - liveness check
    Query.collectionInfo   - how many chunks are indexed
    Query.cacheInfo        - cache backend (redis/memory), hits, misses
    Query.chat(question, topK) - ask a question, get a grounded answer + sources

Repeated questions are served from the answer cache (src/cache/cache.py),
skipping retrieval and the LLM call entirely. Every chat response reports
whether it was cached and how long it took.

Run locally:
    uvicorn src.api.main:app --reload

Then visit http://localhost:8000/graphql for the interactive GraphiQL
playground.

Example query to run in GraphiQL:
    query {
      chat(question: "How does sunscreen protect skin?") {
        answer
        cached
        latencyMs
        sources {
          title
          url
        }
      }
    }
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow importing from src/
from src.rag.rag import generate_answer, DEFAULT_TOP_K
from src.embeddings.embed import get_collection, DEFAULT_DB_PATH, default_collection_name, DEFAULT_PROVIDER
from src.cache.cache import get_cache

cache = get_cache()   # connects to Redis if available, otherwise in-memory


# ---------------------------------------------------------------------------
# GraphQL types
# ---------------------------------------------------------------------------

@strawberry.type
class Source:
    title: Optional[str]
    url: Optional[str]


@strawberry.type
class ChatResult:
    answer: str
    sources: List[Source]
    cached: bool
    latency_ms: int


@strawberry.type
class CollectionInfo:
    collection: str
    chunk_count: int


@strawberry.type
class CacheInfo:
    backend: str
    hits: int
    misses: int
    entries: int
    ttl_seconds: int


# ---------------------------------------------------------------------------
# Shared logic (kept separate from the resolvers so it's easy to unit test)
# ---------------------------------------------------------------------------

NO_INFO_PHRASES = [
    "does not include",
    "don't have any relevant information",
    "doesn't contain",
    "cannot answer",
]


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def run_chat(question: str, top_k: int = DEFAULT_TOP_K) -> ChatResult:
    start = time.perf_counter()

    # 1. Cache hit: no retrieval, no LLM call
    cached = cache.get(question, top_k)
    if cached is not None:
        return ChatResult(
            answer=cached["answer"],
            sources=[Source(title=s.get("title"), url=s.get("url")) for s in cached["sources"]],
            cached=True,
            latency_ms=_elapsed_ms(start),
        )

    # 2. Cache miss: full RAG pipeline
    result = generate_answer(question, top_k=top_k)

    answer_lower = result["answer"].lower()
    hide_sources = any(phrase in answer_lower for phrase in NO_INFO_PHRASES)
    sources = [] if hide_sources else result["sources"]

    # Only successful answers reach this point (errors raise before it),
    # so failures are never cached.
    cache.set(question, top_k, {"answer": result["answer"], "sources": sources})

    return ChatResult(
        answer=result["answer"],
        sources=[Source(title=s.get("title"), url=s.get("url")) for s in sources],
        cached=False,
        latency_ms=_elapsed_ms(start),
    )


def run_collection_info() -> CollectionInfo:
    collection_name = default_collection_name(DEFAULT_PROVIDER)
    collection = get_collection(DEFAULT_DB_PATH, collection_name)
    return CollectionInfo(collection=collection_name, chunk_count=collection.count())


def run_cache_info() -> CacheInfo:
    info = cache.info()
    return CacheInfo(
        backend=info["backend"],
        hits=info["hits"],
        misses=info["misses"],
        entries=info["entries"],
        ttl_seconds=info["ttl_seconds"],
    )


# ---------------------------------------------------------------------------
# GraphQL schema
# ---------------------------------------------------------------------------

@strawberry.type
class Query:
    @strawberry.field(description="Simple liveness check")
    def health(self) -> str:
        return "ok"

    @strawberry.field(description="How many chunks are currently indexed")
    def collection_info(self) -> CollectionInfo:
        return run_collection_info()

    @strawberry.field(description="Answer cache status: backend, hits, misses, entries")
    def cache_info(self) -> CacheInfo:
        return run_cache_info()

    @strawberry.field(description="Ask a question and get a grounded answer with sources")
    def chat(self, question: str, top_k: int = DEFAULT_TOP_K) -> ChatResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not (1 <= top_k <= 20):
            raise ValueError("top_k must be between 1 and 20")
        return run_chat(question, top_k=top_k)


schema = strawberry.Schema(query=Query)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RAVEN API (GraphQL)")

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
