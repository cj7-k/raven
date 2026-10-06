"""
src/api/main.py

GraphQL backend for RAVEN, built with Strawberry on top of FastAPI.

Exposes a single GraphQL endpoint (default: /graphql) with:
    Query.health           - liveness check
    Query.collectionInfo   - how many chunks are indexed
    Query.chat(question, topK) - ask a question, get a grounded answer + sources

Run locally:
    uvicorn src.api.main:app --reload

Then visit http://localhost:8000/graphql for the interactive GraphiQL
playground (lets you write and run queries from the browser, similar to
FastAPI's /docs for REST).

Example query to run in GraphiQL:
    query {
      chat(question: "How does sunscreen protect skin?") {
        answer
        sources {
          title
          url
        }
      }
    }
"""

import sys
from pathlib import Path
from typing import List, Optional

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow importing from src/
from src.rag.rag import generate_answer, DEFAULT_TOP_K
from src.embeddings.embed import get_collection, DEFAULT_DB_PATH, default_collection_name, DEFAULT_PROVIDER


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


@strawberry.type
class CollectionInfo:
    collection: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Shared logic (kept separate from the resolvers so it's easy to unit test)
# ---------------------------------------------------------------------------

NO_INFO_PHRASES = [
    "does not include",
    "don't have any relevant information",
    "doesn't contain",
    "cannot answer",
]


def run_chat(question: str, top_k: int = DEFAULT_TOP_K) -> ChatResult:
    result = generate_answer(question, top_k=top_k)

    answer_lower = result["answer"].lower()
    hide_sources = any(phrase in answer_lower for phrase in NO_INFO_PHRASES)
    sources = [] if hide_sources else result["sources"]

    return ChatResult(
        answer=result["answer"],
        sources=[Source(title=s.get("title"), url=s.get("url")) for s in sources],
    )


def run_collection_info() -> CollectionInfo:
    collection_name = default_collection_name(DEFAULT_PROVIDER)
    collection = get_collection(DEFAULT_DB_PATH, collection_name)
    return CollectionInfo(collection=collection_name, chunk_count=collection.count())


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