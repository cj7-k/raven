"""
src/embeddings/embed.py

Takes the processed chunks (data/processed/chunks.json), turns each chunk
into an embedding vector, and stores everything in a local, persistent
ChromaDB collection (data/chroma_db).

Two embedding providers are supported:
  - local  (default): free, runs on your own machine using the small
                      all-MiniLM-L6-v2 model that ships with ChromaDB.
                      No API key, no cost. Downloads ~80 MB once.
  - openai          : text-embedding-3-small. Needs OPENAI_API_KEY in .env
                      and a paid OpenAI account.

IMPORTANT: use the same provider for indexing and for searching. Each
provider gets its own collection (raven_local / raven_openai) so their
vectors can never be mixed by accident.

Usage:
    # 1. Build the vector store from chunks.json (free, local)
    python src/embeddings/embed.py

    # 2. Rebuild from scratch
    python src/embeddings/embed.py --reset

    # 3. Test retrieval with a question
    python src/embeddings/embed.py --query "How does sunscreen protect skin?"

    # Optional: use OpenAI instead
    python src/embeddings/embed.py --provider openai
"""

import argparse
import json
import os
import time
from pathlib import Path

import chromadb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_PROVIDER = "local"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100      # chunks embedded per batch
DEFAULT_TOP_K = 4           # results returned per query
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

_local_embedder = None


def embed_texts_local(client, texts: list) -> list:
    """Free local embeddings (all-MiniLM-L6-v2 via ONNX, bundled with ChromaDB).

    `client` is unused - it's only here so every provider has the same signature.
    The model (~80 MB) is downloaded automatically the first time this runs.
    If your connection is unstable, see the README note about downloading
    the model file manually through your browser instead.
    """
    global _local_embedder
    if _local_embedder is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        print("Loading local embedding model...")
        _local_embedder = DefaultEmbeddingFunction()

    vectors = _local_embedder(texts)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def get_openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file, "
            "or use the free local provider (the default)."
        )
    return OpenAI(api_key=api_key)


def embed_texts_openai(client, texts: list) -> list:
    """Embed with OpenAI. Returns one vector per string, in order."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  -> embedding request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)


def get_embedder(provider: str):
    """Return (client, embed_fn) for the chosen provider."""
    if provider == "local":
        return None, embed_texts_local
    if provider == "openai":
        return get_openai_client(), embed_texts_openai
    raise ValueError(f"Unknown provider '{provider}'. Use 'local' or 'openai'.")


def default_collection_name(provider: str) -> str:
    return f"raven_{provider}"


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def get_collection(db_path: str, collection_name: str, reset: bool = False):
    Path(db_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass  # collection didn't exist yet, nothing to delete

    # cosine distance is the standard choice for text embeddings
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_metadata(chunk: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool - never None."""
    return {
        "source_url": chunk.get("source_url", ""),
        "source_title": chunk.get("source_title", ""),
        "depth": chunk.get("depth") if chunk.get("depth") is not None else -1,
        "chunk_id": chunk["chunk_id"],
    }


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunks(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    db_path: str = DEFAULT_DB_PATH,
    provider: str = DEFAULT_PROVIDER,
    collection_name: str = None,
    reset: bool = False,
    client=None,
    embed_fn=None,
) -> int:
    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Could not find {chunks_path}. Run the preprocessing step first."
        )

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in the file - nothing to embed.")
        return 0

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    if embed_fn is None:
        client, embed_fn = get_embedder(provider)
    collection_name = collection_name or default_collection_name(provider)
    collection = get_collection(db_path, collection_name, reset=reset)

    total = 0
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = embed_fn(client, texts)

        # upsert = safe to re-run: existing chunk IDs are overwritten, not duplicated
        collection.upsert(
            ids=[f"chunk_{c['chunk_id']}" for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[build_metadata(c) for c in batch],
        )
        total += len(batch)
        print(f"  embedded and stored {total}/{len(chunks)} chunks")

    print(f"\nDone. Collection '{collection_name}' now holds {collection.count()} chunks.")
    print(f"Vector store saved at: {db_path}")
    return total


# ---------------------------------------------------------------------------
# Retrieval (used by the RAG step later)
# ---------------------------------------------------------------------------

def search(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: str = DEFAULT_DB_PATH,
    provider: str = DEFAULT_PROVIDER,
    collection_name: str = None,
    client=None,
    embed_fn=None,
) -> list:
    """Return the top_k chunks most similar in meaning to the question."""
    if embed_fn is None:
        client, embed_fn = get_embedder(provider)
    collection_name = collection_name or default_collection_name(provider)

    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(collection_name)

    query_embedding = embed_fn(client, [question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source_url": meta.get("source_url"),
                "source_title": meta.get("source_title"),
                "distance": dist,  # lower = more similar (cosine distance)
            }
        )
    return hits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB for RAVEN")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH, help="Path to chunks.json")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Where ChromaDB stores its data")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["local", "openai"],
                        help="Embedding provider (default: local, free)")
    parser.add_argument("--collection", default=None, help="Chroma collection name (default: raven_<provider>)")
    parser.add_argument("--reset", action="store_true", help="Delete the collection before rebuilding it")
    parser.add_argument("--query", help="Run a test search instead of indexing")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Results to return for --query")
    args = parser.parse_args()

    if args.query:
        hits = search(args.query, top_k=args.top_k, db_path=args.db_path,
                      provider=args.provider, collection_name=args.collection)
        print(f"\nTop {len(hits)} results for: {args.query!r}\n")
        for i, hit in enumerate(hits, start=1):
            preview = hit["text"][:200].replace("\n", " ")
            print(f"{i}. [{hit['distance']:.3f}] {hit['source_title']}")
            print(f"   {hit['source_url']}")
            print(f"   {preview}...\n")
    else:
        index_chunks(
            chunks_path=args.chunks,
            db_path=args.db_path,
            provider=args.provider,
            collection_name=args.collection,
            reset=args.reset,
        )


if __name__ == "__main__":
    main()