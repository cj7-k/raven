"""
Generates vector embeddings for text chunks.
TODO: Call OpenAI (or another) embeddings API on a list of chunks.
"""
import os


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors, one per chunk."""
    raise NotImplementedError("Implement embedding generation here.")
