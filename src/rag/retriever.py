"""
Top-k retrieval logic: given a user query, fetch the most relevant chunks.
"""
from src.embeddings.vector_store import VectorStore
from src.embeddings.embedder import embed_chunks


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and return top_k matching chunks from the vector store."""
    raise NotImplementedError("Implement retrieval logic here.")
