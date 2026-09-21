"""
Vector store wrapper (Chroma by default; swap for FAISS/Pinecone later).
TODO: Implement add() and query() against the chosen vector DB.
"""
import os

VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")


class VectorStore:
    def __init__(self):
        self.backend = VECTOR_STORE
        # TODO: initialize Chroma/FAISS/Pinecone client here

    def add(self, chunks: list[str], embeddings: list[list[float]], metadata: list[dict] | None = None):
        raise NotImplementedError

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        raise NotImplementedError
