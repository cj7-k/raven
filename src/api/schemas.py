"""Pydantic request/response models for the API."""
from pydantic import BaseModel


class IngestRequest(BaseModel):
    url: str
    depth: int | None = None


class IngestResponse(BaseModel):
    status: str
    pages_discovered: int = 0


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
