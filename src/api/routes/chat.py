"""
/chat endpoint: takes a question, retrieves context, returns a grounded answer.
"""
from fastapi import APIRouter
from src.api.schemas import ChatRequest, ChatResponse
from src.rag.generator import generate_answer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    answer = generate_answer(payload.question, top_k=payload.top_k)
    return ChatResponse(answer=answer, sources=[])
