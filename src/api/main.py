"""FastAPI app entrypoint."""
from fastapi import FastAPI
from src.api.routes import chat, ingest

app = FastAPI(title="RAVEN - RAG-Powered Website Chatbot")

app.include_router(ingest.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
