"""
src/rag/rag.py

The core RAG (Retrieval-Augmented Generation) pipeline for RAVEN.

Given a question:
  1. Retrieve the most relevant chunks from ChromaDB (via src/embeddings/embed.py)
  2. Build a prompt that grounds the LLM in that retrieved content
  3. Call Groq (openai/gpt-oss-120b) to generate an answer
  4. Return the answer along with the sources it was based on

Requires GROQ_API_KEY in your .env file. Get a free key at
https://console.groq.com/keys (no credit card needed).

Usage:
    # One-off question
    python src/rag/rag.py --question "How does sunscreen protect skin?"

    # Interactive chat loop
    python src/rag/rag.py
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow importing from src/
from src.embeddings.embed import search as search_chunks, DEFAULT_PROVIDER as DEFAULT_EMBED_PROVIDER


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TOP_K = 4
MAX_CONTEXT_CHARS_PER_CHUNK = 800  # keep prompts a reasonable size

SYSTEM_PROMPT = """You are RAVEN, a helpful assistant that answers questions \
using ONLY the information provided in the context below. Follow these rules:

1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say so clearly instead of guessing or making something up.
3. Be clear and concise. Write in plain, natural language.
4. Do not mention "the context" or "the provided chunks" in your answer - just \
answer naturally, as if you simply know this information.
"""


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def get_groq_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Get a free key at https://console.groq.com/keys "
            "and add it to your .env file as GROQ_API_KEY=..."
        )
    from groq import Groq
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Retrieval + prompt building
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = DEFAULT_TOP_K, embed_provider: str = DEFAULT_EMBED_PROVIDER) -> list:
    """Fetch the most relevant chunks for a question from the vector store."""
    return search_chunks(question, top_k=top_k, provider=embed_provider)


def build_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:MAX_CONTEXT_CHARS_PER_CHUNK]
        title = chunk.get("source_title") or "Unknown source"
        context_blocks.append(f"[Source {i}: {title}]\n{text}")

    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    embed_provider: str = DEFAULT_EMBED_PROVIDER,
    client=None,
    chunks: list = None,
) -> dict:
    """Run the full RAG pipeline for one question. Returns answer + sources."""
    if chunks is None:
        chunks = retrieve(question, top_k=top_k, embed_provider=embed_provider)

    if not chunks:
        return {
            "answer": "I don't have any relevant information to answer that question.",
            "sources": [],
        }

    user_prompt = build_prompt(question, chunks)

    if client is None:
        client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    answer = response.choices[0].message.content

    # de-duplicate sources (multiple chunks can come from the same page)
    seen = set()
    sources = []
    for chunk in chunks:
        url = chunk.get("source_url")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": chunk.get("source_title"), "url": url})

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_result(result: dict):
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['title']}: {s['url']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RAVEN RAG pipeline")
    parser.add_argument("--question", help="Ask a single question and exit")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to retrieve")
    parser.add_argument("--embed-provider", default=DEFAULT_EMBED_PROVIDER, choices=["local", "openai"])
    args = parser.parse_args()

    if args.question:
        result = generate_answer(args.question, top_k=args.top_k, embed_provider=args.embed_provider)
        print_result(result)
        return

    print("RAVEN chat - ask a question (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = generate_answer(question, top_k=args.top_k, embed_provider=args.embed_provider)
        print_result(result)


if __name__ == "__main__":
    main()