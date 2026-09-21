"""
Combines retrieved context with an LLM call to produce a grounded answer.
TODO: Call OpenAI/Anthropic chat completion with RAG_SYSTEM_PROMPT + retrieved context.
"""
from src.rag.retriever import retrieve
from src.rag.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


def generate_answer(question: str, top_k: int = 5) -> str:
    """Retrieve context for the question and generate a grounded answer."""
    chunks = retrieve(question, top_k=top_k)
    context = "\n\n".join(c.get("text", "") for c in chunks)
    prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
    raise NotImplementedError("Call your LLM API here with RAG_SYSTEM_PROMPT + prompt.")
