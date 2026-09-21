"""Prompt templates for grounded answer generation."""

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say you don't know — do not make things up.
"""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer using only the context above."""
