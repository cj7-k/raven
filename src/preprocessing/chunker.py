"""
Splits cleaned content into retrieval-friendly chunks.
TODO: Use LangChain's RecursiveCharacterTextSplitter (or a semantic splitter)
to break text into overlapping chunks sized for the embedding model.
"""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Return a list of text chunks."""
    raise NotImplementedError("Implement chunking logic here.")
