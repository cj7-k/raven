"""
src/preprocessing/preprocess.py

Takes the raw scraped pages (data/raw/scraped_pages.json) and:
  1. Cleans text further (removes boilerplate patterns, extra whitespace)
  2. Deduplicates near-identical pages/content
  3. Splits each page into overlapping chunks suitable for embedding
  4. Saves the result to data/processed/chunks.json, ready for the
     embeddings + vector store step

Usage:
    python src/preprocessing/preprocess.py
    python src/preprocessing/preprocess.py --input data/raw/scraped_pages.json --chunk-size 500 --chunk-overlap 50
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older langchain versions expose it here instead
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/raw/scraped_pages.json"
DEFAULT_OUTPUT = "data/processed/chunks.json"
DEFAULT_CHUNK_SIZE = 500       # characters per chunk (not tokens, kept simple)
DEFAULT_CHUNK_OVERLAP = 50     # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 30          # discard tiny/near-empty chunks (nav leftovers etc.)

# Common boilerplate lines/phrases to strip out (case-insensitive, whole line match)
BOILERPLATE_PATTERNS = [
    r"^cookie(s)? (policy|settings|notice)$",
    r"^accept all cookies$",
    r"^all rights reserved\.?$",
    r"^©\s?\d{4}.*$",
    r"^skip to (main )?content$",
    r"^subscribe to (our )?newsletter$",
]

_BOILERPLATE_RE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Extra cleanup pass on top of what the scraper already did."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _BOILERPLATE_RE):
            continue
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    # Collapse repeated whitespace/newlines left over after removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def content_hash(text: str) -> str:
    """Hash of normalized text, used to detect exact/near-duplicate pages."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_pages(pages: list) -> list:
    """Remove pages with identical (or whitespace-only-different) content."""
    seen_hashes = set()
    unique_pages = []

    for page in pages:
        h = content_hash(page["text"])
        if h in seen_hashes:
            print(f"  -> skipping duplicate content: {page['url']}")
            continue
        seen_hashes.add(h)
        unique_pages.append(page)

    return unique_pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        if len(text) < MIN_CHUNK_LENGTH:
            continue

        splits = splitter.split_text(text)

        for split in splits:
            split = split.strip()
            if len(split) < MIN_CHUNK_LENGTH:
                continue

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": split,
                    "source_url": page["url"],
                    "source_title": page.get("title", ""),
                    "depth": page.get("depth", None),
                }
            )
            chunk_id += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Run the scraper first (Day 1 module)."
        )

    with open(input_file, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    print(f"Loaded {len(raw_pages)} raw pages from {input_path}")

    # 1. Clean
    for page in raw_pages:
        page["text"] = clean_text(page["text"])

    # 2. Deduplicate
    unique_pages = deduplicate_pages(raw_pages)
    print(f"{len(unique_pages)} pages remain after deduplication")

    # 3. Chunk
    chunks = chunk_pages(unique_pages, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")

    # 4. Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to {output_path}")
    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean, dedupe, and chunk scraped pages for RAVEN")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to scraped_pages.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save chunks.json")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Character overlap between chunks")
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()