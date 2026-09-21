"""
CLI to run the full pipeline end-to-end for a single URL:
scrape -> clean -> chunk -> embed -> index.

Usage:
    python scripts/run_ingest.py https://example.com
"""
import sys
from src.scraper.crawler import crawl
from src.scraper.extractor import extract_content
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunker import chunk_text
from src.embeddings.embedder import embed_chunks
from src.embeddings.vector_store import VectorStore


def main(seed_url: str):
    print(f"Crawling from: {seed_url}")
    urls = crawl(seed_url)
    print(f"Discovered {len(urls)} pages. Pipeline not yet fully implemented.")
    # TODO: for each URL -> extract -> clean -> chunk -> embed -> store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ingest.py <seed_url>")
        sys.exit(1)
    main(sys.argv[1])
