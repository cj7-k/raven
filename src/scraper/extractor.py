"""
HTML/text/table extraction from a fetched page.
TODO: Use BeautifulSoup to pull clean text, tables, and metadata from raw HTML.
"""


def extract_content(html: str) -> dict:
    """Return {'text': ..., 'tables': [...], 'title': ...} from raw HTML."""
    raise NotImplementedError("Implement HTML extraction logic here.")
