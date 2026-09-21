"""
Recursive URL crawler.
TODO: Implement recursive discovery of linked pages starting from a seed URL,
respecting robots.txt and staying within CRAWL_DOMAIN_WHITELIST / CRAWL_DEPTH.
"""
from src.scraper.config import CRAWL_DEPTH, CRAWL_DOMAIN_WHITELIST


def crawl(seed_url: str, depth: int = CRAWL_DEPTH) -> list[str]:
    """Return a list of discovered URLs starting from seed_url."""
    raise NotImplementedError("Implement recursive crawling logic here.")
