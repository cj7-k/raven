"""Scraper configuration: crawl depth, domain boundaries."""
import os
from dotenv import load_dotenv

load_dotenv()

CRAWL_DEPTH = int(os.getenv("CRAWL_DEPTH", 3))
CRAWL_DOMAIN_WHITELIST = os.getenv("CRAWL_DOMAIN_WHITELIST", "").split(",") if os.getenv("CRAWL_DOMAIN_WHITELIST") else []
