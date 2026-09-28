"""
src/scraper/scraper.py

Recursive same-domain website scraper for the RAVEN RAG chatbot.

Given a starting URL, this crawls all internal links up to a max depth
and max page count, extracts clean readable text (and simple tables)
from each page using Playwright (handles JS-rendered pages too), and
saves the results as JSON files under data/raw/.

Usage:
    python scraper.py https://example.com --max-depth 2 --max-pages 50
"""

import argparse
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_PAGES = 50
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_DELAY_SECONDS = 0.5  # be polite, avoid hammering the target site

# Tags whose text content is noise, not real page content
NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "svg", "form", "iframe"]

# File extensions to skip when following links (non-HTML resources)
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".zip", ".mp4", ".mp3", ".woff", ".woff2", ".ttf",
    ".xml", ".json", ".rss",
)

# Wikipedia (and similar wiki-style sites) namespace prefixes that are not
# real article content - talk pages, category indexes, admin pages, etc.
WIKI_NON_ARTICLE_PREFIXES = (
    "Talk:", "Category:", "Special:", "Wikipedia:", "Help:", "Portal:",
    "User:", "User_talk:", "File:", "Template:", "Template_talk:",
    "Module:", "MediaWiki:", "Draft:", "TimedText:", "Media:",
)


def is_wiki_utility_url(url: str) -> bool:
    """Detect Wikipedia-style utility/meta pages that aren't real article content."""
    parsed = urlparse(url)

    # URLs like /w/index.php?title=...&action=edit (edit, history, print, etc.)
    if "/w/index.php" in parsed.path:
        return True

    # Query-string actions on article pages, e.g. ?action=edit, ?action=info
    if re.search(r"[?&]action=", url):
        return True

    # Non-article namespaces, e.g. /wiki/Talk:Skin_care, /wiki/Category:Skin_care
    if "/wiki/" in parsed.path:
        page_title = parsed.path.split("/wiki/", 1)[-1]
        if any(page_title.startswith(prefix) for prefix in WIKI_NON_ARTICLE_PREFIXES):
            return True

    # Identifier/reference stub pages, e.g. S2CID_(identifier), ISBN_(identifier)
    if "_(identifier)" in url:
        return True

    return False


@dataclass
class ScrapedPage:
    url: str
    depth: int
    title: str
    text: str
    tables: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes so we don't revisit the same page twice."""
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="")
    path = normalized.path.rstrip("/") or "/"
    normalized = normalized._replace(path=path)
    return normalized.geturl()


def normalize_domain(netloc: str) -> str:
    """Treat 'www.example.com' and 'example.com' as the same domain."""
    return netloc.lower().removeprefix("www.")


def is_same_domain(url: str, base_domain: str) -> bool:
    return normalize_domain(urlparse(url).netloc) == normalize_domain(base_domain)


def should_skip_url(url: str) -> bool:
    lowered = url.lower()
    if any(lowered.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    if is_wiki_utility_url(url):
        return True
    return False


def extract_links(html: str, current_url: str, base_domain: str, debug: bool = False) -> set:
    soup = BeautifulSoup(html, "html.parser")
    all_anchor_tags = soup.find_all("a", href=True)
    links = set()

    if debug:
        print(f"     [debug] total <a href> tags on page: {len(all_anchor_tags)}")
        for a_tag in all_anchor_tags[:10]:
            print(f"     [debug] sample href: {a_tag['href']!r}")

    for a_tag in all_anchor_tags:
        href = a_tag["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(current_url, href)
        absolute = normalize_url(absolute)
        if is_same_domain(absolute, base_domain) and not should_skip_url(absolute):
            links.add(absolute)

    return links


def extract_clean_text(html: str) -> tuple:
    """Return (title, clean_text, tables) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Pull out simple tables as list-of-rows before stripping tags
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Prefer <main> or <article> if present, else fall back to <body>
    main_content = soup.find("main") or soup.find("article") or soup.body or soup

    text = main_content.get_text(separator="\n")
    # Collapse excess whitespace/blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    clean_text = "\n".join(lines)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return title, clean_text, tables


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------

def crawl(
    start_url: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_pages: int = DEFAULT_MAX_PAGES,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    headless: bool = True,
) -> list:
    start_url = normalize_url(start_url)
    base_domain = urlparse(start_url).netloc

    visited = set()
    queue = deque([(start_url, 0)])
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
        )
        # Hide the most common automation fingerprint that bot-detection
        # services (like Cloudflare) check for.
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        while queue and len(results) < max_pages:
            url, depth = queue.popleft()

            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            print(f"[depth {depth}] scraping: {url}")

            try:
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                # Give any late-loading content a brief moment, but don't
                # block indefinitely on networkidle (many real sites have
                # continuous background requests - ads, analytics, chat
                # widgets - that never let networkidle fire).
                page.wait_for_timeout(1500)

                # Some sites (e.g. behind Cloudflare) show a brief
                # "checking your browser" challenge page before the real
                # content loads. If we detect that, wait longer for it
                # to resolve automatically.
                page_title = page.title().lower()
                if any(marker in page_title for marker in ["just a moment", "checking your browser", "attention required"]):
                    print("  -> detected bot-check page, waiting for it to resolve...")
                    page.wait_for_timeout(6000)
            except PlaywrightTimeoutError:
                print(f"  -> timeout, skipping")
                continue
            except Exception as e:
                print(f"  -> error: {e}, skipping")
                continue

            html = page.content()
            title, clean_text, tables = extract_clean_text(html)

            if depth == 0:
                debug_html_path = Path("data/raw/_debug_first_page.html")
                debug_html_path.parent.mkdir(parents=True, exist_ok=True)
                debug_html_path.write_text(html, encoding="utf-8")
                print(f"  -> saved raw HTML for inspection: {debug_html_path}")

            if clean_text:  # skip pages with no real content
                results.append(
                    ScrapedPage(url=url, depth=depth, title=title, text=clean_text, tables=tables)
                )

            if depth < max_depth:
                links = extract_links(html, url, base_domain, debug=True)
                print(f"  -> found {len(links)} internal link(s)")
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))

            time.sleep(delay_seconds)

        browser.close()

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(pages: list, output_dir: str = "data/raw") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    combined = []
    for i, page in enumerate(pages):
        record = {
            "id": i,
            "url": page.url,
            "depth": page.depth,
            "title": page.title,
            "text": page.text,
            "tables": page.tables,
        }
        combined.append(record)

    combined_file = out_path / "scraped_pages.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(pages)} pages to {combined_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recursive website scraper for RAVEN")
    parser.add_argument("url", help="Starting URL to crawl")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Max link depth to follow")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max number of pages to scrape")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between requests (seconds)")
    parser.add_argument("--output", default="data/raw", help="Output directory for scraped JSON")
    parser.add_argument("--headed", action="store_true", help="Show the browser window instead of running invisibly (useful for debugging bot-checks)")
    args = parser.parse_args()

    pages = crawl(
        start_url=args.url,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        delay_seconds=args.delay,
        headless=not args.headed,
    )
    save_results(pages, output_dir=args.output)


if __name__ == "__main__":
    main()