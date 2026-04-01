import random
import time
import csv
import logging
import requests
import cloudscraper
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── User-agents & session helpers ──────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


def get_random_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.moneycontrol.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def refresh_session() -> requests.Session:
    new_session = requests.Session()
    new_session.headers.update(get_random_headers())
    return new_session


# Global session + cloudscraper instance
session = refresh_session()
scraper = cloudscraper.create_scraper()
scraper.headers.update(get_random_headers())


# ── Fetch helpers ──────────────────────────────────────────────────────────────
def fetch_with_requests(url: str, timeout: int = 12):
    """Try a plain requests fetch first (faster, no overhead)."""
    try:
        logger.info(f"requests fetch → {url}")
        resp = session.get(url, timeout=timeout, allow_redirects=True, headers=get_random_headers())
        if resp.status_code == 200:
            logger.info("requests succeeded")
            return resp
        logger.warning(f"requests returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"requests error: {e}")
    return None


def fetch_with_cloudscraper(url: str, timeout: int = 15):
    """Fallback: bypass Cloudflare / JS challenges."""
    try:
        logger.info(f"cloudscraper fetch → {url}")
        resp = scraper.get(url, timeout=timeout, allow_redirects=True, headers=get_random_headers())
        if resp.status_code == 200:
            logger.info("cloudscraper succeeded")
            return resp
        logger.warning(f"cloudscraper returned HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"cloudscraper error: {e}")
    return None


def fetch(url: str):
    """Try requests first; fall back to cloudscraper."""
    resp = fetch_with_requests(url)
    if resp is None:
        resp = fetch_with_cloudscraper(url)
    return resp


# ── Content extractor ──────────────────────────────────────────────────────────
# Moneycontrol uses several article body containers depending on article type.
_ARTICLE_SELECTORS = [
    {"class": "article-desc"},          # standard news articles
    {"class": "arti-flow"},             # some opinion / analysis pages
    {"id": "article-content"},          # alternate layout
    {"class": "content_wrapper"},       # markets / data pages
    {"class": "artText"},               # older article layout
]

# Moneycontrol headline containers (tried in order)
_HEADLINE_SELECTORS = [
    ("h1", {"class": "article_title"}),
    ("h1", {"class": "artTitle"}),
    ("h1", {"itemprop": "headline"}),
    ("h1", {"class": "title"}),
    ("h1", {}),                          # any <h1> as last resort
]

_JUNK_TAGS = ["script", "style", "noscript", "iframe",
               "figure", "figcaption", "aside", "nav",
               "header", "footer", "form", "button"]


def extract_headline(soup: BeautifulSoup) -> str:
    """Extract the article headline from a parsed BeautifulSoup object."""
    for tag, attrs in _HEADLINE_SELECTORS:
        el = soup.find(tag, attrs) if attrs else soup.find(tag)
        if el:
            headline = el.get_text(separator=" ", strip=True)
            if headline:
                return headline
    return ""


def extract_article_content(html: str, url: str) -> tuple[str, str]:
    """
    Parse headline and body from a Moneycontrol page HTML string.
    Returns (headline, content) — both plain-text, empty string on failure.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract headline before removing any tags
    headline = extract_headline(soup)

    # Remove noisy tags
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    article_div = None
    for selector in _ARTICLE_SELECTORS:
        article_div = soup.find("div", selector)
        if article_div:
            break

    if article_div is None:
        # Last-resort: grab every <p> inside <article> or <main>
        container = soup.find("article") or soup.find("main")
        if container:
            paragraphs = container.find_all("p")
        else:
            logger.warning(f"No article container found for {url}")
            return headline, ""
    else:
        paragraphs = article_div.find_all("p")

    lines = [p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True)]
    content = "\n".join(lines)
    return headline, content


# ── Main scraper ───────────────────────────────────────────────────────────────
def scrape_articles(urls: list[str], output_csv: str = "articles.csv") -> None:
    """
    Scrape article content from each URL and save to a single-column CSV.

    Args:
        urls:       List of Moneycontrol article URLs.
        output_csv: Path to the output CSV file.
    """
    results = []
    session_refresh_every = 8   # refresh requests session periodically

    for idx, url in enumerate(urls, start=1):
        logger.info(f"── Article {idx}/{len(urls)} ──")

        # Periodically refresh session & rotate user-agent
        if idx % session_refresh_every == 0:
            global session
            session = refresh_session()
            scraper.headers.update(get_random_headers())
            logger.info("Session refreshed")

        resp = fetch(url)
        if resp is None:
            logger.error(f"Skipping (could not fetch): {url}")
            results.append({"headline": "", "content": ""})
        else:
            headline, content = extract_article_content(resp.text, url)
            if headline:
                logger.info(f"Headline: {headline[:80]}")
            else:
                logger.warning("No headline found")
            if content:
                logger.info(f"Extracted ~{len(content)} chars")
            else:
                logger.warning("Empty content extracted")
            results.append({"headline": headline, "content": content})

        # Human-like delay between requests
        if idx < len(urls):
            delay = random.uniform(2.0, 4.5)
            logger.info(f"Sleeping {delay:.1f}s …")
            time.sleep(delay)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "content"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nDone! {len(results)} rows written → {output_csv}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ↓↓ Replace with your actual list of Moneycontrol article URLs ↓↓
    URLS = [
        "https://www.moneycontrol.com/news/business/example-article-1.html",
        "https://www.moneycontrol.com/news/business/example-article-2.html",
        # add more URLs here …
    ]

    scrape_articles(urls=URLS, output_csv="articles.csv")