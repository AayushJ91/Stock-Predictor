import requests
from bs4 import BeautifulSoup, Comment
import logging
import time
import random
import cloudscraper
from .logger import setup_logger

# Extracting the links of every page for a stock from moneycontrol
logger = setup_logger("headlines_scraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.moneycontrol.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

HEADERS_PAGES = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.moneycontrol.com/"
}

def refresh_session():
    new_session = requests.Session()
    new_session.headers.update(get_random_headers())
    return new_session

session = requests.Session()
random_header = get_random_headers()
session.headers.update(random_header)

# Initialize cloudscraper for anti-scraping bypass
scraper = cloudscraper.create_scraper()
scraper.headers.update(HEADERS_PAGES)


def fetch_page_with_cloudscraper(url, timeout=10):
    """
    Fetch page content using cloudscraper to bypass Cloudflare protection.
    
    Parameters
    ----------
    url : str
        URL to fetch
    timeout : int
        Request timeout in seconds
        
    Returns
    -------
    requests.Response or None
        Response object if successful, None if failed
    """
    try:
        logger.info(f"Attempting cloudscraper fetch: {url}")
        response = scraper.get(url, timeout=timeout, headers=get_random_headers())
        if response.status_code == 200:
            logger.info(f"Cloudscraper succeeded: {url}")
            return response
        else:
            logger.warning(f"Cloudscraper returned {response.status_code}: {url}")
            return None
    except Exception as e:
        logger.error(f"Cloudscraper failed for {url}: {e}")
        return None


def headlines_extractor(url, max_pages=30):
    global session, scraper
    """
    will go to every article of the specified news article on money control and fetch the link of the each and every article.

    input: url of the news of that particle ticker
    output: a lsit of links
    """
    page = 1
    all_news = []
    seen_links = set()
    logger.info(f"Starting scrape for URL: {url}")

    while True:
        if page > max_pages:
            break

        if page % 7 == 0:
            session = refresh_session()
            scraper = cloudscraper.create_scraper()
            scraper.headers.update(get_random_headers())
        if page == 1:
            using_url = url+"/"
        else:
            using_url = f"{url}/page-{page}/"

        logger.info(f"Scraping page {page}: {using_url}")

        response = session.get(using_url, timeout=10, headers=get_random_headers())

        if response.status_code != 200:
            logger.warning(f"Regular request returned {response.status_code}. Trying cloudscraper...")
            response = fetch_page_with_cloudscraper(using_url)
            
            if response is None or response.status_code != 200:
                logger.warning(f"Both methods failed for page {page}. Stopping.")
                break

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="clearfix")

        if not articles:
            logger.info("No articles found. Stopping.")
            break

        new_found = False

        for article in articles:
            headline_tag = article.find("h2")
            link_tag = article.find("a", href=True)

            if headline_tag and link_tag:
                link = link_tag["href"]

                if link not in seen_links:
                    seen_links.add(link)
                    new_found = True

                    all_news.append({
                        "headline": headline_tag.text.strip(),
                        "link": link
                    })

        logger.info(f"Page {page}: Found {len(articles)} articles, {len(seen_links)} unique so far")
        
        if not new_found:
            logger.info("No new articles. Reached last page.")
            break

        
        time.sleep(random.uniform(2.5, 5.0)) # Human-like random delay
        page += 1

    logger.info(f"Scraping complete. Total unique articles: {len(all_news)}")
    return all_news


