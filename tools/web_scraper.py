import httpx
from bs4 import BeautifulSoup
from urllib import robotparser
from urllib.parse import urlparse
import time
import asyncio
from utils.logging import setup_logger

logger = setup_logger("web_scraper")
class ScrapeError(Exception):
    """Custom exception for scraping errors."""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)

async def allowed_to_scrape(url: str) -> bool:
    parsed = urlparse(url)
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    rp.read()
    return rp.can_fetch("*", url)

async def fetch_page(url: str, retries: int = 3, backoff: float = 1.0) -> str:
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                print(f"Got response from {url}: {resp.text}")
                resp.raise_for_status()
                return resp.text
        except (httpx.HTTPError, httpx.RequestError) as e:
            if i < retries - 1:
                await asyncio.sleep(backoff * (2 ** i))
                continue
            raise ScrapeError(f"Failed fetching {url}: {e}")

def parse_html(html: str) -> str:
    print("Parsing HTML content...")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")