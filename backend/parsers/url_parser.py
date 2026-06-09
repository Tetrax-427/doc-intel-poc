"""
Handles URLs — fetches and extracts text via httpx + BeautifulSoup.
Special-cases Wikipedia URLs to use the Wikipedia API for clean text.
"""

import os
import re
import uuid
import time
from core.document import Document, DocumentPage, make_document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError
from parsers.base import BaseParser
import httpx
from bs4 import BeautifulSoup
     
logger = get_logger("url")

CHUNK_SIZE = 3000


class UrlParser(BaseParser):

    def get_name(self) -> str:
        return "url"

    def can_handle(self, file_path: str) -> bool:
        """Handles strings that look like URLs (http/https)."""
        return file_path.startswith("http://") or file_path.startswith("https://")

    def is_available(self, config: Config) -> bool:
        return True

    def parse(self, file_path: str, config: Config) -> Document:
        """
        file_path is actually the URL string for this parser.
        Kept as file_path to satisfy the BaseParser interface.
        """
        start = time.time()
        url = file_path

        title, text = self._fetch(url)

        if not text.strip():
            raise ParseError(
                f"Could not extract text from URL: {url}",
                file_name=url,
                retryable=True,
            )

        pages = []
        chunks = [
            text[i:i + CHUNK_SIZE].strip()
            for i in range(0, len(text), CHUNK_SIZE)
            if text[i:i + CHUNK_SIZE].strip()
        ]

        for page_num, chunk in enumerate(chunks, start=1):
            pages.append(DocumentPage(
                page_num=page_num,
                text=chunk,
                tables=[],
                images=[],
                layout=[],
                entities=[],
                word_count=len(chunk.split()),
                ocr_confidence=1.0,
            ))

        if not pages:
            raise ParseError(
                f"URL produced no content after chunking: {url}",
                file_name=url,
                retryable=False,
            )

        # Use a safe filename from the title
        safe_name = f"{title[:60].replace('/', '-')}.url" if title else "webpage.url"

        duration_ms = int((time.time() - start) * 1000)
        logger.info("URL parse complete",
                    url=url,
                    title=title,
                    pages=len(pages),
                    duration_ms=duration_ms)

        return make_document(
            id=str(uuid.uuid4()),
            file_name=safe_name,
            file_type=".url",
            file_path=url,
            pages=pages,
            parser_used="url",
            metadata={
                "is_scanned": False,
                "parse_duration_ms": duration_ms,
                "source_url": url,
                "title": title,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> tuple[str, str]:
        """Returns (title, text). Tries Wikipedia API first for wiki URLs."""
        wiki_text = self._try_wikipedia(url)
        if wiki_text:
            return wiki_text

        return self._fetch_generic(url)

    def _try_wikipedia(self, url: str) -> tuple[str, str] | None:
        """Use Wikipedia API for clean plain-text extraction."""
        wiki_match = re.search(r"wikipedia\.org/wiki/(.+)", url)
        if not wiki_match:
            return None

        page_title = wiki_match.group(1).split("#")[0]
        try:
            r = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": page_title,
                    "prop": "extracts",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "format": "json",
                    "redirects": "1",
                },
                headers={"User-Agent": "DocIntel/1.0"},
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                page = next(iter(pages.values()))
                if page.get("pageid"):
                    title = page.get("title", page_title)
                    text = page.get("extract", "")
                    if text:
                        logger.info("Wikipedia API fetch successful", title=title)
                        return title, text
        except Exception as e:
            logger.warning("Wikipedia API failed — falling back to generic fetch",
                           url=url, error=str(e))
        return None

    def _fetch_generic(self, url: str) -> tuple[str, str]:
        """Generic HTML fetch and text extraction via BeautifulSoup."""
        try:
            
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            raise ParseError(
                f"HTTP fetch failed: {e}",
                file_name=url,
                retryable=True,
            )

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url
        main = soup.find("article") or soup.find("main") or soup.find("body")
        raw_text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return title, "\n".join(lines)