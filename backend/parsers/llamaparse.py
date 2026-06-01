"""
backend/parsers/llamaparse.py

Handles PDFs (text + scanned) and images via the LlamaParse cloud API.
Falls back signals to AutoRouter if LLAMA_CLOUD_API_KEY is not set.
"""

import os
import uuid
import time
from core.document import Document, DocumentPage, Table, ImageElement, make_document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError
from parsers.base import BaseParser

logger = get_logger("llamaparse")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"}


class LlamaParseParser(BaseParser):

    def get_name(self) -> str:
        return "llamaparse"

    def can_handle(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

    def is_available(self, config: Config) -> bool:
        return bool(config.llama_cloud_api_key)

    def parse(self, file_path: str, config: Config) -> Document:
        start = time.time()
        file_name = os.path.basename(file_path)

        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(
                api_key=config.llama_cloud_api_key,
                result_type="markdown",
                verbose=False,
            )
            docs = parser.load_data(file_path)
        except Exception as e:
            raise ParseError(
                f"LlamaParse API call failed: {e}",
                file_name=file_name,
                retryable=True,
            )

        pages = []
        for i, doc in enumerate(docs):
            if not doc.text.strip():
                continue
            clean_text = doc.text.replace("\x00", " ").strip()
            tables = self._extract_tables_from_markdown(clean_text, page_num=i + 1)
            pages.append(DocumentPage(
                page_num=i + 1,
                text=clean_text,
                tables=tables,
                images=[],
                layout=[],
                entities=[],
                word_count=len(clean_text.split()),
                ocr_confidence=0.9,  # LlamaParse is high quality
            ))

        if not pages:
            raise ParseError(
                "LlamaParse returned no content",
                file_name=file_name,
                retryable=False,
            )

        duration_ms = int((time.time() - start) * 1000)
        logger.info("LlamaParse complete",
                    file=file_name,
                    pages=len(pages),
                    duration_ms=duration_ms)

        return make_document(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_type=os.path.splitext(file_path)[1].lower(),
            file_path=file_path,
            pages=pages,
            parser_used="llamaparse",
            vision_used=False,
            metadata={
                "is_scanned": False,
                "parse_duration_ms": duration_ms,
                "file_size_bytes": os.path.getsize(file_path),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_tables_from_markdown(self, text: str, page_num: int) -> list[Table]:
        """Parse markdown pipe tables into structured Table objects."""
        tables = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            # A markdown table starts with a pipe row followed by a separator row
            if "|" in lines[i] and i + 1 < len(lines) and "---" in lines[i + 1]:
                headers = [h.strip() for h in lines[i].split("|") if h.strip()]
                rows = []
                j = i + 2
                while j < len(lines) and "|" in lines[j]:
                    row = [c.strip() for c in lines[j].split("|") if c.strip()]
                    if row:
                        rows.append(row)
                    j += 1

                if headers and rows:
                    tables.append(Table(
                        page_num=page_num,
                        title="",
                        headers=headers,
                        rows=rows,
                        cells=[],
                        raw_text="\n".join(lines[i:j]),
                    ))
                i = j
            else:
                i += 1
        return tables