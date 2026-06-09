"""
Handles .txt, .md, and .rtf files.
No external dependencies beyond striprtf for RTF.
"""

import os
import uuid
import time
from core.document import Document, DocumentPage, make_document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError
from parsers.base import BaseParser
from striprtf.striprtf import rtf_to_text

logger = get_logger("text")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rtf"}
CHUNK_SIZE = 3000  # characters per logical page


class TextParser(BaseParser):

    def get_name(self) -> str:
        return "text"

    def can_handle(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

    def is_available(self, config: Config) -> bool:
        return True

    def parse(self, file_path: str, config: Config) -> Document:
        start = time.time()
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        text = self._read_file(file_path, ext, file_name)

        if not text.strip():
            raise ParseError(
                "File contains no extractable text",
                file_name=file_name,
                retryable=False,
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

        duration_ms = int((time.time() - start) * 1000)
        logger.info("Text parse complete",
                    file=file_name,
                    ext=ext,
                    pages=len(pages),
                    duration_ms=duration_ms)

        return make_document(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_type=ext,
            file_path=file_path,
            pages=pages,
            parser_used="text",
            metadata={
                "is_scanned": False,
                "parse_duration_ms": duration_ms,
                "file_size_bytes": os.path.getsize(file_path),
                "format": ext.lstrip("."),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self, file_path: str, ext: str, file_name: str) -> str:
        try:
            if ext == ".rtf":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return rtf_to_text(f.read())
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            raise ParseError(
                f"Could not read file: {e}",
                file_name=file_name,
                retryable=False,
            )