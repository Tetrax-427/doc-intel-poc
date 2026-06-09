"""
Handles text-based PDFs locally via pypdf.
No API key required — always available as fallback.
"""

import os
import uuid
import time
from core.document import Document, DocumentPage, make_document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError
from parsers.base import BaseParser
import pypdf

logger = get_logger("pypdf")


class PyPDFParser(BaseParser):

    def get_name(self) -> str:
        return "pypdf"

    def can_handle(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() == ".pdf"

    def is_available(self, config: Config) -> bool:
        return True  # local library, always available

    def parse(self, file_path: str, config: Config) -> Document:
        start = time.time()
        file_name = os.path.basename(file_path)

        try:
            reader = pypdf.PdfReader(file_path)
        except Exception as e:
            raise ParseError(
                f"Could not open PDF: {e}",
                file_name=file_name,
                retryable=False,
            )

        pages = []
        for i, pdf_page in enumerate(reader.pages):
            try:
                text = pdf_page.extract_text() or ""
                text = text.replace("\x00", " ").strip()
            except Exception as e:
                logger.warning("Failed to extract text from page",
                               file=file_name, page=i + 1, error=str(e))
                text = ""

            if not text:
                continue

            pages.append(DocumentPage(
                page_num=i + 1,
                text=text,
                tables=[],
                images=[],
                layout=[],
                entities=[],
                word_count=len(text.split()),
                ocr_confidence=1.0,  # text PDF — no OCR needed
            ))

        if not pages:
            raise ParseError(
                "pypdf extracted no text — file may be scanned or empty",
                file_name=file_name,
                retryable=False,
            )

        duration_ms = int((time.time() - start) * 1000)
        logger.info("pypdf parse complete",
                    file=file_name,
                    pages=len(pages),
                    duration_ms=duration_ms)

        return make_document(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_type=".pdf",
            file_path=file_path,
            pages=pages,
            parser_used="pypdf",
            metadata={
                "is_scanned": False,
                "parse_duration_ms": duration_ms,
                "file_size_bytes": os.path.getsize(file_path),
                "total_pdf_pages": len(reader.pages),
            },
        )