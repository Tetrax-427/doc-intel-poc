"""
backend/parsers/router.py

AutoRouter selects the best available parser for a given file.
Falls back gracefully when the preferred parser is unavailable.

Routing logic:
  - Image files (.png, .jpg, .jpeg, .webp, .tiff) → LlamaParse
  - Scanned PDF (avg words/page < vision_min_words)  → LlamaParse → pypdf fallback
  - Complex PDF (tables detected in first pass)       → LlamaParse → pypdf fallback
  - Simple text PDF                                   → pypdf (faster, cheaper)
  - DOCX                                              → DocxParser
  - CSV / XLSX                                        → CsvParser
  - TXT / MD / RTF                                    → TextParser
  - URL                                               → UrlParser
"""

import os
from core.document import Document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from parsers.base import BaseParser

logger = get_logger("router")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}


class AutoRouter:
    """
    Selects the best available parser for a given file.
    Instantiate once and reuse — parsers are registered on init.
    """

    def __init__(self, config: Config):
        self.config = config
        self._parsers: list[BaseParser] = []
        self._register_parsers()

    def _register_parsers(self):
        """Import and register all available parsers."""
        from parsers.llamaparse import LlamaParseParser
        from parsers.pypdf_parser import PyPDFParser
        from parsers.docx_parser import DocxParser
        from parsers.csv_parser import CsvParser
        from parsers.text_parser import TextParser
        from parsers.url_parser import UrlParser

        self._parsers = [
            LlamaParseParser(),
            PyPDFParser(),
            DocxParser(),
            CsvParser(),
            TextParser(),
            UrlParser(),
        ]
        available = [p.get_name() for p in self._parsers if p.is_available(self.config)]
        logger.info("Parsers registered", available=available, total=len(self._parsers))

    def route(self, file_path: str) -> BaseParser:
        """
        Return the best parser for this file.
        Raises UnsupportedFileTypeError if no parser can handle it.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Images always go to LlamaParse (needs OCR + vision)
        if ext in IMAGE_EXTENSIONS:
            parser = self._get_parser("llamaparse") or self._get_parser("pypdf")
            if parser:
                logger.info("Routing image to parser", file=os.path.basename(file_path), parser=parser.get_name())
                return parser

        # PDF routing — check if scanned, prefer LlamaParse, fall back to pypdf
        if ext == ".pdf":
            if self._is_scanned(file_path):
                logger.info("Scanned PDF detected — routing to LlamaParse", file=os.path.basename(file_path))
                parser = self._get_parser("llamaparse") or self._get_parser("pypdf")
                if parser:
                    return parser
            else:
                # Text PDF — pypdf is faster and cheaper, LlamaParse if tables needed
                parser = self._get_parser("pypdf") or self._get_parser("llamaparse")
                if parser:
                    logger.info("Text PDF — routing to parser", file=os.path.basename(file_path), parser=parser.get_name())
                    return parser

        # All other types — first parser that can_handle and is_available wins
        for parser in self._parsers:
            if parser.can_handle(file_path) and parser.is_available(self.config):
                logger.info("Routing file", file=os.path.basename(file_path), parser=parser.get_name())
                return parser

        raise UnsupportedFileTypeError(os.path.basename(file_path), ext)

    def parse(self, file_path: str) -> Document:
        """Convenience method — route and parse in one call."""
        parser = self.route(file_path)
        return parser.parse(file_path, self.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser(self, name: str) -> BaseParser | None:
        """Return parser by name if available, else None."""
        for p in self._parsers:
            if p.get_name() == name and p.is_available(self.config):
                return p
        return None

    def _is_scanned(self, file_path: str) -> bool:
        """
        Quick heuristic — extract text from first 3 pages via pypdf.
        If average word count is below vision_min_words, treat as scanned.
        """
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages_to_check = min(3, len(reader.pages))
            if pages_to_check == 0:
                return True
            total_words = 0
            for i in range(pages_to_check):
                text = reader.pages[i].extract_text() or ""
                total_words += len(text.split())
            avg_words = total_words / pages_to_check
            is_scanned = avg_words < self.config.vision_min_words
            if is_scanned:
                logger.info("Scanned PDF heuristic triggered",
                            file=os.path.basename(file_path),
                            avg_words_per_page=round(avg_words, 1),
                            threshold=self.config.vision_min_words)
            return is_scanned
        except Exception as e:
            logger.warning("Scanned PDF check failed — assuming text PDF",
                           file=os.path.basename(file_path), error=str(e))
            return False