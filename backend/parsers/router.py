"""
AutoRouter selects the best available parser for a given file.
Falls back gracefully when the preferred parser is unavailable.

Routing logic (in priority order):
  Image files (.png/.jpg/.jpeg/.webp/.tiff)
      → LlamaParse → Docling → pypdf fallback

  Scanned PDF (avg words/page < vision_min_words)
      → LlamaParse → Docling (OCR) → pypdf fallback

  Regular PDF
      → LlamaParse (best, paid) → Docling (free, local) → pypdf (fast fallback)

  DOCX                → DocxParser
  CSV / XLSX          → CsvParser
  TXT / MD / RTF      → TextParser
  URL                 → UrlParser

DoclingParser is skipped transparently if docling is not installed —
is_available() returns False and _get_parser() returns None, so the
chain falls through to pypdf automatically.
"""

import os
from core.document import Document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from parsers.base import BaseParser
from parsers.llamaparse import LlamaParseParser
from parsers.docling_parser import DoclingParser
from parsers.pypdf_parser import PyPDFParser
from parsers.docx_parser import DocxParser
from parsers.csv_parser import CsvParser
from parsers.text_parser import TextParser
from parsers.url_parser import UrlParser

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
        """Register all parsers in priority order."""
        self._parsers = [
            LlamaParseParser(),
            DoclingParser(),   
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
        Return the best available parser for this file.
        Raises UnsupportedFileTypeError if no parser can handle it.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Images —
        if ext in IMAGE_EXTENSIONS:
            parser = (
                self._get_parser("llamaparse") or
                self._get_parser("docling") or
                self._get_parser("pypdf")
            )
            if parser:
                logger.info("Routing image", file=os.path.basename(file_path),
                            parser=parser.get_name())
                return parser

        if ext == ".pdf":
            if self._is_scanned(file_path):
                # Scanned PDFs need OCR — LlamaParse or Docling (both support it)
                logger.info("Scanned PDF detected", file=os.path.basename(file_path))
                parser = (
                    self._get_parser("llamaparse") or
                    self._get_parser("docling") or
                    self._get_parser("pypdf")
                )
            else:
                # Text PDF — full priority chain
                parser = (
                    self._get_parser("llamaparse") or
                    self._get_parser("docling") or
                    self._get_parser("pypdf")
                )

            if parser:
                logger.info("Routing PDF", file=os.path.basename(file_path),
                            parser=parser.get_name())
                return parser

        # All other types — first parser that can_handle + is_available wins
        for parser in self._parsers:
            if parser.can_handle(file_path) and parser.is_available(self.config):
                logger.info("Routing file", file=os.path.basename(file_path),
                            parser=parser.get_name())
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
            total_words = sum(
                len((reader.pages[i].extract_text() or "").split())
                for i in range(pages_to_check)
            )
            avg_words = total_words / pages_to_check
            is_scanned = avg_words < self.config.vision_min_words
            if is_scanned:
                logger.info(
                    "Scanned PDF heuristic triggered",
                    file=os.path.basename(file_path),
                    avg_words_per_page=round(avg_words, 1),
                    threshold=self.config.vision_min_words,
                )
            return is_scanned
        except Exception as e:
            logger.warning("Scanned PDF check failed — assuming text PDF",
                           file=os.path.basename(file_path), error=str(e))
            return False