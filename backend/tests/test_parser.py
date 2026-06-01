"""
tests/test_parsers.py

Tests for backend/parsers/
Validates that every parser returns a proper Document and fails correctly.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.document import Document
from core.errors import ParseError, UnsupportedFileTypeError
from parsers.base import BaseParser
from parsers.router import AutoRouter
from parsers.llamaparse import LlamaParseParser
from parsers.pypdf_parser import PyPDFParser
from parsers.docx_parser import DocxParser
from parsers.csv_parser import CsvParser
from parsers.text_parser import TextParser
from parsers.url_parser import UrlParser


# ---------------------------------------------------------------------------
# BaseParser contract
# ---------------------------------------------------------------------------

class TestBaseParser:

    def test_cannot_instantiate_directly(self):
        """BaseParser is abstract — must not be instantiable."""
        with pytest.raises(TypeError):
            BaseParser()

    def test_all_parsers_are_subclasses(self):
        parsers = [LlamaParseParser(), PyPDFParser(), DocxParser(),
                   CsvParser(), TextParser(), UrlParser()]
        for p in parsers:
            assert isinstance(p, BaseParser), f"{p} is not a BaseParser subclass"

    def test_all_parsers_have_required_methods(self):
        parsers = [LlamaParseParser(), PyPDFParser(), DocxParser(),
                   CsvParser(), TextParser(), UrlParser()]
        for p in parsers:
            assert callable(p.can_handle)
            assert callable(p.parse)
            assert callable(p.get_name)
            assert callable(p.is_available)

    def test_all_parsers_have_unique_names(self):
        parsers = [LlamaParseParser(), PyPDFParser(), DocxParser(),
                   CsvParser(), TextParser(), UrlParser()]
        names = [p.get_name() for p in parsers]
        assert len(names) == len(set(names)), f"Duplicate parser names: {names}"


# ---------------------------------------------------------------------------
# can_handle routing
# ---------------------------------------------------------------------------

class TestCanHandle:

    def test_llamaparse_handles_pdf_and_images(self):
        p = LlamaParseParser()
        for ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"]:
            assert p.can_handle(f"file{ext}") is True

    def test_llamaparse_does_not_handle_csv(self):
        assert LlamaParseParser().can_handle("file.csv") is False

    def test_pypdf_handles_pdf_only(self):
        assert PyPDFParser().can_handle("file.pdf") is True
        assert PyPDFParser().can_handle("file.docx") is False

    def test_docx_handles_docx_only(self):
        assert DocxParser().can_handle("file.docx") is True
        assert DocxParser().can_handle("file.pdf") is False

    def test_csv_handles_csv_and_xlsx(self):
        assert CsvParser().can_handle("file.csv") is True
        assert CsvParser().can_handle("file.xlsx") is True
        assert CsvParser().can_handle("file.pdf") is False

    def test_text_handles_txt_md_rtf(self):
        for ext in [".txt", ".md", ".rtf"]:
            assert TextParser().can_handle(f"file{ext}") is True
        assert TextParser().can_handle("file.pdf") is False

    def test_url_handles_http_and_https(self):
        assert UrlParser().can_handle("https://example.com") is True
        assert UrlParser().can_handle("http://example.com") is True
        assert UrlParser().can_handle("file.pdf") is False


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:

    def test_local_parsers_always_available(self, cfg):
        for parser in [PyPDFParser(), DocxParser(), CsvParser(), TextParser(), UrlParser()]:
            assert parser.is_available(cfg) is True

    def test_llamaparse_unavailable_without_key(self, cfg):
        """LlamaParse requires LLAMA_CLOUD_API_KEY — should be False in test env."""
        assert LlamaParseParser().is_available(cfg) is False


# ---------------------------------------------------------------------------
# CsvParser
# ---------------------------------------------------------------------------

class TestCsvParser:

    def test_returns_document(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert isinstance(doc, Document)

    def test_document_has_required_fields(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.id
        assert doc.file_name == "sample_table.csv"
        assert doc.file_type == ".csv"
        assert doc.full_text

    def test_document_has_pages(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert len(doc.pages) > 0

    def test_document_has_tables(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.has_tables is True

    def test_table_has_correct_headers(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.tables[0].headers == ["Name", "Amount", "Date", "Status"]

    def test_table_has_correct_row_count(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.tables[0].row_count == 3

    def test_parser_used_is_csv(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.parser_used == "csv"

    def test_metadata_populated(self, sample_csv, cfg):
        doc = CsvParser().parse(sample_csv, cfg)
        assert doc.metadata["file_size_bytes"] > 0
        assert doc.metadata["parse_duration_ms"] >= 0

    def test_raises_parse_error_on_empty_file(self, sample_empty_csv, cfg):
        with pytest.raises((ParseError, Exception)):
            CsvParser().parse(sample_empty_csv, cfg)

    def test_never_returns_none(self, sample_csv, cfg):
        result = CsvParser().parse(sample_csv, cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# TextParser
# ---------------------------------------------------------------------------

class TestTextParser:

    def test_returns_document_for_txt(self, sample_txt, cfg):
        doc = TextParser().parse(sample_txt, cfg)
        assert isinstance(doc, Document)

    def test_returns_document_for_md(self, sample_md, cfg):
        doc = TextParser().parse(sample_md, cfg)
        assert isinstance(doc, Document)

    def test_document_has_required_fields(self, sample_txt, cfg):
        doc = TextParser().parse(sample_txt, cfg)
        assert doc.id and doc.file_name and doc.full_text

    def test_word_count_positive(self, sample_txt, cfg):
        doc = TextParser().parse(sample_txt, cfg)
        assert doc.word_count > 0

    def test_parser_used_is_text(self, sample_txt, cfg):
        doc = TextParser().parse(sample_txt, cfg)
        assert doc.parser_used == "text"

    def test_md_file_type(self, sample_md, cfg):
        doc = TextParser().parse(sample_md, cfg)
        assert doc.file_type == ".md"

    def test_raises_on_empty_file(self, sample_empty_txt, cfg):
        with pytest.raises(ParseError) as exc_info:
            TextParser().parse(sample_empty_txt, cfg)
        assert exc_info.value.code == "PARSE_001"

    def test_never_returns_none(self, sample_txt, cfg):
        result = TextParser().parse(sample_txt, cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# AutoRouter
# ---------------------------------------------------------------------------

class TestAutoRouter:

    @pytest.fixture
    def router(self, cfg):
        return AutoRouter(cfg)

    def test_routes_csv_to_csv_parser(self, router, sample_csv):
        parser = router.route(sample_csv)
        assert parser.get_name() == "csv"

    def test_routes_txt_to_text_parser(self, router, sample_txt):
        parser = router.route(sample_txt)
        assert parser.get_name() == "text"

    def test_routes_md_to_text_parser(self, router, sample_md):
        parser = router.route(sample_md)
        assert parser.get_name() == "text"

    def test_routes_url_to_url_parser(self, router):
        parser = router.route("https://example.com/page")
        assert parser.get_name() == "url"

    def test_routes_pdf_to_pypdf_without_llamaparse_key(self, router, tmp_path):
        """Without LLAMA_CLOUD_API_KEY, PDFs must route to pypdf."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        parser = router.route(str(fake_pdf))
        assert parser.get_name() == "pypdf"

    def test_raises_for_unknown_extension(self, router, unknown_extension_file):
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            router.route(unknown_extension_file)
        assert exc_info.value.code == "PARSE_002"

    def test_parse_returns_document(self, router, sample_csv):
        doc = router.parse(sample_csv)
        assert isinstance(doc, Document)

    def test_all_parsers_registered(self, router):
        names = {p.get_name() for p in router._parsers}
        expected = {"llamaparse", "pypdf", "docx", "csv", "text", "url"}
        assert expected == names