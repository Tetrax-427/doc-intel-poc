"""
Tests for DoclingParser (Milestone A1 — parser, A2 — reading order).

Most tests are unit tests that don't need a real PDF.
Integration tests (marked with @pytest.mark.integration) require:
  - docling installed: pip install docling
  - tests/fixtures/sample_invoice.pdf to exist

Run unit tests only:
    pytest tests/test_docling_parser.py -m "not integration"

Run all including integration:
    pytest tests/test_docling_parser.py
"""

import pytest
from unittest.mock import MagicMock, patch
from parsers.docling_parser import DoclingParser
from core.document import Document, DocumentPage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return DoclingParser()


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.vision_min_words = 50
    return cfg


@pytest.fixture
def app_config():
    """Real config — used for integration tests."""
    from core.config import config
    return config


@pytest.fixture
def app_config_no_llamaparse(app_config):
    """
    Config where LlamaParse is unavailable (no API key).
    Used to verify AutoRouter falls through to Docling.
    """
    cfg = MagicMock()
    cfg.vision_min_words = app_config.vision_min_words
    cfg.chunk_size = app_config.chunk_size
    cfg.chunk_overlap = app_config.chunk_overlap
    cfg.llama_cloud_api_key = None   # forces LlamaParseParser.is_available() → False
    return cfg


# ---------------------------------------------------------------------------
# Unit tests — no docling install or PDF required
# ---------------------------------------------------------------------------

class TestDoclingParserCapabilities:

    def test_get_name(self, parser):
        assert parser.get_name() == "docling"

    def test_can_handle_pdf(self, parser):
        assert parser.can_handle("test.pdf") is True

    def test_can_handle_image(self, parser):
        assert parser.can_handle("test.png") is True

    def test_can_handle_docx(self, parser):
        assert parser.can_handle("document.docx") is True

    def test_cannot_handle_csv(self, parser):
        assert parser.can_handle("data.csv") is False

    def test_cannot_handle_txt(self, parser):
        assert parser.can_handle("notes.txt") is False

    def test_is_available_when_docling_installed(self, parser, mock_config):
        """is_available() should return True if docling can be imported."""
        with patch.dict("sys.modules", {"docling": MagicMock()}):
            assert parser.is_available(mock_config) is True

    def test_is_available_when_docling_missing(self, parser, mock_config):
        """is_available() should return False if docling is not installed."""
        with patch.dict("sys.modules", {"docling": None}):
            # Force ImportError by removing docling from modules
            import sys
            original = sys.modules.pop("docling", None)
            try:
                # Patch builtins.__import__ to raise ImportError for docling
                import builtins
                real_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "docling":
                        raise ImportError("No module named 'docling'")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = parser.is_available(mock_config)
                assert result is False
            finally:
                if original is not None:
                    sys.modules["docling"] = original


class TestDoclingParserReadingOrder:

    def test_reading_order_field_default_empty(self):
        """DocumentPage.reading_order defaults to empty list — backward compatible."""
        page = DocumentPage(
            page_num=1, text="hello world", tables=[], images=[],
            layout=[], entities=[], word_count=2, ocr_confidence=1.0,
        )
        assert page.reading_order == []

    def test_reading_order_field_can_be_set(self):
        """reading_order can be populated."""
        page = DocumentPage(
            page_num=1, text="hello world", tables=[], images=[],
            layout=[], entities=[], word_count=2, ocr_confidence=1.0,
            reading_order=["## Heading", "Paragraph text here."],
        )
        assert len(page.reading_order) == 2
        assert page.reading_order[0] == "## Heading"

    def test_reading_order_in_to_dict(self):
        """reading_order is included in to_dict() output."""
        page = DocumentPage(
            page_num=1, text="text", tables=[], images=[],
            layout=[], entities=[], word_count=1, ocr_confidence=1.0,
            reading_order=["Segment one", "Segment two"],
        )
        d = page.to_dict()
        assert "reading_order" in d
        assert d["reading_order"] == ["Segment one", "Segment two"]


class TestDoclingParserRouting:

    def test_router_selects_docling_when_llamaparse_unavailable(
        self, app_config_no_llamaparse
    ):
        """
        AutoRouter should pick DoclingParser when LlamaParse has no API key
        and docling is installed.
        """
        try:
            import docling  # noqa: F401
        except ImportError:
            pytest.skip("docling not installed — skipping routing test")

        from parsers.router import AutoRouter
        router = AutoRouter(app_config_no_llamaparse)
        parser = router.route("tests/fixtures/sample_invoice.pdf")
        assert parser.get_name() == "docling"


# ---------------------------------------------------------------------------
# Integration tests — require docling + sample_invoice.pdf
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDoclingParserIntegration:

    FIXTURE = "tests/fixtures/sample_invoice.pdf"

    @pytest.fixture(autouse=True)
    def require_docling_and_fixture(self):
        try:
            import docling  # noqa: F401
        except ImportError:
            pytest.skip("docling not installed")
        import os
        if not os.path.exists(self.FIXTURE):
            pytest.skip(f"Fixture not found: {self.FIXTURE}")

    def test_returns_document(self, parser, app_config):
        doc = parser.parse(self.FIXTURE, app_config)
        assert isinstance(doc, Document)
        assert doc.parser_used == "docling"
        assert len(doc.pages) > 0
        assert doc.full_text != ""

    def test_reading_order_populated(self, parser, app_config):
        doc = parser.parse(self.FIXTURE, app_config)
        for page in doc.pages:
            assert isinstance(page.reading_order, list)
            # For a real document Docling should populate at least some segments
            # (empty list is still valid for blank pages — just not for all pages)
        assert any(len(p.reading_order) > 0 for p in doc.pages), \
            "Expected at least one page with reading_order segments"

    def test_tables_extracted(self, parser, app_config):
        doc = parser.parse(self.FIXTURE, app_config)
        assert len(doc.tables) > 0, "Expected at least one table in the invoice fixture"
        for table in doc.tables:
            assert isinstance(table.headers, list)
            assert len(table.headers) > 0

    def test_metadata_complete(self, parser, app_config):
        doc = parser.parse(self.FIXTURE, app_config)
        assert doc.metadata.get("parser_used") == "docling"
        assert "parse_duration_ms" in doc.metadata
        assert doc.metadata.get("has_reading_order") is True

    def test_parse_error_raised_on_bad_file(self, parser, app_config):
        from core.errors import ParseError
        with pytest.raises(ParseError):
            parser.parse("nonexistent_file.pdf", app_config)