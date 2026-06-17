"""
Tests for Group B — Document Understanding.

Covers:
  B1 — BoundingBox dataclass, reading_order_bboxes on DocumentPage
  B2 — layout_elements: checkbox detection, signature detection
  B3 — ImageElement extensions, figure chunks in ingestion, format_source in retrieval

All tests are pure unit tests. No LLM calls, no DB, no file I/O, no docling install needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_config_no_vision():
    """Config with vision disabled — VISION_PROVIDER not set."""
    cfg = MagicMock()
    cfg.vision_provider = None
    cfg.vision_model = None
    cfg.vision_min_words = 50
    cfg.chunk_size = 512
    cfg.chunk_overlap = 50
    return cfg


@pytest.fixture
def app_config_with_vision():
    """Config with vision enabled."""
    cfg = MagicMock()
    cfg.vision_provider = "openai"
    cfg.vision_model = "gpt-4o"
    cfg.vision_min_words = 50
    cfg.chunk_size = 512
    cfg.chunk_overlap = 50
    return cfg


def make_page(page_num=1, text="Sample text content here.", reading_order=None,
              reading_order_bboxes=None, images=None, layout=None):
    """Build a minimal DocumentPage for testing."""
    from core.document import DocumentPage
    return DocumentPage(
        page_num=page_num,
        text=text,
        tables=[],
        images=images or [],
        layout=layout or [],
        entities=[],
        word_count=len(text.split()),
        ocr_confidence=1.0,
        reading_order=reading_order or [],
        reading_order_bboxes=reading_order_bboxes or [],
    )


def make_bbox(x=0.1, y=0.1, w=0.5, h=0.2, page=1, pw=595.0, ph=842.0):
    """Build a BoundingBox for testing."""
    from core.document import BoundingBox
    return BoundingBox(x=x, y=y, width=w, height=h,
                       page_num=page, page_width=pw, page_height=ph)


# ===========================================================================
# B1 — BoundingBox
# ===========================================================================

class TestBoundingBox:

    def test_to_dict_contains_all_fields(self):
        from core.document import BoundingBox
        bbox = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.3,
                           page_num=1, page_width=595.0, page_height=842.0)
        d = bbox.to_dict()
        assert d["x"] == 0.1
        assert d["y"] == 0.2
        assert d["width"] == 0.5
        assert d["height"] == 0.3
        assert d["page"] == 1
        assert d["page_width"] == 595.0
        assert d["page_height"] == 842.0

    def test_to_dict_rounds_to_4_decimal_places(self):
        from core.document import BoundingBox
        bbox = BoundingBox(x=0.123456789, y=0.1, width=0.5, height=0.3,
                           page_num=1, page_width=595.0, page_height=842.0)
        d = bbox.to_dict()
        assert d["x"] == 0.1235

    def test_from_dict_round_trips(self):
        from core.document import BoundingBox
        bbox = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.3,
                           page_num=2, page_width=595.0, page_height=842.0)
        reconstructed = BoundingBox.from_dict(bbox.to_dict())
        assert reconstructed.x == bbox.x
        assert reconstructed.y == bbox.y
        assert reconstructed.page_num == 2

    def test_from_dict_returns_none_for_none(self):
        from core.document import BoundingBox
        assert BoundingBox.from_dict(None) is None

    def test_normalized_coords_in_range(self):
        """x + width should not exceed 1.0 (sanity check on normalization)."""
        bbox = make_bbox(x=0.1, w=0.5)
        assert bbox.x + bbox.width <= 1.0
        assert bbox.y + bbox.height <= 1.0


class TestDocumentPageBboxes:

    def test_reading_order_bboxes_defaults_empty(self):
        page = make_page()
        assert page.reading_order_bboxes == []

    def test_reading_order_bboxes_parallel_to_reading_order(self):
        bbox1 = make_bbox(x=0.0, y=0.0)
        bbox2 = make_bbox(x=0.0, y=0.3)
        page = make_page(
            reading_order=["Segment one", "Segment two"],
            reading_order_bboxes=[bbox1, bbox2],
        )
        assert len(page.reading_order_bboxes) == len(page.reading_order)
        assert page.reading_order_bboxes[0].y == 0.0
        assert page.reading_order_bboxes[1].y == 0.3

    def test_reading_order_bboxes_can_contain_none(self):
        """Some segments may have no bbox — None is valid."""
        page = make_page(
            reading_order=["Segment one", "Segment two"],
            reading_order_bboxes=[make_bbox(), None],
        )
        assert page.reading_order_bboxes[0] is not None
        assert page.reading_order_bboxes[1] is None

    def test_to_dict_serializes_bboxes(self):
        page = make_page(
            reading_order=["Hello"],
            reading_order_bboxes=[make_bbox(x=0.1, y=0.2)],
        )
        d = page.to_dict()
        assert "reading_order_bboxes" in d
        assert d["reading_order_bboxes"][0]["x"] == 0.1
        assert d["reading_order_bboxes"][0]["y"] == 0.2

    def test_to_dict_handles_none_bboxes(self):
        page = make_page(
            reading_order=["Hello"],
            reading_order_bboxes=[None],
        )
        d = page.to_dict()
        assert d["reading_order_bboxes"] == [None]


# ===========================================================================
# B2 — Layout elements: checkboxes and signatures
# ===========================================================================

class TestExtractCheckboxes:

    def _make_docling_doc(self, elements):
        doc = MagicMock()
        doc.texts = elements
        return doc

    def _make_element(self, label, page_no, text=""):
        elem = MagicMock()
        elem.label = label
        elem.text = text
        prov = MagicMock()
        prov.page_no = page_no
        prov.bbox = MagicMock(l=10, t=20, r=100, b=50)
        elem.prov = [prov]
        return elem

    def _make_page_size(self, w=595.0, h=842.0):
        ps = MagicMock()
        ps.width = w
        ps.height = h
        return ps

    def test_checked_checkbox_detected(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("checkbox_selected", page_no=1, text="I agree")
        doc = self._make_docling_doc([elem])
        result = extract_checkboxes(doc, page_num=1, page_size=self._make_page_size())
        assert len(result) == 1
        assert result[0].element_type == "checkbox"
        assert result[0].state == "checked"
        assert result[0].text == "I agree"

    def test_unchecked_checkbox_detected(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("checkbox_unselected", page_no=1)
        doc = self._make_docling_doc([elem])
        result = extract_checkboxes(doc, page_num=1, page_size=self._make_page_size())
        assert len(result) == 1
        assert result[0].state == "unchecked"

    def test_non_checkbox_element_ignored(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("paragraph", page_no=1)
        doc = self._make_docling_doc([elem])
        result = extract_checkboxes(doc, page_num=1, page_size=self._make_page_size())
        assert result == []

    def test_wrong_page_ignored(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("checkbox_selected", page_no=2)
        doc = self._make_docling_doc([elem])
        result = extract_checkboxes(doc, page_num=1, page_size=self._make_page_size())
        assert result == []

    def test_bbox_normalized_correctly(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("checkbox_selected", page_no=1)
        # bbox: l=10, t=20, r=110, b=70  on 595x842 page
        elem.prov[0].bbox = MagicMock(l=10, t=20, r=110, b=70)
        doc = self._make_docling_doc([elem])
        ps = self._make_page_size(w=595.0, h=842.0)
        result = extract_checkboxes(doc, page_num=1, page_size=ps)
        assert result[0].bbox is not None
        assert abs(result[0].bbox.x - 10 / 595.0) < 0.001
        assert abs(result[0].bbox.width - 100 / 595.0) < 0.001

    def test_none_page_size_yields_none_bbox(self):
        from parsers.layout_elements import extract_checkboxes
        elem = self._make_element("checkbox_selected", page_no=1)
        doc = self._make_docling_doc([elem])
        result = extract_checkboxes(doc, page_num=1, page_size=None)
        assert len(result) == 1
        assert result[0].bbox is None

    def test_missing_texts_attr_returns_empty(self):
        from parsers.layout_elements import extract_checkboxes
        doc = MagicMock(spec=[])   # no 'texts' attribute
        result = extract_checkboxes(doc, page_num=1, page_size=None)
        assert result == []

    def test_bad_element_does_not_raise(self):
        from parsers.layout_elements import extract_checkboxes
        bad_elem = MagicMock()
        bad_elem.label = "checkbox_selected"
        bad_elem.prov = None   # will cause AttributeError on prov[0]
        doc = self._make_docling_doc([bad_elem])
        result = extract_checkboxes(doc, page_num=1, page_size=None)
        assert result == []   # silently skipped


class TestExtractSignatureRegions:

    def _make_docling_doc(self, elements):
        doc = MagicMock()
        doc.texts = elements
        return doc

    def _make_element(self, label, page_no, text=""):
        elem = MagicMock()
        elem.label = label
        elem.text = text
        prov = MagicMock()
        prov.page_no = page_no
        prov.bbox = MagicMock(l=10, t=700, r=300, b=750)
        elem.prov = [prov]
        return elem

    def test_docling_signature_label_detected(self):
        from parsers.layout_elements import extract_signature_regions
        elem = self._make_element("signature", page_no=1, text="John Doe")
        doc = self._make_docling_doc([elem])
        result = extract_signature_regions(doc, 1, None, "")
        assert len(result) == 1
        assert result[0].element_type == "signature"
        assert result[0].state == "signed"
        assert result[0].confidence == 0.75

    def test_handwritten_text_label_detected(self):
        from parsers.layout_elements import extract_signature_regions
        elem = self._make_element("handwritten_text", page_no=1)
        doc = self._make_docling_doc([elem])
        result = extract_signature_regions(doc, 1, None, "")
        assert len(result) == 1
        assert result[0].state == "signed"

    def test_keyword_hint_fallback(self):
        from parsers.layout_elements import extract_signature_regions
        doc = MagicMock()
        doc.texts = []
        result = extract_signature_regions(
            doc, 1, None, "Please sign here to confirm your agreement."
        )
        assert len(result) == 1
        assert result[0].state == "unknown"
        assert result[0].confidence == 0.5

    def test_no_signal_returns_empty(self):
        from parsers.layout_elements import extract_signature_regions
        doc = MagicMock()
        doc.texts = []
        result = extract_signature_regions(doc, 1, None, "Regular paragraph text here.")
        assert result == []

    def test_label_takes_priority_over_hint(self):
        """When both label and text hint present, label wins (only one entry returned)."""
        from parsers.layout_elements import extract_signature_regions
        elem = self._make_element("signature", page_no=1)
        doc = self._make_docling_doc([elem])
        result = extract_signature_regions(
            doc, 1, None, "Please sign here."
        )
        assert len(result) == 1
        assert result[0].state == "signed"   # label, not hint


# ===========================================================================
# B3 — ImageElement extensions and figure chunks
# ===========================================================================

class TestImageElementExtensions:

    def test_new_fields_have_defaults(self):
        from core.document import ImageElement
        img = ImageElement(
            page_num=1,
            image_ref="page_1_figure_1",
            ocr_text="",
            description="",
            chunk_type="figure",
            vision_prompt_used="invoice",
        )
        assert img.bbox is None
        assert img.caption == ""
        assert img.element_type == "figure"

    def test_to_dict_includes_new_fields(self):
        from core.document import ImageElement
        img = ImageElement(
            page_num=1,
            image_ref="page_1_figure_1",
            ocr_text="",
            description="A bar chart showing revenue.",
            chunk_type="figure",
            vision_prompt_used="invoice",
            bbox=make_bbox(),
            caption="Figure 1: Revenue Q1-Q4",
            element_type="figure",
        )
        d = img.to_dict()
        assert "bbox" in d
        assert d["bbox"] is not None
        assert d["caption"] == "Figure 1: Revenue Q1-Q4"
        assert d["element_type"] == "figure"

    def test_to_dict_bbox_none_when_not_set(self):
        from core.document import ImageElement
        img = ImageElement(
            page_num=1, image_ref="ref", ocr_text="", description="",
            chunk_type="figure", vision_prompt_used="",
        )
        assert img.to_dict()["bbox"] is None

    def test_chunk_type_figure_preserved(self):
        from core.document import ImageElement
        img = ImageElement(
            page_num=1, image_ref="ref", ocr_text="", description="desc",
            chunk_type="figure", vision_prompt_used="",
        )
        assert img.chunk_type == "figure"


class TestExtractFigures:

    def _make_picture(self, page_no, caption=""):
        pic = MagicMock()
        prov = MagicMock()
        prov.page_no = page_no
        prov.bbox = MagicMock(l=50, t=100, r=400, b=350)
        pic.prov = [prov]
        pic.caption = caption
        return pic

    def _make_docling_doc(self, pictures):
        doc = MagicMock()
        doc.pictures = pictures
        return doc

    def _make_page_size(self):
        ps = MagicMock()
        ps.width = 595.0
        ps.height = 842.0
        return ps

    def test_figure_detected(self):
        from parsers.layout_elements import extract_figures
        pic = self._make_picture(page_no=1, caption="Figure 1")
        doc = self._make_docling_doc([pic])
        result = extract_figures(doc, None, 1, self._make_page_size())
        assert len(result) == 1
        assert result[0].element_type == "figure"
        assert result[0].chunk_type == "figure"
        assert result[0].caption == "Figure 1"

    def test_image_ref_numbered_correctly(self):
        from parsers.layout_elements import extract_figures
        pics = [self._make_picture(1), self._make_picture(1)]
        doc = self._make_docling_doc(pics)
        result = extract_figures(doc, None, 1, self._make_page_size())
        assert result[0].image_ref == "page_1_figure_1"
        assert result[1].image_ref == "page_1_figure_2"

    def test_wrong_page_figure_skipped(self):
        from parsers.layout_elements import extract_figures
        pic = self._make_picture(page_no=2)
        doc = self._make_docling_doc([pic])
        result = extract_figures(doc, None, 1, self._make_page_size())
        assert result == []

    def test_description_starts_empty(self):
        """description must be empty — filled later by vision engine."""
        from parsers.layout_elements import extract_figures
        pic = self._make_picture(page_no=1)
        doc = self._make_docling_doc([pic])
        result = extract_figures(doc, None, 1, self._make_page_size())
        assert result[0].description == ""

    def test_no_pictures_attr_returns_empty(self):
        from parsers.layout_elements import extract_figures
        doc = MagicMock(spec=[])
        result = extract_figures(doc, None, 1, None)
        assert result == []

    def test_bbox_populated(self):
        from parsers.layout_elements import extract_figures
        pic = self._make_picture(page_no=1)
        doc = self._make_docling_doc([pic])
        result = extract_figures(doc, None, 1, self._make_page_size())
        assert result[0].bbox is not None
        assert 0.0 <= result[0].bbox.x <= 1.0

    def test_bad_picture_does_not_raise(self):
        from parsers.layout_elements import extract_figures
        bad = MagicMock()
        bad.prov = None
        doc = self._make_docling_doc([bad])
        result = extract_figures(doc, None, 1, None)
        assert result == []


# ===========================================================================
# format_source in retrieval
# ===========================================================================

class TestFormatSource:

    def _make_chunk(self, chunk_type="text", image_ref=None,
                    caption="", bbox=None):
        return {
            "chunk_num":  1,
            "page":       "3",
            "file":       "report.pdf",
            "content":    "This is the chunk content for testing purposes.",
            "chunk_type": chunk_type,
            "image_ref":  image_ref,
            "caption":    caption,
            "bbox":       bbox,
        }

    def test_text_chunk_base_fields(self):
        from retrieval import format_source
        result = format_source(self._make_chunk("text"))
        assert result["chunk_type"] == "text"
        assert result["page"] == "3"
        assert result["file"] == "report.pdf"
        assert "preview" in result
        assert "exact_sentence" in result
        assert "image_ref" not in result
        assert "caption" not in result
        assert "bbox" not in result

    def test_figure_chunk_has_extra_fields(self):
        from retrieval import format_source
        bbox = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.3,
                "page": 1, "page_width": 595.0, "page_height": 842.0}
        result = format_source(self._make_chunk(
            chunk_type="figure",
            image_ref="page_3_figure_1",
            caption="Figure 1: Revenue chart",
            bbox=bbox,
        ))
        assert result["chunk_type"] == "figure"
        assert result["image_ref"] == "page_3_figure_1"
        assert result["caption"] == "Figure 1: Revenue chart"
        assert result["bbox"] == bbox

    def test_description_chunk_has_image_ref(self):
        from retrieval import format_source
        result = format_source(self._make_chunk(
            chunk_type="description",
            image_ref="page_2",
        ))
        assert result["image_ref"] == "page_2"
        assert "caption" not in result
        assert "bbox" not in result

    def test_table_chunk_has_image_ref(self):
        from retrieval import format_source
        result = format_source(self._make_chunk(chunk_type="table"))
        assert "image_ref" in result

    def test_preview_truncated_to_150_chars(self):
        from retrieval import format_source
        long_content = "x" * 300
        chunk = self._make_chunk()
        chunk["content"] = long_content
        result = format_source(chunk)
        assert len(result["preview"]) == 150

    def test_exact_sentence_starts_empty(self):
        """format_source sets exact_sentence to '' — caller fills it."""
        from retrieval import format_source
        result = format_source(self._make_chunk())
        assert result["exact_sentence"] == ""

    def test_bad_chunk_does_not_raise(self):
        """format_source must never raise — even on a completely empty dict."""
        from retrieval import format_source
        result = format_source({})
        assert isinstance(result, dict)
        assert "chunk_type" in result

    def test_unknown_chunk_type_returns_base_only(self):
        from retrieval import format_source
        result = format_source(self._make_chunk(chunk_type="unknown_future_type"))
        assert result["chunk_type"] == "unknown_future_type"
        assert "caption" not in result
        assert "bbox" not in result


# ===========================================================================
# LayoutElement — state and bbox fields
# ===========================================================================

class TestLayoutElementExtensions:

    def test_state_defaults_none(self):
        from core.document import LayoutElement
        elem = LayoutElement(
            element_type="header", page_num=1, text="Header", confidence=0.9
        )
        assert elem.state is None
        assert elem.bbox is None

    def test_checkbox_state_in_to_dict(self):
        from core.document import LayoutElement
        elem = LayoutElement(
            element_type="checkbox", page_num=1, text="",
            confidence=0.85, state="checked", bbox=make_bbox(),
        )
        d = elem.to_dict()
        assert d["state"] == "checked"
        assert d["bbox"] is not None

    def test_signature_state_in_to_dict(self):
        from core.document import LayoutElement
        elem = LayoutElement(
            element_type="signature", page_num=2, text="",
            confidence=0.5, state="unknown",
        )
        d = elem.to_dict()
        assert d["state"] == "unknown"
        assert d["bbox"] is None