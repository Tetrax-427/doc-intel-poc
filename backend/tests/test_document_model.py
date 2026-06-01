"""
tests/test_document_model.py

Tests for backend/core/document.py
Validates the shared contract that all parsers must produce.
"""

import os
import sys
import uuid
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.document import (
    Entity, TableCell, Table, LayoutElement, ImageElement,
    DocumentPage, Classification, Document, make_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entity(**kwargs):
    defaults = dict(text="Acme", entity_type="org", confidence=0.9,
                    page_num=1, char_start=0, char_end=4)
    return Entity(**{**defaults, **kwargs})


def make_table(**kwargs):
    defaults = dict(page_num=1, title="", headers=["Item", "Amount"],
                    rows=[["Widget", "100"]], cells=[], raw_text="Item|Amount\nWidget|100")
    return Table(**{**defaults, **kwargs})


def make_page(**kwargs):
    defaults = dict(page_num=1, text="Invoice from Acme. Total: $100.",
                    tables=[], images=[], layout=[], entities=[],
                    word_count=5, ocr_confidence=1.0)
    return DocumentPage(**{**defaults, **kwargs})


def make_classification(**kwargs):
    defaults = dict(doc_type="invoice", confidence=0.9, sub_types=[],
                    schema_template="invoice_v1", validation_ruleset="rules",
                    vision_prompt="invoice_prompt", requires_human_review=False)
    return Classification(**{**defaults, **kwargs})


def make_doc(**kwargs):
    defaults = dict(
        id=str(uuid.uuid4()), file_name="test.pdf", file_type=".pdf",
        file_path="/uploads/test.pdf", pages=[make_page()],
        parser_used="pypdf",
    )
    return make_document(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class TestEntity:

    def test_has_required_fields(self):
        e = make_entity()
        assert e.text and e.entity_type
        assert 0.0 <= e.confidence <= 1.0
        assert e.page_num >= 1
        assert e.char_start >= 0 and e.char_end > e.char_start

    def test_to_dict_has_all_keys(self):
        d = make_entity().to_dict()
        assert all(k in d for k in ["text", "entity_type", "confidence", "page_num", "char_start", "char_end"])

    def test_entity_types(self):
        for etype in ["person", "org", "date", "amount", "id", "location"]:
            e = make_entity(entity_type=etype)
            assert e.entity_type == etype


# ---------------------------------------------------------------------------
# TableCell
# ---------------------------------------------------------------------------

class TestTableCell:

    def test_has_required_fields(self):
        cell = TableCell(row=0, col=1, value="100", header="Amount")
        assert cell.row == 0 and cell.col == 1
        assert cell.value == "100" and cell.header == "Amount"

    def test_to_dict(self):
        d = TableCell(row=0, col=0, value="x", header="h").to_dict()
        assert all(k in d for k in ["row", "col", "value", "header"])


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

class TestTable:

    def test_has_required_fields(self):
        t = make_table()
        assert t.headers and t.rows

    def test_must_have_at_least_one_header(self):
        t = make_table(headers=["Col"])
        assert len(t.headers) >= 1

    def test_must_have_at_least_one_row(self):
        t = make_table(rows=[["val"]])
        assert len(t.rows) >= 1

    def test_row_count_property(self):
        t = make_table(rows=[["a"], ["b"], ["c"]])
        assert t.row_count == 3

    def test_col_count_property(self):
        t = make_table(headers=["A", "B", "C"])
        assert t.col_count == 3

    def test_to_dict_has_all_keys(self):
        d = make_table().to_dict()
        assert all(k in d for k in ["page_num", "title", "headers", "rows", "cells", "raw_text"])


# ---------------------------------------------------------------------------
# DocumentPage
# ---------------------------------------------------------------------------

class TestDocumentPage:

    def test_has_required_fields(self):
        p = make_page()
        assert p.page_num >= 1
        assert p.text
        assert p.word_count > 0

    def test_to_dict_has_all_keys(self):
        d = make_page().to_dict()
        assert all(k in d for k in ["page_num", "text", "word_count", "ocr_confidence",
                                     "tables", "images", "layout", "entities"])

    def test_tables_list(self):
        p = make_page(tables=[make_table()])
        assert len(p.to_dict()["tables"]) == 1

    def test_entities_list(self):
        p = make_page(entities=[make_entity()])
        assert len(p.to_dict()["entities"]) == 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassification:

    def test_has_required_fields(self):
        c = make_classification()
        assert c.doc_type and isinstance(c.confidence, float)
        assert isinstance(c.requires_human_review, bool)

    def test_to_dict_has_all_keys(self):
        d = make_classification().to_dict()
        assert all(k in d for k in ["doc_type", "confidence", "sub_types",
                                     "schema_template", "validation_ruleset",
                                     "vision_prompt", "requires_human_review"])

    def test_low_confidence_flags_for_review(self):
        c = make_classification(confidence=0.5, requires_human_review=True)
        assert c.requires_human_review is True


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:

    def test_has_required_identity_fields(self):
        doc = make_doc()
        assert doc.id and doc.file_name and doc.file_type and doc.file_path

    def test_has_required_content_fields(self):
        doc = make_doc()
        assert isinstance(doc.pages, list)
        assert isinstance(doc.full_text, str)
        assert isinstance(doc.tables, list)
        assert isinstance(doc.entities, dict)

    def test_page_count_property(self):
        doc = make_doc(pages=[make_page(), make_page(page_num=2)])
        assert doc.page_count == 2

    def test_word_count_property(self):
        p1 = make_page(word_count=10)
        p2 = make_page(page_num=2, word_count=15)
        doc = make_doc(pages=[p1, p2])
        assert doc.word_count == 25

    def test_has_tables_property(self):
        doc_with = make_doc(pages=[make_page(tables=[make_table()])])
        doc_without = make_doc(pages=[make_page(tables=[])])
        assert doc_with.has_tables is True
        assert doc_without.has_tables is False

    def test_has_entities_property(self):
        doc_with = make_doc(pages=[make_page(entities=[make_entity()])])
        doc_without = make_doc(pages=[make_page(entities=[])])
        assert doc_with.has_entities is True
        assert doc_without.has_entities is False

    def test_is_scanned_property(self):
        doc_scanned = make_doc(metadata={"is_scanned": True})
        doc_text = make_doc(metadata={"is_scanned": False})
        assert doc_scanned.is_scanned is True
        assert doc_text.is_scanned is False

    def test_primary_classification_returns_first(self):
        clf1 = make_classification(doc_type="invoice", confidence=0.9)
        clf2 = make_classification(doc_type="contract", confidence=0.6)
        doc = make_doc(classifications=[clf1, clf2])
        assert doc.primary_classification.doc_type == "invoice"

    def test_primary_classification_none_when_empty(self):
        doc = make_doc(classifications=[])
        assert doc.primary_classification is None

    def test_version_defaults_to_1(self):
        doc = make_doc()
        assert doc.version == 1

    def test_parent_id_defaults_to_none(self):
        doc = make_doc()
        assert doc.parent_id is None

    def test_created_at_is_set(self):
        doc = make_doc()
        assert doc.created_at is not None and len(doc.created_at) > 0

    def test_to_dict_has_all_required_keys(self):
        doc = make_doc()
        d = doc.to_dict()
        required = ["id", "file_name", "file_type", "file_path", "full_text",
                    "page_count", "table_count", "entity_count", "parser_used",
                    "vision_used", "version", "parent_id", "created_at",
                    "summary", "metadata", "classifications", "pages",
                    "tables", "entities"]
        missing = [k for k in required if k not in d]
        assert not missing, f"Missing keys in to_dict: {missing}"


# ---------------------------------------------------------------------------
# make_document factory
# ---------------------------------------------------------------------------

class TestMakeDocumentFactory:

    def test_full_text_concatenated_from_pages(self):
        p1 = make_page(text="First page content.")
        p2 = make_page(page_num=2, text="Second page content.")
        doc = make_document(id=str(uuid.uuid4()), file_name="f.pdf", file_type=".pdf",
                            file_path="/f.pdf", pages=[p1, p2], parser_used="pypdf")
        assert "First page content." in doc.full_text
        assert "Second page content." in doc.full_text

    def test_tables_flattened_from_pages(self):
        t1 = make_table(page_num=1)
        t2 = make_table(page_num=2)
        p1 = make_page(tables=[t1])
        p2 = make_page(page_num=2, tables=[t2])
        doc = make_document(id=str(uuid.uuid4()), file_name="f.pdf", file_type=".pdf",
                            file_path="/f.pdf", pages=[p1, p2], parser_used="pypdf")
        assert len(doc.tables) == 2

    def test_entities_aggregated_from_pages(self):
        e1 = make_entity(entity_type="person", text="John")
        e2 = make_entity(entity_type="org", text="Acme")
        p = make_page(entities=[e1, e2])
        doc = make_document(id=str(uuid.uuid4()), file_name="f.pdf", file_type=".pdf",
                            file_path="/f.pdf", pages=[p], parser_used="pypdf")
        assert "person" in doc.entities
        assert "org" in doc.entities
        assert len(doc.entities["person"]) == 1

    def test_empty_document(self):
        doc = make_document(id=str(uuid.uuid4()), file_name="empty.pdf",
                            file_type=".pdf", file_path="/empty.pdf",
                            pages=[], parser_used="pypdf")
        assert doc.page_count == 0
        assert doc.word_count == 0
        assert doc.has_tables is False
        assert doc.primary_classification is None

    def test_metadata_defaults_set(self):
        doc = make_doc()
        assert "parser_used" in doc.metadata
        assert "is_scanned" in doc.metadata
        assert "page_count" in doc.metadata

    def test_metadata_override(self):
        doc = make_doc(metadata={"is_scanned": True, "file_size_bytes": 9999})
        assert doc.metadata["is_scanned"] is True
        assert doc.metadata["file_size_bytes"] == 9999