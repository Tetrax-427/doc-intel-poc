"""
tests/test_comparison.py

Unit tests for backend/comparison.py.

Uses lightweight stand-in Document/DocumentPage classes so these tests run
without depending on real parsers — swap in your actual core.document
classes if you'd rather test against the real dataclasses directly.
"""

from dataclasses import dataclass, field
from typing import List

from comparison import diff_documents, normalize, tokenize


@dataclass
class FakePage:
    page_number: int
    text: str
    position_type: str = "page"


@dataclass
class FakeDocument:
    pages: List[FakePage] = field(default_factory=list)


def make_doc(pages_text, position_type="page"):
    return FakeDocument(pages=[
        FakePage(page_number=i + 1, text=t, position_type=position_type)
        for i, t in enumerate(pages_text)
    ])


def test_normalize_collapses_whitespace():
    assert normalize("hello    world\n\nfoo") == "hello world foo"


def test_tokenize_roundtrip():
    text = "The rent is $1,200 per month."
    tokens = tokenize(text)
    assert "".join(tokens) == text


def test_identical_documents_have_no_diff():
    doc_a = make_doc(["The rent is $1,200 per month."])
    doc_b = make_doc(["The rent is $1,200 per month."])
    result = diff_documents(doc_a, doc_b)

    assert all(s.type == "unchanged" for s in result.segments)
    assert result.stats["additions"] == 0
    assert result.stats["removals"] == 0


def test_single_word_replacement_is_isolated():
    doc_a = make_doc(["The rent is $1,200 per month."])
    doc_b = make_doc(["The rent is $1,450 per month."])
    result = diff_documents(doc_a, doc_b)

    removed = [s.text for s in result.segments if s.type == "removed"]
    added = [s.text for s in result.segments if s.type == "added"]

    assert removed == ["$1,200"]
    assert added == ["$1,450"]
    # surrounding text should stay unchanged, not get swept into the diff
    assert any(s.type == "unchanged" and "per month" in s.text for s in result.segments)


def test_pure_addition():
    doc_a = make_doc(["Section 1: Intro."])
    doc_b = make_doc(["Section 1: Intro. Section 2: New clause."])
    result = diff_documents(doc_a, doc_b)

    added = "".join(s.text for s in result.segments if s.type == "added")
    assert "Section 2: New clause." in added
    assert result.stats["removals"] == 0
    assert result.stats["additions"] > 0


def test_pure_deletion():
    doc_a = make_doc(["Section 1: Intro. Section 2: Old clause."])
    doc_b = make_doc(["Section 1: Intro."])
    result = diff_documents(doc_a, doc_b)

    removed = "".join(s.text for s in result.segments if s.type == "removed")
    assert "Section 2: Old clause." in removed
    assert result.stats["additions"] == 0
    assert result.stats["removals"] > 0


def test_whitespace_only_differences_are_ignored():
    doc_a = make_doc(["Hello   world"])
    doc_b = make_doc(["Hello world"])
    result = diff_documents(doc_a, doc_b)

    assert all(s.type == "unchanged" for s in result.segments)


def test_page_numbers_are_tracked_across_pages():
    doc_a = make_doc(["Page one text.", "Page two original."])
    doc_b = make_doc(["Page one text.", "Page two changed."])
    result = diff_documents(doc_a, doc_b)

    removed_segments = [s for s in result.segments if s.type == "removed"]
    added_segments = [s for s in result.segments if s.type == "added"]

    assert removed_segments and removed_segments[0].page_a == 2
    assert added_segments and added_segments[0].page_b == 2


def test_docx_uses_paragraph_position_type():
    doc_a = make_doc(["Para one.", "Para two."], position_type="paragraph")
    doc_b = make_doc(["Para one.", "Para two changed."], position_type="paragraph")
    result = diff_documents(doc_a, doc_b)

    assert result.position_type == "paragraph"


def test_mismatched_position_types_reported_as_mixed():
    doc_a = make_doc(["Page text."], position_type="page")
    doc_b = make_doc(["Para text."], position_type="paragraph")
    result = diff_documents(doc_a, doc_b)

    assert result.position_type == "mixed"