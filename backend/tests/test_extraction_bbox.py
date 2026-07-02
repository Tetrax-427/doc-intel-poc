"""
tests/test_extraction_bbox.py

Tests for E2 — find_field_bbox() and the enriched extract_fields() shape.

Self-contained — no network, no DB, no LLM.
Run: pytest tests/test_extraction_bbox.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Inline find_field_bbox for isolated testing
# ---------------------------------------------------------------------------

def find_field_bbox(field_value, chunks: list) -> dict | None:
    if not field_value or len(field_value.strip()) < 4:
        return None
    value_lower = field_value.strip().lower()
    matching = [
        c for c in chunks
        if value_lower in c.get("content", "").lower()
        and c.get("metadata", {}).get("bbox") is not None
    ]
    if not matching:
        return None
    def match_pos(chunk):
        idx = chunk["content"].lower().find(value_lower)
        return idx if idx >= 0 else 9999
    return min(matching, key=match_pos)["metadata"]["bbox"]


def make_chunk(content: str, bbox: dict | None, page: str = "1") -> dict:
    return {
        "content":  content,
        "metadata": {
            "page": page, "file": "test.pdf",
            "chunk_type": "text", "bbox": bbox,
            "chunk_level": "flat",
        },
    }


BBOX_1 = {"x": 0.1, "y": 0.05, "width": 0.4, "height": 0.1,
           "page": 1, "page_width": 612, "page_height": 792}
BBOX_2 = {"x": 0.6, "y": 0.8,  "width": 0.35, "height": 0.05,
           "page": 1, "page_width": 612, "page_height": 792}

CHUNKS_WITH_BBOX = [
    make_chunk("Invoice No: INV-001\nBill To: ABC Corp", BBOX_1),
    make_chunk("Total Amount Due: ₹45,000\nPayment Due: 31-Jan-2024", BBOX_2),
    make_chunk("Terms and Conditions apply.", None),
]

CHUNKS_NO_BBOX = [
    make_chunk("Invoice No: INV-001", None),
    make_chunk("Total: ₹45,000", None),
]


# ---------------------------------------------------------------------------
# find_field_bbox tests
# ---------------------------------------------------------------------------

def test_find_bbox_exact_match():
    result = find_field_bbox("INV-001", CHUNKS_WITH_BBOX)
    assert result is not None
    assert result["page"] == 1
    assert result["x"] == 0.1
    print("  exact match found bbox ✓")


def test_find_bbox_partial_match():
    result = find_field_bbox("₹45,000", CHUNKS_WITH_BBOX)
    assert result is not None
    assert abs(result["x"] - 0.6) < 1e-6
    print(f"  partial match found bbox x={result['x']} ✓")


def test_find_bbox_no_match_returns_none():
    result = find_field_bbox("XYZ-999-not-in-any-chunk", CHUNKS_WITH_BBOX)
    assert result is None
    print("  no match → None ✓")


def test_find_bbox_none_value_returns_none():
    assert find_field_bbox(None, CHUNKS_WITH_BBOX) is None
    print("  None value → None ✓")


def test_find_bbox_empty_string_returns_none():
    assert find_field_bbox("", CHUNKS_WITH_BBOX) is None
    print("  empty string → None ✓")


def test_find_bbox_short_value_returns_none():
    # Values < 4 chars are unreliable — skip bbox lookup
    assert find_field_bbox("INV", CHUNKS_WITH_BBOX) is None
    assert find_field_bbox("AB", CHUNKS_WITH_BBOX) is None
    print("  short value (< 4 chars) → None ✓")


def test_find_bbox_skips_chunks_without_bbox():
    # "Terms and Conditions" is in CHUNKS_WITH_BBOX but that chunk has bbox=None
    result = find_field_bbox("Terms and Conditions", CHUNKS_WITH_BBOX)
    assert result is None
    print("  matching chunk with no bbox → None ✓")


def test_find_bbox_no_bbox_in_any_chunk():
    # All chunks parsed by pypdf (no bbox at all)
    result = find_field_bbox("INV-001", CHUNKS_NO_BBOX)
    assert result is None
    print("  pypdf document (no bbox anywhere) → None ✓")


def test_find_bbox_case_insensitive():
    result = find_field_bbox("inv-001", CHUNKS_WITH_BBOX)
    assert result is not None
    print("  case-insensitive match ✓")


def test_find_bbox_prefers_earliest_match():
    early_bbox = {"x": 0.1, "y": 0.1, "page": 1}
    late_bbox  = {"x": 0.9, "y": 0.9, "page": 2}
    chunks = [
        make_chunk("INV-001 appears first here", early_bbox),
        make_chunk("See also INV-001 at the end of the document", late_bbox),
    ]
    result = find_field_bbox("INV-001", chunks)
    assert result == early_bbox
    print("  prefers chunk where match appears earliest ✓")


def test_find_bbox_returns_correct_structure():
    result = find_field_bbox("INV-001", CHUNKS_WITH_BBOX)
    assert result is not None
    for key in ("x", "y", "width", "height", "page"):
        assert key in result, f"Missing bbox key: {key}"
    print("  bbox has all required keys ✓")


# ---------------------------------------------------------------------------
# Enriched extraction shape tests
# ---------------------------------------------------------------------------

def simulate_extract_fields_enriched(raw_extracted: dict, chunks: list) -> dict:
    """Simulate the E2 extract_fields() return shape."""
    enriched = {}
    for field_name, value in raw_extracted.items():
        bbox = find_field_bbox(value, chunks)
        enriched[field_name] = {"value": value, "bbox": bbox}
    return enriched


def test_enriched_shape_has_value_and_bbox():
    raw = {"invoice_number": "INV-001", "total_amount": "₹45,000"}
    result = simulate_extract_fields_enriched(raw, CHUNKS_WITH_BBOX)
    for field, data in result.items():
        assert "value" in data,  f"Missing 'value' for {field}"
        assert "bbox"  in data,  f"Missing 'bbox' for {field}"
    print("  enriched shape has value+bbox per field ✓")


def test_enriched_value_preserved():
    raw = {"invoice_number": "INV-001"}
    result = simulate_extract_fields_enriched(raw, CHUNKS_WITH_BBOX)
    assert result["invoice_number"]["value"] == "INV-001"
    print("  value preserved in enriched shape ✓")


def test_enriched_bbox_none_when_no_match():
    raw = {"mystery_field": "XYZ-value-not-in-any-chunk"}
    result = simulate_extract_fields_enriched(raw, CHUNKS_WITH_BBOX)
    assert result["mystery_field"]["bbox"] is None
    print("  bbox=None when no chunk match ✓")


def test_enriched_bbox_none_for_pypdf_doc():
    raw = {"invoice_number": "INV-001"}
    result = simulate_extract_fields_enriched(raw, CHUNKS_NO_BBOX)
    assert result["invoice_number"]["value"] == "INV-001"   # value still there
    assert result["invoice_number"]["bbox"]  is None        # no bbox
    print("  pypdf doc: value extracted but bbox=None ✓")


def test_enriched_bbox_not_none_when_found():
    raw = {"invoice_number": "INV-001"}
    result = simulate_extract_fields_enriched(raw, CHUNKS_WITH_BBOX)
    assert result["invoice_number"]["bbox"] is not None
    print("  bbox populated when chunk match found ✓")


def pytest_approx(value, abs=1e-6):
    """Minimal approx helper for standalone runs (no pytest needed)."""
    class _Approx:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) < 1e-6
    return _Approx(value)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_find_bbox_exact_match,
        test_find_bbox_partial_match,
        test_find_bbox_no_match_returns_none,
        test_find_bbox_none_value_returns_none,
        test_find_bbox_empty_string_returns_none,
        test_find_bbox_short_value_returns_none,
        test_find_bbox_skips_chunks_without_bbox,
        test_find_bbox_no_bbox_in_any_chunk,
        test_find_bbox_case_insensitive,
        test_find_bbox_prefers_earliest_match,
        test_find_bbox_returns_correct_structure,
        test_enriched_shape_has_value_and_bbox,
        test_enriched_value_preserved,
        test_enriched_bbox_none_when_no_match,
        test_enriched_bbox_none_for_pypdf_doc,
        test_enriched_bbox_not_none_when_found,
    ]
    passed = failed = 0
    for t in tests:
        try:
            print(f"\n▶ {t.__name__}")
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    print(f"\n{'='*50}\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
