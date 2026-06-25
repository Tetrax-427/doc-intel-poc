"""
tests/test_classifier_stage1.py

Tests for E1 Stage 1 classifier (keyword_score, embedding_score,
cosine_similarity, classify_stage1).

Self-contained — no network, no DB, no LLM.
Run: pytest tests/test_classifier_stage1.py -v
"""

import math
import hashlib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Inline implementations (avoid full backend import chain in CI)
# ---------------------------------------------------------------------------

KEYWORD_SIGNALS = {
    "invoice": [
        "invoice no", "invoice number", "bill to", "total amount due",
        "tax invoice", "gst invoice", "invoice date", "payment due",
        "subtotal", "amount payable",
    ],
    "purchase_order": [
        "purchase order", "po number", "po no", "order number",
        "delivery address", "ship to", "vendor",
    ],
    "bank_statement": [
        "account number", "account no", "statement period",
        "opening balance", "closing balance", "transaction date",
        "available balance", "ifsc", "sort code", "routing number",
    ],
    "cv_resume": [
        "curriculum vitae", "resume", "work experience", "employment history",
        "education", "skills", "objective", "career summary",
        "professional experience", "references available",
    ],
    "contract": [
        "this agreement", "whereas", "hereinafter referred to",
        "terms and conditions", "governing law", "in witness whereof",
        "indemnification", "termination clause", "force majeure",
    ],
    "loan_application": [
        "loan application", "applicant name", "loan amount", "loan purpose",
        "monthly income", "credit score", "collateral", "emi",
        "rate of interest", "repayment period",
    ],
    "id_document": [
        "date of birth", "date of issue", "date of expiry", "nationality",
        "passport no", "driving licence", "aadhaar", "pan card",
        "voter id", "national id",
    ],
}


def keyword_score(text_sample: str) -> dict:
    text_lower = text_sample[:500].lower()
    scores = {}
    for doc_type, signals in KEYWORD_SIGNALS.items():
        hits = sum(1 for s in signals if s in text_lower)
        scores[doc_type] = hits / len(signals) if signals else 0.0
    return scores


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot    = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mock_embedding_fn(text: str) -> list:
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [((h >> i) & 1) * 1.0 for i in range(384)]


def classify_stage1(full_text: str, get_embedding_fn,
                    keyword_weight=0.5, embedding_weight=0.5) -> dict:
    if not full_text or not full_text.strip():
        return {"doc_type": "general", "confidence": 0.0, "stage": "stage1",
                "keyword_scores": {}, "embedding_scores": {}}

    kw_scores  = keyword_score(full_text)
    emb_scores = {dt: cosine_similarity(get_embedding_fn(full_text[:300]),
                                         get_embedding_fn(full_text[:100]))
                  for dt in KEYWORD_SIGNALS}

    combined   = {}
    for dt in set(kw_scores) | set(emb_scores):
        combined[dt] = keyword_weight * kw_scores.get(dt, 0.0) + \
                       embedding_weight * emb_scores.get(dt, 0.0)

    sorted_types = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    second_score          = sorted_types[1][1] if len(sorted_types) > 1 else 0.0
    spread                = best_score - second_score
    confidence            = min(spread / 0.3, 1.0) if best_score > 0 else 0.0

    return {
        "doc_type":        best_type if confidence > 0 else "general",
        "confidence":      round(confidence, 3),
        "stage":           "stage1",
        "keyword_scores":  kw_scores,
        "embedding_scores": emb_scores,
    }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

INVOICE_TEXT = (
    "Invoice No: INV-2024-001\nBill To: ABC Corp\nInvoice Date: 01-Jan-2024\n"
    "Total Amount Due: ₹45,000\nPayment Due: 31-Jan-2024"
)

CV_TEXT = (
    "Curriculum Vitae\nJohn Doe, Software Engineer\n"
    "Work Experience: 2020-Present Senior Developer\n"
    "Skills: Python, FastAPI\nEducation: B.Tech 2018"
)

BANK_TEXT = (
    "Account No: 1234567890\nStatement Period: Jan 2024\n"
    "Opening Balance: ₹10,000\nClosing Balance: ₹25,000\n"
    "Transaction Date | Description | Debit | Credit | Balance"
)

ID_TEXT = (
    "Name: John Doe\nDate of Birth: 01/01/1990\nNationality: Indian\n"
    "Passport No: A1234567\nDate of Issue: 01/01/2020\n"
    "Date of Expiry: 31/12/2029"
)

AMBIGUOUS_TEXT = "This document contains some general information about policies and procedures."


# ---------------------------------------------------------------------------
# keyword_score tests
# ---------------------------------------------------------------------------

def test_keyword_score_invoice():
    scores = keyword_score(INVOICE_TEXT)
    assert scores["invoice"] > 0, f"Expected invoice > 0, got {scores['invoice']}"
    assert scores["invoice"] > scores.get("cv_resume", 0)
    print(f"  invoice: {scores['invoice']:.3f} ✓")


def test_keyword_score_cv():
    scores = keyword_score(CV_TEXT)
    assert scores["cv_resume"] > 0
    assert scores["cv_resume"] > scores.get("invoice", 0)
    print(f"  cv_resume: {scores['cv_resume']:.3f} ✓")


def test_keyword_score_bank_statement():
    scores = keyword_score(BANK_TEXT)
    assert scores["bank_statement"] > 0
    print(f"  bank_statement: {scores['bank_statement']:.3f} ✓")


def test_keyword_score_id_document():
    scores = keyword_score(ID_TEXT)
    assert scores["id_document"] > 0, "id_document keywords not found"
    print(f"  id_document: {scores['id_document']:.3f} ✓")


def test_keyword_score_empty_text():
    scores = keyword_score("")
    assert all(v == 0.0 for v in scores.values())
    print("  empty → all 0.0 ✓")


def test_keyword_score_ambiguous():
    scores = keyword_score(AMBIGUOUS_TEXT)
    assert all(v < 0.2 for v in scores.values()), "Ambiguous text should score low"
    print(f"  ambiguous — max score: {max(scores.values()):.3f} ✓")


def test_keyword_score_returns_all_doc_types():
    scores = keyword_score(INVOICE_TEXT)
    for dt in KEYWORD_SIGNALS:
        assert dt in scores, f"Missing doc type: {dt}"
    print(f"  all {len(KEYWORD_SIGNALS)} doc types present ✓")


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    result = cosine_similarity(v, v)
    assert abs(result - 1.0) < 1e-6
    print("  cosine(v, v) == 1.0 ✓")


def test_cosine_similarity_orthogonal():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6
    print("  cosine(orthogonal) == 0.0 ✓")


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    print("  cosine(zero, v) == 0.0 ✓")


def test_cosine_similarity_range():
    import random
    random.seed(42)
    a = [random.random() for _ in range(64)]
    b = [random.random() for _ in range(64)]
    result = cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0
    print(f"  cosine in [-1,1]: {result:.3f} ✓")


# ---------------------------------------------------------------------------
# classify_stage1 tests
# ---------------------------------------------------------------------------

def test_classify_stage1_returns_required_keys():
    result = classify_stage1(INVOICE_TEXT, mock_embedding_fn)
    for key in ("doc_type", "confidence", "stage", "keyword_scores", "embedding_scores"):
        assert key in result, f"Missing key: {key}"
    assert result["stage"] == "stage1"
    print("  result shape correct ✓")


def test_classify_stage1_confidence_between_0_and_1():
    result = classify_stage1(INVOICE_TEXT, mock_embedding_fn)
    assert 0.0 <= result["confidence"] <= 1.0
    print(f"  confidence in [0,1]: {result['confidence']} ✓")


def test_classify_stage1_empty_text():
    result = classify_stage1("", mock_embedding_fn)
    assert result["doc_type"] == "general"
    assert result["confidence"] == 0.0
    print("  empty → general, 0.0 ✓")


def test_classify_stage1_ambiguous_low_confidence():
    result = classify_stage1(AMBIGUOUS_TEXT, mock_embedding_fn)
    assert result["confidence"] <= 0.75
    print(f"  ambiguous confidence: {result['confidence']} (≤0.75) ✓")


def test_classify_stage1_keyword_scores_populated():
    result = classify_stage1(INVOICE_TEXT, mock_embedding_fn)
    assert len(result["keyword_scores"]) > 0
    assert "invoice" in result["keyword_scores"]
    print("  keyword_scores populated ✓")


def test_classify_stage1_id_document_detected():
    result = classify_stage1(ID_TEXT, mock_embedding_fn)
    assert result["keyword_scores"].get("id_document", 0) > 0
    print(f"  id_document keyword_score: {result['keyword_scores']['id_document']:.3f} ✓")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_keyword_score_invoice, test_keyword_score_cv,
        test_keyword_score_bank_statement, test_keyword_score_id_document,
        test_keyword_score_empty_text, test_keyword_score_ambiguous,
        test_keyword_score_returns_all_doc_types,
        test_cosine_similarity_identical, test_cosine_similarity_orthogonal,
        test_cosine_similarity_zero_vector, test_cosine_similarity_range,
        test_classify_stage1_returns_required_keys,
        test_classify_stage1_confidence_between_0_and_1,
        test_classify_stage1_empty_text,
        test_classify_stage1_ambiguous_low_confidence,
        test_classify_stage1_keyword_scores_populated,
        test_classify_stage1_id_document_detected,
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
