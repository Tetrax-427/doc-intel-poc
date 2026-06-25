"""
E1 — Stage 1 classifier exemplars.

Two signal sources per doc type:
  KEYWORD_SIGNALS      — short phrases that strongly indicate a doc type.
                         Checked against the first 500 chars of document text.
  EMBEDDING_EXEMPLARS  — short representative passages used for cosine
                         similarity against the document's first ~300 chars.

Both are static domain knowledge — no DB, no LLM, no network.

Module-level embedding cache (_exemplar_embeddings) is populated lazily
on first use and lives for the process lifetime.  If the embedding model
changes (env-var swap + restart), the cache resets automatically.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keyword signals
# ---------------------------------------------------------------------------
# Higher-specificity keywords first — more specific = higher base confidence.
# Keep lists to 8–12 entries; too many dilutes the per-hit score.

KEYWORD_SIGNALS: dict[str, list[str]] = {
    "invoice": [
        "invoice no", "invoice number", "bill to", "total amount due",
        "tax invoice", "gst invoice", "invoice date", "payment due",
        "subtotal", "amount payable",
    ],
    "receipt": [
        "receipt no", "receipt number", "amount paid", "payment received",
        "cash receipt", "transaction id", "paid by",
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
    "nda": [
        "non-disclosure", "confidentiality agreement", "confidential information",
        "disclosing party", "receiving party", "trade secret",
    ],
    "loan_application": [
        "loan application", "applicant name", "loan amount", "loan purpose",
        "monthly income", "credit score", "collateral", "emi",
        "rate of interest", "repayment period",
    ],
    "bank_statement": [
        "account number", "account no", "statement period",
        "opening balance", "closing balance", "transaction date",
        "available balance", "ifsc", "sort code", "routing number",
    ],
    "gst_return": [
        "gstr", "gst return", "gstin", "outward supplies", "inward supplies",
        "input tax credit", "integrated tax", "central tax", "state tax",
    ],
    "offer_letter": [
        "offer letter", "job offer", "compensation package", "start date",
        "position offered", "reporting to", "ctc",
    ],
    "medical": [
        "patient name", "date of birth", "diagnosis", "prescription",
        "dosage", "physician", "medical record", "icd", "rx",
    ],
    "legal": [
        "plaintiff", "defendant", "court of", "whereas", "petitioner",
        "respondent", "affidavit", "hereby ordered", "jurisdiction",
    ],
}

# ---------------------------------------------------------------------------
# Embedding exemplars
# ---------------------------------------------------------------------------

EMBEDDING_EXEMPLARS: dict[str, str] = {
    "invoice": (
        "Invoice No: INV-2024-001\nBill To: ABC Corporation\n"
        "Invoice Date: 01-Jan-2024\nDue Date: 31-Jan-2024\n"
        "Item: Professional Services\nAmount: ₹45,000\nTotal: ₹45,000"
    ),
    "receipt": (
        "Receipt No: RCP-001\nAmount Paid: ₹5,000\nPayment Received from: John Doe\n"
        "Date: 01-Jan-2024\nCash Receipt\nTransaction ID: TXN12345"
    ),
    "purchase_order": (
        "Purchase Order No: PO-2024-001\nVendor: XYZ Supplies Ltd\n"
        "Ship To: Warehouse, Mumbai\nItem Description: Office Supplies\n"
        "Quantity: 100\nUnit Price: ₹500\nTotal: ₹50,000"
    ),
    "bank_statement": (
        "Account No: 1234567890\nStatement Period: Jan 2024\n"
        "Opening Balance: ₹10,000\nClosing Balance: ₹25,000\n"
        "Date | Description | Debit | Credit | Balance"
    ),
    "cv_resume": (
        "John Doe\nSoftware Engineer\nWork Experience:\n"
        "2020-Present: Senior Developer at ABC Corp\n"
        "Education: B.Tech Computer Science 2018\n"
        "Skills: Python, FastAPI, PostgreSQL"
    ),
    "contract": (
        "This Agreement is entered into as of January 1, 2024 between "
        "Party A and Party B. Whereas the parties desire to enter into "
        "this agreement, the following terms and conditions shall apply. "
        "Governing law: Laws of India."
    ),
    "nda": (
        "Non-Disclosure Agreement between Disclosing Party and Receiving Party. "
        "Confidential Information shall not be disclosed. Trade secrets protected. "
        "Effective date: January 2024."
    ),
    "loan_application": (
        "Loan Application Form\nApplicant Name: Jane Smith\n"
        "Loan Amount Requested: ₹5,00,000\nLoan Purpose: Home Renovation\n"
        "Monthly Income: ₹75,000\nRepayment Period: 36 months\nEMI: ₹15,000"
    ),
    "gst_return": (
        "GSTIN: 27AAAAA0000A1Z5\nGSTR-3B Return for January 2024\n"
        "Outward Supplies: ₹10,00,000\nInput Tax Credit: ₹80,000\n"
        "Central Tax: ₹90,000\nState Tax: ₹90,000"
    ),
    "offer_letter": (
        "Dear John Doe,\nWe are pleased to offer you the position of Software Engineer.\n"
        "Compensation Package: ₹12,00,000 CTC per annum\n"
        "Start Date: 15-Feb-2024\nReporting to: Engineering Manager"
    ),
    "medical": (
        "Patient Name: Jane Doe\nDate of Birth: 01/01/1985\n"
        "Diagnosis: Hypertension\nRx: Amlodipine 5mg\nDosage: Once daily\n"
        "Physician: Dr. Smith\nMedical Record No: MR-12345"
    ),
    "legal": (
        "In the Court of Civil Judge\nPlaintiff: ABC Corporation\n"
        "Defendant: XYZ Limited\nCase No: CIV-2024-001\n"
        "Whereas the petitioner hereby files this affidavit before the court."
    ),
    "report": (
        "Executive Summary\nThis report presents the findings of the quarterly "
        "analysis. Key metrics show a 15% improvement. Recommendations include "
        "further investment in technology infrastructure."
    ),
    "general": (
        "This document contains general information and text content "
        "that does not match a specific document type."
    ),
}

# ---------------------------------------------------------------------------
# Embedding cache (module-level, process lifetime)
# ---------------------------------------------------------------------------

_exemplar_embeddings: dict[str, list[float]] = {}


def get_exemplar_embedding(doc_type: str, get_embedding_fn) -> list[float]:
    """
    Return the cached embedding for a doc type's exemplar text.
    Computed on first access, then cached for the process lifetime.

    Args:
        doc_type:         Key into EMBEDDING_EXEMPLARS.
        get_embedding_fn: Callable(text) -> list[float].
                          Should be the same model used for document chunks.
    """
    if doc_type not in _exemplar_embeddings:
        text = EMBEDDING_EXEMPLARS.get(doc_type, "")
        _exemplar_embeddings[doc_type] = get_embedding_fn(text, model=None)
    return _exemplar_embeddings[doc_type]


def clear_exemplar_cache() -> None:
    """
    Clear the embedding cache.
    Call this if the embedding model changes at runtime (rare).
    Normal process restarts clear it automatically.
    """
    _exemplar_embeddings.clear()

# ---------------------------------------------------------------------------
# id_document — added per spec (was missing from initial implementation)
# ---------------------------------------------------------------------------

KEYWORD_SIGNALS["id_document"] = [
    "date of birth", "date of issue", "date of expiry", "nationality",
    "passport no", "driving licence", "aadhaar", "pan card",
    "voter id", "national id",
]

EMBEDDING_EXEMPLARS["id_document"] = (
    "Name: John Doe\nDate of Birth: 01/01/1990\nNationality: Indian\n"
    "Passport No: A1234567\nDate of Issue: 01/01/2020\n"
    "Date of Expiry: 31/12/2029"
)