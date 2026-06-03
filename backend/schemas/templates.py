"""
schemas/templates.py
Extraction template definitions + doc-type → template mapping.

Public API (used by routers and retrieval):
    list_templates()                    → list of {id, label, description, schema}
    get_template(template_id)           → single template dict or None
    get_template_for_doc_type(doc_type) → template ID string
    TEMPLATE_MAP                        → raw dict for direct lookups
"""

# ── Template definitions ──────────────────────────────────────────────────────

_TEMPLATES = [
    {
        "id": "invoice",
        "label": "🧾 Invoice / Receipt",
        "description": "Extract billing details, line items, totals, and payment terms.",
        "schema": {
            "vendor_name": "Name of the company or person issuing the invoice",
            "vendor_address": "Vendor address",
            "client_name": "Name of the billed party",
            "invoice_number": "Invoice or receipt number",
            "invoice_date": "Date the invoice was issued",
            "due_date": "Payment due date",
            "subtotal": "Amount before tax",
            "tax_amount": "Tax amount",
            "total_amount": "Total amount due including tax",
            "payment_terms": "Payment terms (e.g. Net 30)",
            "line_items": "List of items/services billed as a list",
        },
    },
    {
        "id": "cv_resume",
        "label": "👤 CV / Resume",
        "description": "Extract candidate profile, skills, experience, and education.",
        "schema": {
            "full_name": "Candidate's full name",
            "email": "Email address",
            "phone": "Phone number",
            "location": "City and country",
            "linkedin": "LinkedIn profile URL if present",
            "current_title": "Most recent job title",
            "current_company": "Most recent employer",
            "total_experience_years": "Total years of professional experience",
            "skills": "List of technical and professional skills as a list",
            "education": "Highest qualification — degree, institution, year",
            "languages": "Languages spoken as a list",
            "summary": "Professional summary or objective if present",
        },
    },
    {
        "id": "contract",
        "label": "📑 Contract / Agreement / NDA",
        "description": "Extract parties, obligations, dates, and key clauses.",
        "schema": {
            "party_1": "First party name",
            "party_2": "Second party name",
            "agreement_type": "Type of agreement (e.g. NDA, Service Agreement)",
            "effective_date": "Date the contract takes effect",
            "expiry_date": "Expiry or termination date if specified",
            "governing_law": "Jurisdiction or governing law",
            "key_obligations": "Main obligations of each party as a list",
            "payment_terms": "Payment terms if applicable",
            "confidentiality_clause": "Summary of confidentiality terms if present",
            "termination_clause": "Conditions under which the contract can be terminated",
        },
    },
    {
        "id": "bank_statement",
        "label": "💰 Bank Statement / Financial",
        "description": "Extract account details, balances, and transaction summary.",
        "schema": {
            "account_holder": "Name of the account holder",
            "account_number": "Account number (last 4 digits if masked)",
            "bank_name": "Name of the bank",
            "statement_period": "Period covered by the statement",
            "opening_balance": "Opening balance",
            "closing_balance": "Closing balance",
            "total_credits": "Total credits during the period",
            "total_debits": "Total debits during the period",
            "currency": "Currency of the account",
        },
    },
    {
        "id": "gst_return",
        "label": "🧮 GST Return",
        "description": "Extract GST registration, filing period, and tax liability.",
        "schema": {
            "gstin": "GST Identification Number",
            "legal_name": "Legal name of the taxpayer",
            "trade_name": "Trade name if different",
            "filing_period": "Month and year of the return",
            "return_type": "Type of return (GSTR-1, GSTR-3B, etc.)",
            "taxable_turnover": "Total taxable turnover",
            "total_tax_liability": "Total tax liability (CGST + SGST + IGST)",
            "tax_paid": "Tax paid via cash/ITC",
            "late_fee": "Late fee if applicable",
        },
    },
    {
        "id": "offer_letter",
        "label": "📨 Offer Letter",
        "description": "Extract job offer details, compensation, and joining date.",
        "schema": {
            "candidate_name": "Name of the candidate",
            "job_title": "Offered job title",
            "department": "Department",
            "reporting_to": "Reporting manager or role",
            "joining_date": "Expected joining date",
            "base_salary": "Annual or monthly base salary",
            "total_ctc": "Total Cost to Company",
            "probation_period": "Probation period if mentioned",
            "work_location": "Office location or remote/hybrid",
            "offer_expiry": "Date by which the offer must be accepted",
        },
    },
    {
        "id": "loan_application",
        "label": "🏦 Loan / Credit Application",
        "description": "Extract applicant details, loan amount, and terms.",
        "schema": {
            "applicant_name": "Full name of applicant",
            "applicant_dob": "Date of birth",
            "loan_type": "Type of loan (personal, home, auto, etc.)",
            "loan_amount_requested": "Amount applied for",
            "loan_tenure": "Requested repayment tenure",
            "annual_income": "Declared annual income",
            "employer_name": "Current employer",
            "employment_type": "Salaried, self-employed, etc.",
            "existing_loans": "Existing loan obligations if mentioned",
            "collateral": "Collateral offered if any",
        },
    },
    {
        "id": "general",
        "label": "📄 General Document",
        "description": "General-purpose extraction for unclassified documents.",
        "schema": {
            "title": "Document title or heading",
            "author": "Author or issuing party",
            "date": "Document date",
            "summary": "One paragraph summary of the document",
            "key_points": "Main points or conclusions as a list",
            "entities": "People, companies, or organisations mentioned as a list",
            "amounts": "Any monetary amounts mentioned as a list",
            "dates": "Any important dates mentioned as a list",
        },
    },
    {
        "id": "custom",
        "label": "✏️ Custom Schema",
        "description": "Define your own fields.",
        "schema": {
            "field_name": "description of what to extract",
            "another_field": "description of this field",
        },
    },
]

# Index by ID for O(1) lookup
_TEMPLATE_INDEX = {t["id"]: t for t in _TEMPLATES}


# ── Public functions ──────────────────────────────────────────────────────────

def list_templates() -> list[dict]:
    """Return all templates (id, label, description, schema)."""
    return _TEMPLATES


def get_template(template_id: str) -> dict | None:
    """Return a single template by ID, or None if not found."""
    return _TEMPLATE_INDEX.get(template_id)


# ── TEMPLATE_MAP — moved here from retrieval.py ───────────────────────────────
# Maps every LLM-returned doc_type string to a template ID.
# LLM output is normalised to lowercase before lookup.
# Any unknown type falls back to "custom" via get_template_for_doc_type().

TEMPLATE_MAP: dict[str, str] = {
    # Invoice / billing
    "invoice":                  "invoice",
    "receipt":                  "invoice",
    "bill":                     "invoice",
    "purchase order":           "invoice",

    # CV / resume
    "resume":                   "cv_resume",
    "cv":                       "cv_resume",
    "curriculum vitae":         "cv_resume",

    # Contracts
    "contract":                 "contract",
    "agreement":                "contract",
    "nda":                      "contract",
    "mou":                      "contract",
    "memorandum of understanding": "contract",

    # Financial / banking
    "financial statement":      "bank_statement",
    "balance sheet":            "bank_statement",
    "income statement":         "bank_statement",
    "bank statement":           "bank_statement",
    "profit and loss":          "bank_statement",

    # GST / tax
    "gst return":               "gst_return",
    "gstr-1":                   "gst_return",
    "gstr-3b":                  "gst_return",
    "tax return":               "gst_return",

    # Employment
    "offer letter":             "offer_letter",
    "appointment letter":       "offer_letter",
    "employment contract":      "offer_letter",

    # Loans
    "loan application":         "loan_application",
    "credit application":       "loan_application",
    "mortgage application":     "loan_application",

    # Reports (no specific template — use general)
    "report":                   "general",
    "research paper":           "general",
    "annual report":            "general",

    # Medical (no specific template yet)
    "medical record":           "general",
    "prescription":             "general",

    # Legal (no specific template yet)
    "legal document":           "general",
    "court filing":             "general",

    # Fallback categories → custom
    "article":                  "custom",
    "email":                    "custom",
    "letter":                   "custom",
    "general":                  "custom",
}


def get_template_for_doc_type(doc_type: str) -> str:
    """
    Return the schema template ID for a given doc_type string.
    Normalises to lowercase before lookup.
    Falls back to 'custom' for any unknown type.

    Example:
        get_template_for_doc_type("Invoice")   → "invoice"
        get_template_for_doc_type("unknown")   → "custom"
    """
    return TEMPLATE_MAP.get(doc_type.lower().strip(), "custom")