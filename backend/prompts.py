"""
prompts.py — LLM prompt templates for DocIntel.

Every template is now split into two parts:
  <NAME>_SYSTEM  — the static instruction (passed as `system=` to call_llm)
  <NAME>_USER    — the per-call content template (passed as `user=` after .format())

retrieval.py imports the SYSTEM halves directly; the user half is assembled
inline at each call site since it already contains {}-format placeholders that
need to be filled with runtime values (chunks, history, question, etc.).

The old combined QA_PROMPT / QA_PROMPT_MULTI / etc. constants are kept as
aliases below to avoid breaking any code outside retrieval.py that still
imports them — they will be removed in a future cleanup pass.
"""


# ---------------------------------------------------------------------------
# QA — single document
# ---------------------------------------------------------------------------

QA_SYSTEM = (
    "You are DocIntel, a helpful AI assistant. You have access to the user's documents. "
    "Answer the user's question based ONLY on the provided document chunks.\n\n"
    "Rules:\n"
    "- Cite sources like [1], [2] after every claim\n"
    "- If the answer is not in the chunks, say exactly: "
    "\"I couldn't find this in the document.\"\n"
    "- Never make up facts not present in the chunks\n"
    "- Use bullet points for lists, proper paragraphs for explanations\n"
    "- If part of the question is general knowledge (math, definitions, etc.), "
    "answer that part from your own knowledge and the document part from the chunks.\n"
    "- Be concise but complete"
)

# user side is assembled inline in retrieval.py:
# f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# QA — multiple documents
# ---------------------------------------------------------------------------

QA_MULTI_SYSTEM = (
    "You are DocIntel, a helpful AI assistant with access to multiple documents. "
    "Answer based ONLY on the provided chunks below.\n\n"
    "Rules:\n"
    "- Cite sources like [Doc: filename, Page: X] after every claim\n"
    "- If the answer is not in the chunks, say exactly: "
    "\"I couldn't find this in the selected documents.\"\n"
    "- Never make up facts not present in the chunks\n"
    "- Use bullet points for lists, proper paragraphs for explanations\n"
    "- If part of the question is general knowledge (math, definitions, etc.), "
    "answer that part from your own knowledge and the document part from the chunks."
)


# ---------------------------------------------------------------------------
# General (no document)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM = (
    "You are DocIntel, a helpful AI assistant. "
    "The user is asking a general question not related to any specific document. "
    "Answer helpfully, concisely, and accurately from your general knowledge."
)

# user side: f"Conversation so far:\n{history_text}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Question classifier
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = (
    "Classify the following question. "
    "Reply with ONLY one word — \"document\" or \"general\".\n\n"
    "Rules:\n"
    "- If the question mentions anything about an uploaded file, document, letter, "
    "contract, invoice, resume, report — reply \"document\"\n"
    "- If the question references \"this\", \"the document\", \"it\", \"the file\", "
    "\"above\", \"here\" — reply \"document\"\n"
    "- If the question is PURELY general with zero document reference "
    "(e.g. \"what is the capital of France\", \"hello\", \"what is 2+2\") — reply \"general\"\n"
    "- When in doubt — reply \"document\""
)

# user side: question (raw string)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

QUERY_EXPANSION_SYSTEM = (
    "Rewrite the following question to be more specific and retrieval-friendly. "
    "Keep the same intent but make it clearer and more detailed. "
    "Return ONLY the rewritten question, nothing else."
)

# user side: question (raw string)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are a precise data extraction assistant. "
    "Extract the following fields from the document below.\n\n"
    "Rules:\n"
    "- Extract ONLY what is explicitly stated in the document\n"
    "- For each field, use the field name and its description as a guide "
    "for WHICH entity's info to extract\n"
    "- If a field is not found, use null for strings or [] for lists\n"
    "- Do NOT guess or infer — only extract what is clearly present\n"
    "- Do NOT mix up entities (e.g. don't return company email when asked for candidate email)\n"
    "- Return ONLY a valid JSON object, no explanation, no markdown fences"
)

# user side assembled in retrieval.py extract_fields():
# f"{correction_examples}Fields to extract:\n{fields_with_desc}\n\nDocument:\n{context}"


# ---------------------------------------------------------------------------
# Document type classifier (Stage 2 — LLM)
# ---------------------------------------------------------------------------
#
# Used by retrieval._classify_from_context() via classification.pipeline._call_stage2().
# Stage 1 (keyword + embedding) runs first; this prompt only fires when Stage 1
# confidence falls below CLASSIFIER_CONFIDENCE_THRESHOLD.
#
# The response is coerced into DocumentClassification (llm/structured.py) by
# Instructor — no JSON format instruction needed here.
#
# user side assembled inline in _classify_from_context():
# f"Document text (first {N} chars):\n\n{context}"

DOCUMENT_CLASSIFIER_SYSTEM = (
    "You are a document classification expert. "
    "Classify the document based on the text excerpt provided.\n\n"
    "Supported document types:\n"
    "invoice, receipt, purchase_order, bank_statement, cv_resume, contract, nda, "
    "loan_application, gst_return, offer_letter, medical, legal, report, general, id_document\n\n"
    "Rules:\n"
    "- Choose the single most specific type that fits the document\n"
    "- Use 'general' only if no other type fits clearly\n"
    "- Provide a confidence score from 0.0 (uncertain) to 1.0 (certain)\n"
    "- Give a one-sentence reasoning explaining your choice\n"
    "- List 2-4 short key signals (phrases or patterns) from the text that "
    "led to your classification"
)


# ---------------------------------------------------------------------------
# Stage 1 classifier exemplars (keyword + embedding)
# ---------------------------------------------------------------------------
#
# Moved here from the former classification/exemplars.py when the
# classification/ package was folded into retrieval.py (see
# retrieval._classify_stage1()). This is static reference data, not LLM
# instruction text, but it lives alongside DOCUMENT_CLASSIFIER_SYSTEM since
# both exist to ground document classification.
#
# KEYWORD_SIGNALS     — short phrases checked against the first 500 chars of
#                       document text (case-insensitive substring match).
# EMBEDDING_EXEMPLARS — short representative passages embedded once and
#                       cached for cosine similarity against the document's
#                       first ~300 chars.
# Both are static domain knowledge — no DB, no LLM, no network calls.

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
    "id_document": [
        "date of birth", "date of issue", "date of expiry", "nationality",
        "passport no", "driving licence", "aadhaar", "pan card",
        "voter id", "national id",
    ],
}

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
    "id_document": (
        "Name: John Doe\nDate of Birth: 01/01/1990\nNationality: Indian\n"
        "Passport No: A1234567\nDate of Issue: 01/01/2020\n"
        "Date of Expiry: 31/12/2029"
    ),
}

