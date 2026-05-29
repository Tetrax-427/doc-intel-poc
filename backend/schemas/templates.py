TEMPLATES = {
    "cv_resume": {
        "label": "📄 CV / Resume",
        "description": "Extract candidate details from a resume or CV",
        "schema": {
            "candidate_name": "full name of the candidate",
            "email": "candidate's email address, not the company's",
            "phone": "candidate's phone or mobile number",
            "location": "candidate's city or location",
            "current_role": "candidate's current or most recent job title",
            "current_company": "candidate's current or most recent employer",
            "total_experience": "total years of professional experience",
            "education": "highest education qualification and institution",
            "skills": "list of technical and professional skills",
            "certifications": "list of certifications or courses completed",
            "linkedin": "LinkedIn profile URL if present",
            "summary": "professional summary or objective statement"
        }
    },

    "invoice": {
        "label": "🧾 Invoice",
        "description": "Extract billing details from an invoice",
        "schema": {
            "invoice_number": "unique invoice or bill number",
            "invoice_date": "date the invoice was issued",
            "due_date": "payment due date",
            "vendor_name": "name of the company or person issuing the invoice",
            "vendor_address": "address of the vendor",
            "vendor_gstin": "GST identification number of the vendor",
            "client_name": "name of the client or buyer",
            "client_address": "address of the client",
            "client_gstin": "GST identification number of the client",
            "line_items": "list of products or services with quantities and prices",
            "subtotal": "amount before tax",
            "tax_amount": "total tax or GST amount",
            "total_amount": "final total amount payable",
            "payment_terms": "payment terms or conditions"
        }
    },

    "loan_application": {
        "label": "🏦 Loan Application",
        "description": "Extract details from a loan application form",
        "schema": {
            "applicant_name": "full name of the loan applicant",
            "applicant_dob": "date of birth of the applicant",
            "applicant_pan": "PAN card number of the applicant",
            "applicant_aadhaar": "Aadhaar number if present",
            "applicant_address": "permanent address of the applicant",
            "applicant_phone": "contact number of the applicant",
            "applicant_email": "email address of the applicant",
            "employer_name": "name of the applicant's employer",
            "monthly_income": "monthly gross income of the applicant",
            "loan_amount": "requested loan amount",
            "loan_purpose": "purpose of the loan",
            "loan_tenure": "requested loan tenure in months or years",
            "existing_loans": "details of any existing loans or EMIs",
            "collateral": "collateral or security offered if any"
        }
    },

    "offer_letter": {
        "label": "📋 Offer Letter",
        "description": "Extract employment offer details",
        "schema": {
            "candidate_name": "name of the candidate receiving the offer",
            "position": "job title or designation offered",
            "department": "department the candidate will join",
            "joining_date": "proposed date of joining",
            "ctc": "total cost to company or annual salary",
            "basic_salary": "basic salary component",
            "company_name": "name of the company making the offer",
            "reporting_to": "name or title of reporting manager",
            "work_location": "office location or city",
            "offer_expiry": "date by which offer must be accepted",
            "probation_period": "probation period duration"
        }
    },

    "gst_return": {
        "label": "📊 GST Return",
        "description": "Extract GST filing details",
        "schema": {
            "gstin": "GST identification number of the taxpayer",
            "legal_name": "legal name of the registered business",
            "return_period": "tax period for which return is filed",
            "return_type": "type of GST return (GSTR-1, GSTR-3B etc.)",
            "filing_date": "date the return was filed",
            "total_taxable_value": "total taxable turnover for the period",
            "total_cgst": "total Central GST amount",
            "total_sgst": "total State GST amount",
            "total_igst": "total Integrated GST amount",
            "total_tax_liability": "total tax liability for the period",
            "itc_claimed": "total input tax credit claimed",
            "net_tax_payable": "net tax payable after ITC"
        }
    },

    "contract": {
        "label": "📝 Contract / Agreement",
        "description": "Extract key terms from a legal contract",
        "schema": {
            "contract_title": "title or name of the agreement",
            "party_1": "name of the first party",
            "party_2": "name of the second party",
            "effective_date": "date the contract becomes effective",
            "expiry_date": "date the contract expires",
            "contract_value": "total value or consideration of the contract",
            "payment_terms": "payment schedule or terms",
            "notice_period": "notice period required for termination",
            "governing_law": "jurisdiction or governing law",
            "key_obligations": "list of main obligations of each party",
            "termination_clause": "conditions under which contract can be terminated",
            "renewal_terms": "auto-renewal or extension conditions"
        }
    },

    "bank_statement": {
        "label": "🏧 Bank Statement",
        "description": "Extract summary from a bank statement",
        "schema": {
            "account_holder": "name of the account holder",
            "account_number": "bank account number",
            "bank_name": "name of the bank",
            "ifsc_code": "IFSC code of the branch",
            "statement_period": "period covered by the statement",
            "opening_balance": "balance at the start of the period",
            "closing_balance": "balance at the end of the period",
            "total_credits": "total amount credited during the period",
            "total_debits": "total amount debited during the period",
            "average_balance": "average monthly balance if mentioned"
        }
    }
}

VISION_PROMPTS = {
    "general": """Describe this image in detail. Include:
- What type of document or scene this is
- All visible text, numbers, and data
- Layout and structure
- Any important visual elements""",

    "cv_resume": """Analyze this CV or resume image. Describe:
- The candidate's name and contact details visible
- Professional experience and companies mentioned
- Education qualifications
- Skills and certifications listed
- Overall document structure and completeness""",

    "invoice": """Analyze this invoice or bill image. Extract and describe:
- Vendor and client names
- Invoice number and date
- All line items with quantities and amounts
- Tax breakdown (GST/VAT)
- Total amount payable
- Payment terms if visible""",

    "construction_loan": """Analyze this construction site or loan document image. Describe:
- Current construction stage and progress visible
- Structural elements present (foundation, columns, slabs, walls)
- Materials and equipment visible
- Estimated completion percentage
- Any visible defects or concerns
- Document details if this is a paper form""",

    "gst_return": """Analyze this GST document or return image. Extract and describe:
- GSTIN and business name
- Tax period and return type
- All financial figures visible
- Tax amounts (CGST, SGST, IGST)
- Filing status if shown""",

    "id_document": """Analyze this identity document image. Describe:
- Document type (Aadhaar, PAN, Passport, Driving License)
- Name and identifying details visible
- Issue and expiry dates if present
- Document number if visible
- Any other relevant fields""",

    "bank_statement": """Analyze this bank statement image. Describe:
- Account holder name and account number
- Bank name and branch
- Statement period
- Opening and closing balances
- Notable transactions if visible"""
}

def get_vision_prompt(template_id: str = "general") -> str:
    return VISION_PROMPTS.get(template_id, VISION_PROMPTS["general"])

def get_template(template_id: str) -> dict:
    return TEMPLATES.get(template_id, {})


def list_templates() -> list[dict]:
    return [
        {"id": k, "label": v["label"], "description": v["description"]}
        for k, v in TEMPLATES.items()
    ]