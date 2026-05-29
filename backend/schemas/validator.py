import re
from datetime import datetime


# --- Field type validators ---

def validate_email(value: str) -> tuple[bool, str]:
    if not value:
        return False, "empty"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return (True, "valid") if re.match(pattern, str(value)) else (False, "invalid email format")


def validate_phone(value: str) -> tuple[bool, str]:
    if not value:
        return False, "empty"
    digits = re.sub(r'[\s\-\+\(\)]', '', str(value))
    return (True, "valid") if 7 <= len(digits) <= 15 and digits.lstrip('+').isdigit() else (False, "invalid phone format")


def validate_date(value: str) -> tuple[bool, str]:
    if not value:
        return False, "empty"
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y", "%Y"]
    for fmt in formats:
        try:
            datetime.strptime(str(value).strip(), fmt)
            return True, "valid"
        except ValueError:
            continue
    return False, "invalid date format"


def validate_amount(value) -> tuple[bool, str]:
    if value is None:
        return False, "empty"
    cleaned = re.sub(r'[₹$€£,\s]', '', str(value))
    try:
        float(cleaned)
        return True, "valid"
    except ValueError:
        return False, "not a number"


def validate_url(value: str) -> tuple[bool, str]:
    if not value:
        return False, "empty"
    return (True, "valid") if str(value).startswith(("http://", "https://")) else (False, "invalid URL")


# --- Field type detector ---

def detect_field_type(field_name: str) -> str:
    """Guess validation type from field name"""
    name = field_name.lower()
    if any(k in name for k in ["email", "mail"]):
        return "email"
    if any(k in name for k in ["phone", "mobile", "contact", "tel"]):
        return "phone"
    if any(k in name for k in ["date", "dob", "birth", "expiry", "issued", "deadline"]):
        return "date"
    if any(k in name for k in ["amount", "salary", "price", "cost", "fee", "total", "balance", "ctc"]):
        return "amount"
    if any(k in name for k in ["url", "website", "linkedin", "link"]):
        return "url"
    if any(k in name for k in ["skills", "experience", "education", "languages", "items"]):
        return "list"
    return "text"


# --- Confidence scorer ---

def score_confidence(value, field_name: str, field_type: str) -> float:
    """Return confidence score 0.0 - 1.0"""
    if value is None:
        return 0.0
    if isinstance(value, list):
        return 0.9 if len(value) > 0 else 0.0
    if isinstance(value, (int, float)):
        return 0.95

    str_val = str(value).strip()
    if not str_val or str_val.lower() in ["null", "none", "n/a", "not found", "unknown", ""]:
        return 0.0

    # Type-specific scoring
    validators = {
        "email": validate_email,
        "phone": validate_phone,
        "date": validate_date,
        "amount": validate_amount,
        "url": validate_url,
    }

    if field_type in validators:
        is_valid, _ = validators[field_type](str_val)
        return 0.95 if is_valid else 0.4

    # Text scoring — longer and more specific = higher confidence
    words = str_val.split()
    if len(words) >= 2:
        return 0.85
    if len(words) == 1 and len(str_val) > 2:
        return 0.7
    return 0.5


# --- Status labeler ---

def get_status(confidence: float) -> str:
    if confidence >= 0.85:
        return "FOUND"
    if confidence >= 0.4:
        return "LOW_CONFIDENCE"
    return "NOT_FOUND"


def get_status_color(status: str) -> str:
    return {"FOUND": "green", "LOW_CONFIDENCE": "orange", "NOT_FOUND": "red"}.get(status, "gray")


# --- Main validator ---

def validate_extraction(extracted: dict, fields_schema: dict) -> dict:
    """
    Takes extracted fields and original schema.
    Returns enriched result with confidence scores and validation status per field.
    """
    results = {}
    overall_scores = []

    for field_name, schema_value in fields_schema.items():
        value = extracted.get(field_name)
        field_type = detect_field_type(field_name)
        confidence = score_confidence(value, field_name, field_type)
        status = get_status(confidence)
        overall_scores.append(confidence)

        # Run format validation
        validation_note = ""
        if value and field_type in ["email", "phone", "date", "amount", "url"]:
            validators = {
                "email": validate_email,
                "phone": validate_phone,
                "date": validate_date,
                "amount": validate_amount,
                "url": validate_url,
            }
            is_valid, note = validators[field_type](str(value))
            if not is_valid:
                validation_note = note
                confidence = min(confidence, 0.4)
                status = "LOW_CONFIDENCE"

        results[field_name] = {
            "value": value,
            "confidence": round(confidence, 2),
            "status": status,
            "field_type": field_type,
            "validation_note": validation_note,
            "color": get_status_color(status)
        }

    overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0.0

    return {
        "fields": results,
        "overall_confidence": overall,
        "overall_status": get_status(overall),
        "found_count": sum(1 for r in results.values() if r["status"] == "FOUND"),
        "total_count": len(results)
    }