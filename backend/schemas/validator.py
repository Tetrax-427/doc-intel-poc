"""
schemas/validator.py — Field-level confidence scoring, format validation, and
(NEW) deterministic verification for nested/dynamic extraction schemas.

CHANGED in this phase (dynamic/complex schema extraction):
  - score_confidence() now handles list-of-dict and single-dict values
    (nested schema fields) via score_nested_list_confidence() /
    score_nested_object_confidence(), instead of only flat scalars/lists
    of strings.
  - validate_extraction() only runs format validation (email/phone/date/
    amount/url regex checks) against flat string values — nested list/dict
    values skip straight to their shape-based confidence score.
  - NEW: date-derived verification for nested list fields (e.g. past_companies
    with start_date/end_date). This is VERIFICATION, not extraction — the LLM
    still only extracts what's stated in the document (see
    prompts.EXTRACTION_SYSTEM_NESTED, which explicitly forbids the LLM from
    computing durations itself). These functions run AFTER extraction and
    compute a duration deterministically from extracted start/end dates,
    then compare it against any stated duration the LLM did extract —
    surfacing agreement/disagreement rather than silently overwriting
    anything. See verify_dynamic_extraction() for the orchestrator, called
    from retrieval.extract_dynamic_fields() right after the LLM call.
"""

import re
from datetime import date, datetime

from core.logger import get_logger

logger = get_logger("schemas.validator")

# Date formats we can parse for verification purposes — broader than
# validate_date()'s formats below since extracted date strings vary a lot
# more than form-filled dates (e.g. "2019-06" year-month only).
_PARSE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%B %d, %Y", "%d %B %Y", "%Y-%m", "%m/%Y", "%B %Y", "%Y",
]

# Month name/abbreviation -> number, for the "Month'YY" pattern below
# (e.g. "Sept'25", "May'25") — common in resume date ranges but not
# parseable by any strptime format string since it's a 4-letter
# non-standard abbreviation ("Sept" isn't a valid %b token) glued to a
# 2-digit year with no day component.
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


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
    """
    Guess validation type from field name.

    CHANGED: matches whole snake_case tokens instead of raw substrings.
    Substring matching previously misclassified "candidate_name" as a date
    field (it contains "date" inside "candi-date") and would similarly
    misclassify any field containing "total" as an amount — both caught
    during testing of the dynamic/nested schema work. All existing template
    field names (schemas/templates.py) are snake_case with underscore
    separators, so this is a strict correctness fix with no behaviour change
    for any field already being classified correctly.
    """
    name = field_name.lower()
    tokens = set(name.split("_"))

    if tokens & {"email", "mail"}:
        return "email"
    if tokens & {"phone", "mobile", "contact", "tel"}:
        return "phone"
    if tokens & {"date", "dob", "birth", "expiry", "issued", "deadline"}:
        return "date"
    if tokens & {"amount", "salary", "price", "cost", "fee", "total", "balance", "ctc"}:
        return "amount"
    if tokens & {"url", "website", "linkedin", "link"}:
        return "url"
    if tokens & {"skills", "experience", "education", "languages", "items"}:
        return "list"
    return "text"


# --- Nested-value confidence helpers ---
# NEW — for schemas.dynamic.SchemaSpec fields whose values are lists of objects
# (e.g. past_companies) or single nested objects, rather than flat scalars/
# lists of strings. score_confidence() below dispatches into these by shape.

def score_nested_list_confidence(items: list) -> float:
    """
    Confidence for a list-of-objects field (e.g. past_companies: [{...}, {...}]):
    the average non-null-field ratio across all items. An item missing half its
    sub-fields drags the field's overall confidence down proportionally, rather
    than the list being scored as "non-empty = 0.9" regardless of content.
    Empty list -> 0.0 (same convention as an empty flat list).
    """
    if not items:
        return 0.0
    ratios = []
    for item in items:
        if not isinstance(item, dict) or not item:
            ratios.append(0.0)
            continue
        # Ignore the injected _verification block itself when scoring
        # completeness — it's derived metadata, not an extracted field.
        real_fields = {k: v for k, v in item.items() if k != "_verification"}
        if not real_fields:
            ratios.append(0.0)
            continue
        non_null = sum(1 for v in real_fields.values() if v not in (None, "", []))
        ratios.append(non_null / len(real_fields))
    return round(sum(ratios) / len(ratios), 2) if ratios else 0.0


def score_nested_object_confidence(value: dict) -> float:
    """Confidence for a single nested-object field: ratio of its non-null sub-fields."""
    if not value:
        return 0.0
    real_fields = {k: v for k, v in value.items() if k != "_verification"}
    if not real_fields:
        return 0.0
    non_null = sum(1 for v in real_fields.values() if v not in (None, "", []))
    return round(non_null / len(real_fields), 2) if real_fields else 0.0


# --- Confidence scorer ---

def score_confidence(value, field_name: str, field_type: str) -> float:
    """Return confidence score 0.0 - 1.0"""
    if value is None:
        return 0.0
    if isinstance(value, list):
        # CHANGED: list of dicts (nested schema list-of-objects) gets its own
        # scorer instead of the flat "non-empty = 0.9" heuristic below, since
        # a list of half-populated items should score lower than a list of
        # fully-populated ones.
        if value and isinstance(value[0], dict):
            return score_nested_list_confidence(value)
        return 0.9 if len(value) > 0 else 0.0
    if isinstance(value, dict):
        # NEW: single nested-object field (schemas.dynamic type="object")
        return score_nested_object_confidence(value)
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

    Works unchanged for nested/dynamic schema fields (schemas.dynamic.SchemaSpec) —
    fields_schema here only needs {name: description} at the TOP level, which is
    what retrieval.extract_dynamic_fields() passes. Per-field confidence for
    list/object values is handled inside score_confidence() above; format
    validation (email/phone/date/amount/url) only ever applies to flat string
    values now — nested values skip straight to their shape-based confidence.
    """
    results = {}
    overall_scores = []

    for field_name, schema_value in fields_schema.items():
        value = extracted.get(field_name)
        field_type = detect_field_type(field_name)
        confidence = score_confidence(value, field_name, field_type)
        status = get_status(confidence)
        overall_scores.append(confidence)

        # Run format validation — only meaningful for flat string values;
        # nested list/dict values are skipped (their confidence already
        # reflects sub-field completeness via score_confidence() above).
        validation_note = ""
        if value and isinstance(value, str) and field_type in ["email", "phone", "date", "amount", "url"]:
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


# ---------------------------------------------------------------------------
# NEW — date-derived verification for nested list fields
# ---------------------------------------------------------------------------
#
# Called by retrieval.extract_dynamic_fields() after the LLM extraction call
# returns. This is deterministic Python, not an LLM call — the model never
# computes durations itself (prompts.EXTRACTION_SYSTEM_NESTED forbids it);
# these functions compute them from the extracted start/end dates and use
# that to VERIFY any stated duration the LLM did pull from the text, rather
# than replacing it.

def _parse_date_loose(value: str | None) -> date | None:
    """
    Best-effort parse of an extracted date string.

    CHANGED: found during testing on real resume data — dates written as
    "Sept'25", "May'25", "Jul'24" (month name/abbreviation + apostrophe +
    2-digit year, day omitted) were silently unparseable, since "Sept" isn't
    a valid strptime %b token (4 letters, non-standard) and there's no day
    component. This is a common resume date-range format, so it's handled
    explicitly via _MONTH_MAP before falling back to _PARSE_FORMATS.

    Note: python's dateutil.parser with fuzzy=True was tried and rejected —
    it misparses "Sept'25" as day=25 of the *current* year (reads the
    2-digit token as a day, not a year), which is silently wrong rather
    than cleanly unparseable. Explicit pattern matching avoids that.
    """
    if not value or not str(value).strip():
        return None
    cleaned = str(value).strip()

    # "Month'YY" / "Month YY" / "Month, YY" — day omitted, defaults to the 1st
    m = re.match(r"^([A-Za-z]{3,9})[\s'\u2019]*,?\s*(\d{2}|\d{4})$", cleaned)
    if m:
        month_token, year_token = m.group(1).lower()[:3], m.group(2)
        if month_token in _MONTH_MAP:
            year = int(year_token)
            if len(year_token) == 2:
                year = 2000 + year  # resume dates are assumed post-2000
            try:
                return date(year, _MONTH_MAP[month_token], 1)
            except ValueError:
                pass

    # Explicit formats — tried against the raw string and an
    # apostrophe-stripped variant (covers "Sept 25" style inputs too)
    for candidate in (cleaned, cleaned.replace("'", " ").replace("\u2019", " ")):
        for fmt in _PARSE_FORMATS:
            try:
                return datetime.strptime(candidate.strip(), fmt).date()
            except ValueError:
                continue
    return None


def compute_duration(start_date: str | None, end_date: str | None) -> str | None:
    """
    Compute a human-readable duration ("1y 9m") from two extracted date
    strings. end_date=None (or unparseable, e.g. "Present"/"Current") is
    treated as "ongoing" and computed against today.

    Args:
        start_date: Raw extracted start date string, any _PARSE_FORMATS shape.
        end_date:   Raw extracted end date string, or None/unparseable for
                    an ongoing/current role.

    Returns:
        "Xy Ym" string, or None if start_date itself couldn't be parsed
        (nothing to compute from).
    """
    start = _parse_date_loose(start_date)
    if start is None:
        return None

    end = _parse_date_loose(end_date) if end_date else None
    if end is None:
        end = date.today()

    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    months = max(months, 0)

    years, remaining_months = divmod(months, 12)
    parts = []
    if years:
        parts.append(f"{years}y")
    if remaining_months or not years:
        parts.append(f"{remaining_months}m")
    return " ".join(parts)


def detect_verifiable_pairs(item_fields: list) -> dict:
    """
    Inspect a list-item's sub-field specs (schemas.dynamic.FieldSpec objects,
    or plain dicts with the same shape) to find a start/end date pair and
    (optionally) a stated-duration field, by naming convention — this is what
    makes verification schema-agnostic instead of hardcoded to "past_companies".

    Args:
        item_fields: The `properties` list of a type="list" FieldSpec — i.e.
                     the sub-field specs describing one item's shape.

    Returns:
        {"start_key": str, "end_key": str, "duration_key": str | None}
        or {} if no start/end date pair is found (verification is skipped
        for that field by the orchestrator below).
    """
    def _name(f):
        return (f.name if hasattr(f, "name") else f.get("name", "")).lower()

    def _type(f):
        return (f.type if hasattr(f, "type") else f.get("type", "")).lower()

    def _orig_name(f):
        return f.name if hasattr(f, "name") else f.get("name", "")

    def _tokens(name: str) -> set[str]:
        # CHANGED: token-based matching instead of substring search. Substring
        # matching on the raw name caused two false positives found during
        # testing: "candidate_name" contains "date" (candi-DATE), and
        # "total_time_spent" contains "to" (TO-tal), which previously made
        # duration fields get misdetected as the *end* date field. Splitting
        # on "_" and matching whole tokens avoids both.
        return set(name.split("_"))

    _DURATION_KEYWORDS = {"duration", "tenure"}
    _DURATION_PHRASES = ("total_time", "time_spent")  # matched as substrings deliberately — these are compound, checked before tokenising
    _START_KEYWORDS = {"start", "from", "begin", "joined", "joining"}
    _END_KEYWORDS = {"end", "till", "until", "left", "relieving"}
    # "to" deliberately excluded from end keywords — too short, collides with
    # "total", "tools", etc. as a substring/token in compound names.

    start_key, end_key, duration_key = None, None, None

    for f in item_fields:
        name, ftype = _name(f), _type(f)
        tokens = _tokens(name)

        # Duration/stated-tenure fields are checked FIRST and exclusively —
        # a field like "total_time_spent" must never fall through to the
        # start/end checks below.
        if any(p in name for p in _DURATION_PHRASES) or (tokens & _DURATION_KEYWORDS):
            duration_key = _orig_name(f)
            continue

        # CHANGED: no longer requires type=="date" or "date"/"dob" in the name —
        # schemas frequently use type="string" for "start_time"/"end_time", or
        # type="integer" for "start_year"/"end_year". Any field whose type is
        # date/string/integer/number AND whose name token matches a start/end
        # keyword now qualifies. Booleans and lists are excluded so a field
        # like "is_current" or a nested list can't be mistaken for a date.
        is_temporal_type = ftype in ("date", "string", "integer", "number", "")
        if not is_temporal_type:
            continue

        if tokens & _START_KEYWORDS:
            start_key = _orig_name(f)
        elif tokens & _END_KEYWORDS:
            end_key = _orig_name(f)

    if not (start_key and end_key):
        return {}
    return {"start_key": start_key, "end_key": end_key, "duration_key": duration_key}


def verify_item(item: dict, pair: dict) -> dict:
    """
    Build the _verification block for a single extracted list item.

    Args:
        item: One extracted item dict (e.g. one past_companies entry).
        pair: Output of detect_verifiable_pairs() for this list's item shape.

    Returns:
        {
          "start_date": <as extracted, for traceability>,
          "end_date":   <as extracted, or None>,
          "total_time_computed": "1y 9m" | None,
          "total_time_stated":   <LLM-extracted stated duration> | None,
          "match": true | false | None   # None when there's nothing stated to compare against
        }
    """
    start_val = item.get(pair["start_key"])
    end_val = item.get(pair["end_key"])
    computed = compute_duration(start_val, end_val)

    stated = item.get(pair["duration_key"]) if pair.get("duration_key") else None

    match = None
    if stated and computed:
        # Loose comparison — normalise whitespace/case, since stated formats
        # vary ("1 yr 9 months" vs "1y 9m"). Exact-string match is the only
        # thing we can safely assert without re-parsing free-text durations;
        # anything looser risks false "match" confirmations.
        match = stated.strip().lower().replace(" ", "") == computed.strip().lower().replace(" ", "")

    return {
        "start_date": start_val,
        "end_date": end_val,
        "total_time_computed": computed,
        "total_time_stated": stated,
        "match": match,
    }


def verify_dynamic_extraction(extracted: dict, spec) -> dict:
    """
    Orchestrator — walks a SchemaSpec's fields, finds every type="list" field
    with a verifiable start/end date pair, and attaches a `_verification`
    block to each item in that list within `extracted`. Called once, right
    after the LLM extraction call returns and before response wrapping.

    Args:
        extracted: The raw `.model_dump()` dict from the dynamic extraction
                   model (schemas.dynamic.spec_to_model result instance).
        spec:      The schemas.dynamic.SchemaSpec used for this extraction.

    Returns:
        The same `extracted` dict, mutated in place, with `_verification`
        added to each item of every verifiable list field. Fields with no
        detectable start/end date pair are left untouched — this is a
        best-effort enrichment, never a hard requirement.
    """
    for field in spec.fields:
        if field.type != "list" or not field.properties:
            continue

        pair = detect_verifiable_pairs(field.properties)
        if not pair:
            continue

        items = extracted.get(field.name)
        if not isinstance(items, list):
            continue

        for item in items:
            if isinstance(item, dict):
                item["_verification"] = verify_item(item, pair)

        logger.info(
            "Attached date verification",
            field=field.name,
            item_count=len(items),
            start_key=pair["start_key"],
            end_key=pair["end_key"],
        )

    return extracted