"""
backend/pii_config.py

Config-based registry of PII entity types to detect and mask before any
text reaches an external LLM. Add/remove/disable an entity type here —
no changes needed anywhere else in the masking pipeline.

Two detection methods:
  - "regex"    -> matched directly against raw text, in this file.
  - "presidio" -> delegated to Presidio's NER analyzer (see presidio_setup.py).
                  `pattern` is unused for these; `presidio_entity` names the
                  Presidio entity type to request.

Regex entries are applied first, in the order below, before Presidio runs —
so PAN/Aadhaar/phone/email/DOB spans are claimed and removed from
consideration before NER gets a chance to mis-tag them as something else.

Placeholder format: __PII_{user_id}_{TYPE}_{n}__
Built at mask-time (see masking.py), using `placeholder` below as the {TYPE}
token. The user_id + underscore-wrapped, double-underscore-delimited format
is intentionally unusual so it's extremely unlikely to collide with real
document text.
"""

PII_ENTITIES = {
    "PAN": {
        "enabled": True,
        "method": "regex",
        # Indian PAN: 5 letters, 4 digits, 1 letter — e.g. ABCDE1234F
        "pattern": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "placeholder": "PAN",
    },
    "AADHAAR": {
        "enabled": True,
        "method": "regex",
        # 12 digits, optionally space/hyphen separated in groups of 4
        "pattern": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "placeholder": "AADHAAR",
    },
    "EMAIL": {
        "enabled": True,
        "method": "regex",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "placeholder": "EMAIL",
    },
    "PHONE": {
        "enabled": True,
        "method": "regex",
        # Indian mobile: optional +91/91/0 prefix, then 10 digits starting 6-9.
        # Also catches common formatted variants: spaces, hyphens.
        "pattern": r"(?<!\d)(?:\+91[\-\s]?|91[\-\s]?|0)?[6-9]\d{9}(?!\d)",
        "placeholder": "PHONE",
    },
    "DOB": {
        "enabled": True,
        "method": "regex",
        # Fast-path only — catches DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY cheaply
        # and deterministically (also matches other numeric dd/mm/yyyy dates
        # in the doc — acceptable over-masking). Does NOT catch natural-
        # language dates like "Oct 2, 1970" — that's DATE_TIME below, via
        # Presidio, which is the actual safety net for DOB as a whole.
        "pattern": r"\b(0?[1-9]|[12][0-9]|3[01])[\/\-\.](0?[1-9]|1[0-2])[\/\-\.](19|20)\d{2}\b",
        "placeholder": "DOB",
    },
    "PERSON": {
        "enabled": True,
        "method": "presidio",
        "presidio_entity": "PERSON",
        "placeholder": "PERSON",
    },
    "ADDRESS": {
        "enabled": True,
        "method": "presidio",
        "presidio_entity": "LOCATION",
        "placeholder": "ADDRESS",
    },
    "DATE_TIME": {
        "enabled": True,
        "method": "presidio",
        "presidio_entity": "DATE_TIME",
        # Reuses DOB's placeholder — regex and Presidio are two detection
        # paths for the same logical entity type, not two separate types.
        # Regex claims numeric dates first; Presidio only sees what's left
        # (natural-language dates like "Oct 2, 1970").
        "placeholder": "DOB",
    },
}

# Order regex entities run in — matters because DOB's date pattern could
# theoretically overlap with stray digit sequences; keeping PAN/AADHAAR/EMAIL/
# PHONE first means the more specific, less ambiguous patterns claim their
# spans before DOB's broader digit pattern runs.
REGEX_ORDER = ["PAN", "AADHAAR", "EMAIL", "PHONE", "DOB"]

# Entity types Presidio should be asked to detect, restricted to just these
# (never run Presidio open-ended) — pulled from config so disabling PERSON
# or ADDRESS above automatically removes it from the Presidio call.
PRESIDIO_ENTITIES = [
    v["presidio_entity"]
    for v in PII_ENTITIES.values()
    if v["method"] == "presidio" and v["enabled"]
]


def enabled_regex_entities():
    """Return regex-based entity configs, in REGEX_ORDER, skipping disabled ones."""
    return [
        (name, PII_ENTITIES[name])
        for name in REGEX_ORDER
        if PII_ENTITIES[name]["enabled"]
    ]