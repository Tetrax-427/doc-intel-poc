"""
backend/pii_mapping.py

build_mapping(text, user_id) is the single entry point for detecting PII
in a block of text and producing a placeholder mapping — built fresh for
every LLM call, used once, and discarded. No document_id, no DB storage,
no upload-time step. This intentionally trades some repeated detection
work (see pii_config.py's regex-first design, which keeps that work cheap)
for correctness across multi-document batch calls (CV Processor, ITR
Helper) where a single persisted per-document mapping can't work, and for
zero PII ever touching disk outside your own DB's normal document rows.

Pipeline (unchanged from the original design, just no longer DB-backed):
    1. Regex pass (PAN, AADHAAR, EMAIL, PHONE, DOB) — in REGEX_ORDER,
       each match's span is blanked out of a working copy so later passes
       can't re-claim it.
    2. Presidio pass, restricted to PRESIDIO_ENTITIES, run on the
       regex-stripped text.
    3. PERSON spans are clustered (build_name_mapping) into canonical
       people before placeholders are assigned — every other entity type
       gets one placeholder per detected span.
    4. Returns a list of dicts: {placeholder, real_values, entity_type, source}
       — ready for mask()/unmask() (pii_masking.py). Not persisted anywhere.

Placeholder format: __PII_{user_id}_{TYPE}_{n}__
Since mapping is now single-call-scoped, this no longer needs to be
globally unique across documents — only unique within one call's
system+user text — but the format is kept as-is since it's still cheap
insurance against collision with real document text.
"""

import logging
import re

from pii_config import enabled_regex_entities, PII_ENTITIES
from presidio_setup import analyze_text

logger = logging.getLogger("pii.mapping")


def _placeholder(user_id: str, entity_type: str, n: int) -> str:
    return f"__PII_{user_id}_{entity_type}_{n}__"


def _run_regex_pass(text: str):
    working = text
    matches = []

    for entity_type, cfg in enabled_regex_entities():
        pattern = re.compile(cfg["pattern"])
        for m in pattern.finditer(working):
            matches.append(
                {
                    "entity_type": cfg["placeholder"],
                    "value": m.group(),
                    "source": "regex",
                    "start": m.start(),
                    "end": m.end(),
                }
            )
        working = pattern.sub(lambda m: " " * len(m.group()), working)

    return working, matches


def _run_presidio_pass(stripped_text: str):
    results = analyze_text(stripped_text)

    presidio_to_placeholder = {
        cfg["presidio_entity"]: cfg["placeholder"]
        for cfg in PII_ENTITIES.values()
        if cfg["method"] == "presidio" and cfg["enabled"]
    }

    matches = []
    for r in results:
        placeholder_type = presidio_to_placeholder.get(r.entity_type)
        if placeholder_type is None:
            continue
        matches.append(
            {
                "entity_type": placeholder_type,
                "value": stripped_text[r.start:r.end],
                "source": "presidio",
                "start": r.start,
                "end": r.end,
            }
        )
    return matches


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _surname(value: str) -> str:
    parts = _normalize_name(value).split(" ")
    return parts[-1] if parts else ""


def build_name_mapping(person_matches: list):
    """
    Conservative clustering of PERSON matches into canonical people.
      1. Exact full-name match -> same cluster, always.
      2. Surname match, AND that surname is unique across clusters so far
         -> merge.
      3. Everything else -> new cluster. No fuzzy/distance matching —
         wrongly merging two different people is worse than leaving them
         separate.
    """
    clusters = []

    for match in person_matches:
        norm = _normalize_name(match["value"])
        surname = _surname(match["value"])

        target = next((c for c in clusters if norm in c["exact"]), None)

        if target is None and surname:
            surname_clusters = [c for c in clusters if c["surname"] == surname]
            if len(surname_clusters) == 1:
                target = surname_clusters[0]

        if target is not None:
            target["exact"].add(norm)
            target["matches"].append(match)
        else:
            clusters.append({"exact": {norm}, "surname": surname, "matches": [match]})

    return [c["matches"] for c in clusters]


def build_mapping(text: str, user_id: str) -> list:
    """
    Main entry point. Returns a list of dicts, one per placeholder:
        {placeholder, real_values, entity_type, source}

    Called fresh inside call_llm() for every non-streaming call — never
    persisted, never reused across calls. Caller is responsible for
    discarding this after mask()/unmask() for that one call.
    """
    if not text or not text.strip():
        return []

    stripped_text, regex_matches = _run_regex_pass(text)
    presidio_matches = _run_presidio_pass(stripped_text)

    person_matches = [m for m in presidio_matches if m["entity_type"] == "PERSON"]
    other_presidio_matches = [m for m in presidio_matches if m["entity_type"] != "PERSON"]

    rows = []

    counters = {}
    for match in regex_matches + other_presidio_matches:
        etype = match["entity_type"]
        counters[etype] = counters.get(etype, 0) + 1
        rows.append(
            {
                "placeholder": _placeholder(user_id, etype, counters[etype]),
                "real_values": [match["value"]],
                "entity_type": etype,
                "source": match["source"],
            }
        )

    clusters = build_name_mapping(person_matches)
    for i, cluster in enumerate(clusters, start=1):
        seen = []
        for match in cluster:
            if match["value"] not in seen:
                seen.append(match["value"])
        rows.append(
            {
                "placeholder": _placeholder(user_id, "PERSON", i),
                "real_values": seen,
                "entity_type": "PERSON",
                "source": cluster[0]["source"],
            }
        )

    return rows