"""
backend/pii_mapping.py

build_pii_mapping(document_id, user_id, full_text) is the single entry
point for turning a document's raw text into a set of PII mapping rows,
called once at upload time (see ingestion.py) — never per-LLM-call.

Pipeline:
    1. Regex pass (PAN, AADHAAR, EMAIL, PHONE, DOB) — in REGEX_ORDER,
       each match's span is masked out of a working copy of the text so
       later regex/Presidio passes can't re-claim it.
    2. Presidio pass, restricted to PRESIDIO_ENTITIES, run on the
       regex-stripped text.
    3. PERSON spans from Presidio are clustered (see build_name_mapping)
       into canonical people before placeholders are assigned — every
       other entity type gets one placeholder per detected span.
    4. Returns a list of dicts shaped for direct insertion into
       pii_mappings (one row per placeholder).

Placeholder format: __PII_{user_id}_{TYPE}_{n}__
Deliberately unusual/unique so it's extremely unlikely to collide with
real document text (per the "ignore the collision edge case, but keep
placeholders specific" decision).
"""

import logging
import re

from pii_config import enabled_regex_entities, PII_ENTITIES
from presidio_setup import analyze_text

logger = logging.getLogger("pii.mapping")


def _placeholder(user_id: str, entity_type: str, n: int) -> str:
    return f"__PII_{user_id}_{entity_type}_{n}__"


def _run_regex_pass(text: str):
    """
    Run each enabled regex entity in order. Matched spans are blanked out
    (replaced with spaces, preserving text length/offsets) in a working
    copy so later regex entries and the Presidio pass never see them.

    Returns (stripped_text, matches) where matches is a list of dicts:
    {entity_type, value, source, start, end}
    """
    working = text
    matches = []

    for entity_type, cfg in enabled_regex_entities():
        pattern = re.compile(cfg["pattern"])
        for m in pattern.finditer(working):
            matches.append(
                {
                    "entity_type": cfg["placeholder"],  # e.g. "PAN"
                    "value": m.group(),
                    "source": "regex",
                    "start": m.start(),
                    "end": m.end(),
                }
            )
        # blank out this entity's matches before the next regex/Presidio pass
        working = pattern.sub(lambda m: " " * len(m.group()), working)

    return working, matches


def _run_presidio_pass(stripped_text: str):
    """
    Run Presidio on regex-stripped text. Returns matches in the same shape
    as _run_regex_pass, mapped through PII_ENTITIES to get each entity's
    configured placeholder type (e.g. DATE_TIME -> "DOB").
    """
    results = analyze_text(stripped_text)

    # map presidio_entity -> our placeholder type, e.g. "DATE_TIME" -> "DOB"
    presidio_to_placeholder = {
        cfg["presidio_entity"]: cfg["placeholder"]
        for cfg in PII_ENTITIES.values()
        if cfg["method"] == "presidio" and cfg["enabled"]
    }

    matches = []
    for r in results:
        placeholder_type = presidio_to_placeholder.get(r.entity_type)
        if placeholder_type is None:
            continue  # shouldn't happen since analyze_text is restricted, but be safe
        matches.append(
            {
                "entity_type": placeholder_type,
                "value": stripped_text[r.start:r.end],
                "source": "presidio",
                "start": r.start,
                "end": r.end,
                "_raw_presidio_entity": r.entity_type,  # kept for name clustering below
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

    Merge rules, in order of confidence:
      1. Exact full-name match (case-insensitive, whitespace-normalized)
         -> same cluster, always.
      2. Surname match, AND that surname is unique across all clusters
         so far (only one person with that surname in the doc) -> merge.
      3. Everything else -> new cluster. No fuzzy/distance-based matching —
         wrongly merging two different people is worse than leaving them
         separate.

    Returns a list of clusters, each a list of the original match dicts
    that were grouped together. Order of clusters follows first-appearance
    order in the document.
    """
    clusters = []  # list of {"exact": set(normalized full names), "surname": str, "matches": [...]}

    for match in person_matches:
        norm = _normalize_name(match["value"])
        surname = _surname(match["value"])

        # rule 1: exact match against any existing cluster
        target = next((c for c in clusters if norm in c["exact"]), None)

        # rule 2: unique surname match
        if target is None and surname:
            surname_clusters = [c for c in clusters if c["surname"] == surname]
            if len(surname_clusters) == 1:
                target = surname_clusters[0]

        if target is not None:
            target["exact"].add(norm)
            target["matches"].append(match)
        else:
            clusters.append(
                {"exact": {norm}, "surname": surname, "matches": [match]}
            )

    return [c["matches"] for c in clusters]


def build_pii_mapping(document_id: str, user_id: str, full_text: str) -> list:
    """
    Main entry point. Returns a list of dicts ready for insertion into
    pii_mappings — ONE ROW PER PLACEHOLDER (not per mention):
        {document_id, user_id, placeholder, real_values, entity_type, source}

    real_values is a list of surface forms sharing that placeholder — e.g.
    a clustered PERSON entry might have real_values = ["Sukanya Verma",
    "S. Verma"]. Every other entity type is a single-element list.

    source reflects the FIRST match's source for that placeholder (regex
    matches never get merged with anything, so this only matters for
    Presidio-only entities, which are all single-source anyway).

    Called once per document, at upload time (ingestion.py), before doc
    summary generation or any other LLM call touches this document.
    """
    stripped_text, regex_matches = _run_regex_pass(full_text)
    presidio_matches = _run_presidio_pass(stripped_text)

    person_matches = [m for m in presidio_matches if m["entity_type"] == "PERSON"]
    other_presidio_matches = [m for m in presidio_matches if m["entity_type"] != "PERSON"]

    rows = []

    # non-person regex + presidio matches: one row per match (each already
    # gets its own placeholder, so real_values is always a single-element list)
    counters = {}
    for match in regex_matches + other_presidio_matches:
        etype = match["entity_type"]
        counters[etype] = counters.get(etype, 0) + 1
        rows.append(
            {
                "document_id": document_id,
                "user_id": user_id,
                "placeholder": _placeholder(user_id, etype, counters[etype]),
                "real_values": [match["value"]],
                "entity_type": etype,
                "source": match["source"],
            }
        )

    # person matches: clustered, one row per canonical person, real_values
    # holds every surface form in that cluster (e.g. full name + initials
    # form) so mask()/unmask() can recognize and restore either one.
    clusters = build_name_mapping(person_matches)
    for i, cluster in enumerate(clusters, start=1):
        # de-dupe exact repeats while preserving first-seen order
        seen = []
        for match in cluster:
            if match["value"] not in seen:
                seen.append(match["value"])
        rows.append(
            {
                "document_id": document_id,
                "user_id": user_id,
                "placeholder": _placeholder(user_id, "PERSON", i),
                "real_values": seen,
                "entity_type": "PERSON",
                "source": cluster[0]["source"],
            }
        )

    if not rows:
        logger.info("No PII detected for document %s.", document_id)

    return rows