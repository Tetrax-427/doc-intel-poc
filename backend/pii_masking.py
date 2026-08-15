"""
backend/pii_masking.py

mask(text, mapping_rows) / unmask(text, mapping_rows) — the pair that
wraps every non-streaming call_llm() invocation (see llm/engine.py).
mapping_rows is the list built fresh by pii_mapping.build_mapping() for
that single call — never persisted, discarded right after unmask() runs.

mask(): replaces every real surface form with its placeholder.
unmask(): replaces every placeholder with its canonical (first) real value.

Both are pure string functions — no DB access, no side effects.
"""

import re


def mask(text: str, mapping_rows: list) -> str:
    """
    Replace every real_values surface form in `text` with its placeholder.

    Longer surface forms are replaced first (across the whole row set,
    not just within one row) so a shorter form that happens to be a
    substring of a longer one (e.g. "Verma" inside "Sukanya Verma") never
    gets partially replaced first and corrupts the longer match.
    """
    if not text or not mapping_rows:
        return text

    # flatten to (surface_form, placeholder) pairs, longest form first
    pairs = []
    for row in mapping_rows:
        for value in row["real_values"]:
            if value and value.strip():
                pairs.append((value, row["placeholder"]))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    masked = text
    for value, placeholder in pairs:
        masked = masked.replace(value, placeholder)

    return masked


def unmask(text: str, mapping_rows: list) -> str:
    """
    Replace every placeholder in `text` with its canonical real value —
    real_values[0], the first-detected surface form for that placeholder.

    Uses regex (not plain .replace) only to guard against placeholders
    that could theoretically appear as a prefix of one another — in
    practice the {n} suffix makes this a non-issue, but re.sub with
    word-boundary-safe literal matching is cheap insurance.
    """
    if not text or not mapping_rows:
        return text

    unmasked = text
    for row in mapping_rows:
        canonical_value = row["real_values"][0]
        unmasked = unmasked.replace(row["placeholder"], canonical_value)

    return unmasked