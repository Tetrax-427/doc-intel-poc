"""
backend/comparison.py

Document comparison engine — deterministic, word-level diff between two
parsed documents (PDF or DOCX for v1).

ASSUMPTIONS — adjust these imports/field names to match your actual code,
since this was written from the codebase guide, not the live source:
  - core.document.Document has a `.pages` attribute: List[DocumentPage]
  - DocumentPage has `.text: str`, `.page_number: int`, and a NEW field
    `.position_type: Literal["page", "paragraph"]` (see note in
    core/document.py — needs adding for DOCX, which has no real pages)
  - llm.engine.call_llm(prompt: str, **kwargs) -> str exists and matches
    your existing signature

No LLM is used to find the diff — only (optionally) to narrate it after
the fact. The diff itself is fully deterministic via difflib.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from core.document import Document
from core.logger import get_logger
from llm.engine import call_llm

logger = get_logger("comparison")

WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\S+|\s+")


@dataclass
class DiffSegment:
    type: str  # "unchanged" | "added" | "removed"
    text: str
    page_a: Optional[int] = None
    page_b: Optional[int] = None


@dataclass
class ComparisonResult:
    segments: List[DiffSegment]
    stats: dict
    position_type: str  # "page", "paragraph", or "mixed" if doc types differ
    summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Step 1-2: flatten each document to one string + normalize whitespace
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Collapse whitespace so PDF/DOCX extraction noise (double spaces,
    inconsistent line breaks) never registers as a false diff."""
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Split into word + whitespace tokens so the diff operates at word
    granularity and the original text can be reconstructed exactly by
    joining tokens back together."""
    return TOKEN_RE.findall(text)


def flatten_document(document: Document) -> Tuple[str, List[Tuple[int, int]], str]:
    """
    Concatenate all pages/paragraphs into one string. Returns:
      - full_text: the concatenated text
      - offsets: list of (char_offset_start, position_number), ascending,
        used to map a diff segment back to a page (PDF) or paragraph (DOCX)
      - position_type: "page" or "paragraph"
    """
    full_text = ""
    offsets: List[Tuple[int, int]] = []
    position_type = "page"

    for p in document.pages:
        position_type = getattr(p, "position_type", "page")
        offsets.append((len(full_text), p.page_number))
        full_text += p.text + "\n"

    return full_text, offsets, position_type


def position_for_offset(char_offset: int, offsets: List[Tuple[int, int]]) -> Optional[int]:
    """Binary search: which page/paragraph does this character offset fall on."""
    if not offsets:
        return None
    starts = [o[0] for o in offsets]
    idx = max(0, bisect_right(starts, char_offset) - 1)
    return offsets[idx][1]


def _char_offset_of_token(tokens: List[str], token_index: int) -> int:
    return len("".join(tokens[:token_index]))


# ---------------------------------------------------------------------------
# Step 3-4: the actual diff — SequenceMatcher opcodes -> segments
# ---------------------------------------------------------------------------

def diff_documents(document_a: Document, document_b: Document) -> ComparisonResult:
    """
    Core, deterministic diff. No LLM involved in finding what changed —
    only difflib's LCS-based matching (same family of algorithm git uses).
    """
    text_a, offsets_a, pos_type_a = flatten_document(document_a)
    text_b, offsets_b, pos_type_b = flatten_document(document_b)

    position_type = pos_type_a if pos_type_a == pos_type_b else "mixed"

    tokens_a = tokenize(normalize(text_a))
    tokens_b = tokenize(normalize(text_b))

    matcher = SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    opcodes = matcher.get_opcodes()

    segments: List[DiffSegment] = []
    additions = 0
    removals = 0
    pages_touched = set()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            page_a = position_for_offset(_char_offset_of_token(tokens_a, i1), offsets_a)
            page_b = position_for_offset(_char_offset_of_token(tokens_b, j1), offsets_b)
            segments.append(DiffSegment(
                type="unchanged",
                text="".join(tokens_a[i1:i2]),
                page_a=page_a,
                page_b=page_b,
            ))
            continue

        # tag is delete, insert, or replace
        if tag in ("delete", "replace") and i2 > i1:
            page_a = position_for_offset(_char_offset_of_token(tokens_a, i1), offsets_a)
            text = "".join(tokens_a[i1:i2])
            segments.append(DiffSegment(type="removed", text=text, page_a=page_a, page_b=None))
            removals += len([t for t in tokens_a[i1:i2] if t.strip()])
            if page_a is not None:
                pages_touched.add(page_a)

        if tag in ("insert", "replace") and j2 > j1:
            page_b = position_for_offset(_char_offset_of_token(tokens_b, j1), offsets_b)
            text = "".join(tokens_b[j1:j2])
            segments.append(DiffSegment(type="added", text=text, page_a=None, page_b=page_b))
            additions += len([t for t in tokens_b[j1:j2] if t.strip()])
            if page_b is not None:
                pages_touched.add(page_b)

    stats = {
        "pages_a": len(document_a.pages),
        "pages_b": len(document_b.pages),
        "additions": additions,
        "removals": removals,
        "pages_touched": sorted(pages_touched),
    }

    logger.info(
        "Comparison complete",
        additions=additions,
        removals=removals,
        pages_touched=len(pages_touched),
    )

    return ComparisonResult(segments=segments, stats=stats, position_type=position_type)


def result_to_dict(result: ComparisonResult) -> dict:
    """Serialize a ComparisonResult for the API response / CONTRACTS.md shape."""
    return {
        "segments": [asdict(s) for s in result.segments],
        "stats": result.stats,
        "position_type": result.position_type,
        "summary": result.summary,
    }


# ---------------------------------------------------------------------------
# Optional narrative layer — separate from, and never a source of, the diff
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """You are summarizing the differences between two versions \
of a document for a busy reader. Below is a list of changes, each marked as \
REMOVED or ADDED, in reading order.

{diff_text}

Write a short plain-English summary (3-6 bullet points) of the material \
changes. Do not invent changes that are not listed above. Focus on meaning, \
not a word-for-word restatement."""


def summarize_changes(result: ComparisonResult) -> str:
    """
    Optional, separate LLM pass purely for narration. Never used to
    determine the diff itself — the segments above are already final and
    deterministic by the time this runs.
    """
    changed = [s for s in result.segments if s.type != "unchanged"]
    if not changed:
        return "No differences found between the two documents."

    diff_lines = []
    for s in changed:
        label = "REMOVED" if s.type == "removed" else "ADDED"
        diff_lines.append(f"[{label}] {s.text.strip()}")

    # Cap input size for very large diffs — keeps this a cheap summary call,
    # not a second full-document pass.
    prompt = SUMMARY_PROMPT.format(diff_text="\n".join(diff_lines[:200]))
    return call_llm(prompt)