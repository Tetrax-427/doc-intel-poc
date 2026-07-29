"""
backend/agents/tools/scoring.py
----------------------------------
Shared structured-scoring call used by every per-criterion sub-agent
(score_education, score_skill, score_experience, ...). One LLM call scores
ALL candidates on ONE criterion at once (cheaper + more consistent than
one call per candidate), returning a uniform {score, reasoning, source}
shape per candidate so merge_scores() downstream has consistent input
regardless of which criterion produced it.

Calls llm.engine.call_llm() directly (agents import it, same as backend
code does) - no separate LLM client, no new dependency.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from core.logger import get_logger
from llm.engine import call_llm

logger = get_logger("agents.tools.scoring")


class CandidateScore(BaseModel):
    document_id: str
    score: float = Field(ge=0, le=10, description="0 = no evidence / does not meet the criterion, 10 = excellent match")
    reasoning: str = Field(description="Short, specific justification citing what was found in the candidate's data")


class ScoreBatch(BaseModel):
    scores: list[CandidateScore]


def score_candidates(
    candidates: list[dict],
    criterion_name: str,
    criterion_instructions: str,
    call_type: str = "agent_score",
    user_id: str = "system",
) -> dict[str, dict]:
    """
    candidates: list of per-candidate dicts. Each MUST include "document_id".
    Callers should pre-trim each candidate dict to just the fields relevant
    to this criterion before calling (e.g. only education fields for
    score_education) - keeps the prompt focused and avoids one criterion's
    scoring being swayed by irrelevant data.

    criterion_instructions: free text describing what "good" looks like for
    this criterion and any stated preferences from plan_evaluation (e.g.
    "IIT > NIT > other institutes; must-have skill - weight heavily if
    missing entirely").

    Returns {document_id: {"score": float, "reasoning": str}}. Any
    document_id from `candidates` NOT in the model's response is filled in
    with a 0-score "no data returned by scoring model" entry, so callers
    never have to null-check a missing key downstream.
    """
    if not candidates:
        return {}

    candidates_text = "\n\n".join(
        f"Candidate document_id={c['document_id']}:\n"
        + json.dumps({k: v for k, v in c.items() if k != "document_id"}, default=str)
        for c in candidates
    )

    system = (
        f"You are scoring a batch of candidates on ONE evaluation criterion: {criterion_name}.\n\n"
        f"{criterion_instructions}\n\n"
        "Score each candidate 0-10 on this criterion alone - do not consider anything outside it. "
        "Give a short, specific reasoning per candidate that cites what you actually saw in their data "
        "(not a generic statement). You MUST return exactly one score entry per candidate "
        "document_id given below, in the same order."
    )

    try:
        result: ScoreBatch = call_llm(
            system=system,
            user=candidates_text,
            temperature=0.0,
            call_type=call_type,
            response_model=ScoreBatch,
            structured_max_retries=1,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error("Batch scoring call failed", criterion=criterion_name, error=str(exc))
        return {
            c["document_id"]: {"score": 0.0, "reasoning": f"Scoring failed: {exc}"}
            for c in candidates
        }

    scores = {s.document_id: {"score": s.score, "reasoning": s.reasoning} for s in result.scores}

    # Backfill any candidate the model silently dropped, rather than letting
    # a downstream KeyError surface deep inside merge_scores().
    for c in candidates:
        doc_id = c["document_id"]
        if doc_id not in scores:
            logger.warning("Scoring model omitted a candidate - backfilling 0", criterion=criterion_name, document_id=doc_id)
            scores[doc_id] = {"score": 0.0, "reasoning": "No score returned by the scoring model for this candidate."}

    return scores