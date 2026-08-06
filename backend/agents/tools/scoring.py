"""
backend/agents/tools/scoring.py
----------------------------------
Shared structured-scoring calls used by the per-criterion sub-agents
(score_education, score_skill_groups, score_experience, ...).

score_candidates(): one LLM call scores ALL candidates on ONE criterion at
once. Used for education and experience.

score_skill_groups(): scores ALL candidates against ALL skill groups,
chunked into batched calls per optimization_config.SCORE_SKILLS_PAIRS_PER_CALL
- each call covers a chunk of candidates x every skill group at once (a
grid), rather than one call per skill group covering every candidate. This
keeps prompt size bounded regardless of candidate count, at the cost of
potentially more total calls when skill groups don't merge well - see
optimization_config.py for the tunable cap.

Both return a uniform {document_id: {"score", "reasoning", "evidence_snippet"}}
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
from agents.cv_processor.prompts import _FAIRNESS_CLAUSE, _SCOPE_CLAUSE, _DEPTH_CLAUSE, _EXCLUSION_CLAUSE
from agents.cv_processor import optimization_config as OPT
from db import insert_agent_flag

logger = get_logger("agents.tools.scoring")


class CandidateScore(BaseModel):
    document_id: str
    score: float = Field(ge=0, le=10, description="0 = no evidence / does not meet the criterion, 10 = excellent match")
    reasoning: str = Field(description="Short, specific justification citing what was found in the candidate's data")
    evidence_snippet: str = Field(
        description="A short quote or close paraphrase (1-2 sentences max) from the candidate's own data that "
                    "most directly supports this score - the specific fact/phrase the reasoning is grounded in. "
                    "Must be traceable to something actually present in the candidate data given, never invented."
    )


class ScoreBatch(BaseModel):
    scores: list[CandidateScore]


def score_candidates(
    candidates: list[dict],
    criterion_name: str,
    criterion_instructions: str,
    run_id: str,
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
    this criterion and any stated preferences from plan_evaluation.

    Returns {document_id: {"score": float, "reasoning": str, "evidence_snippet": str}}.
    Any document_id from `candidates` NOT in the model's response is filled
    in with a 0-score "no data returned by scoring model" entry, so callers
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
        "(not a generic statement). For each candidate, also provide evidence_snippet: a short quote or "
        "close paraphrase (1-2 sentences max) taken from that candidate's own data, which is the specific "
        "fact your score and reasoning are grounded in. Never invent an evidence_snippet - it must be "
        "traceable to something actually present in the candidate data given. You MUST return exactly one "
        "score entry per candidate document_id given below, in the same order."
        f"{_FAIRNESS_CLAUSE}{_SCOPE_CLAUSE}{_DEPTH_CLAUSE}"
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
            c["document_id"]: {"score": 0.0, "reasoning": f"Scoring failed: {exc}", "evidence_snippet": ""}
            for c in candidates
        }

    scores = {
        s.document_id: {"score": s.score, "reasoning": s.reasoning, "evidence_snippet": s.evidence_snippet}
        for s in result.scores
    }
    insert_agent_flag(
        run_id, "bias_check",
        {"criterion": criterion_name, "n_candidates": len(candidates)},
    )
    # Backfill any candidate the model silently dropped, rather than letting
    # a downstream KeyError surface deep inside merge_scores().
    for c in candidates:
        doc_id = c["document_id"]
        if doc_id not in scores:
            logger.warning("Scoring model omitted a candidate - backfilling 0", criterion=criterion_name, document_id=doc_id)
            scores[doc_id] = {
                "score": 0.0,
                "reasoning": "No score returned by the scoring model for this candidate.",
                "evidence_snippet": "",
            }

    return scores


# ---------------------------------------------------------------------------
# Skill-group scoring (chunked grid: candidates x skill_groups per call)
# ---------------------------------------------------------------------------

class SkillGroupCandidateScore(BaseModel):
    document_id: str
    group_name: str = Field(description="Must exactly match one of the given skill group names")
    score: float = Field(ge=0, le=10)
    reasoning: str = Field(description="Short, specific justification citing what was found in the candidate's data")
    evidence_snippet: str = Field(
        description="A short quote or close paraphrase (1-2 sentences max) from the candidate's own data "
                    "supporting this score for this specific skill group. Never invented."
    )


class SkillGroupScoreBatch(BaseModel):
    scores: list[SkillGroupCandidateScore]


def _skill_groups_system_prompt(skill_groups: list[dict]) -> str:
    group_blocks = []
    for g in skill_groups:
        members = ", ".join(g["member_skills"])
        if g["requirement_level"] == "must":
            weight_note = (
                "MUST-HAVE - a candidate with no evidence of this group should score very low (0-2). "
                "This will be weighted heavily in the final ranking, though not as an automatic disqualifier."
            )
        else:
            weight_note = "NICE-TO-HAVE - lack of evidence should only mildly lower the score, not tank it."
        group_blocks.append(
            f"- {g['group_name']} (covers: {members}) - {weight_note}"
        )
    groups_text = "\n".join(group_blocks)

    return (
        "You are scoring a batch of candidates against MULTIPLE skill groups at once. Each skill group "
        "represents one or more related skills/technologies from the job description, merged together "
        "because they describe the same underlying capability at different levels of specificity.\n\n"
        f"Skill groups to evaluate:\n{groups_text}\n\n"
        "For EACH candidate given, score them against EACH skill group listed above - you must return a "
        "full grid: one score entry per (candidate, skill_group) combination, using the exact group_name "
        "values given above. Evidence for a candidate's evidence within a group should reflect their "
        "strongest/most relevant member-skill experience for that specific group."
        f"{_EXCLUSION_CLAUSE}{_FAIRNESS_CLAUSE}{_SCOPE_CLAUSE}{_DEPTH_CLAUSE}"
    )


def _score_skill_groups_one_call(
    candidates_chunk: list[dict],
    skill_groups: list[dict],
    run_id: str,
    call_type: str,
    user_id: str,
) -> dict[str, dict[str, dict]]:
    """One LLM call covering candidates_chunk x every skill_group (a grid). Returns {group_name: {document_id: {...}}}."""
    candidates_text = "\n\n".join(
        f"Candidate document_id={c['document_id']}:\n"
        + json.dumps({k: v for k, v in c.items() if k != "document_id"}, default=str)
        for c in candidates_chunk
    )
    system = _skill_groups_system_prompt(skill_groups)

    try:
        result: SkillGroupScoreBatch = call_llm(
            system=system,
            user=candidates_text,
            temperature=0.0,
            call_type=call_type,
            response_model=SkillGroupScoreBatch,
            structured_max_retries=1,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error("Batched skill-group scoring call failed", n_candidates=len(candidates_chunk), n_groups=len(skill_groups), error=str(exc))
        out: dict[str, dict[str, dict]] = {g["group_name"]: {} for g in skill_groups}
        for c in candidates_chunk:
            for g in skill_groups:
                out[g["group_name"]][c["document_id"]] = {
                    "score": 0.0, "reasoning": f"Scoring failed: {exc}", "evidence_snippet": "",
                }
        return out

    out: dict[str, dict[str, dict]] = {g["group_name"]: {} for g in skill_groups}
    for s in result.scores:
        if s.group_name not in out:
            logger.warning("Model returned an unknown skill group name - dropping", group_name=s.group_name)
            continue
        out[s.group_name][s.document_id] = {"score": s.score, "reasoning": s.reasoning, "evidence_snippet": s.evidence_snippet}

    insert_agent_flag(
        run_id, "bias_check",
        {"criterion": "skill_groups_batch", "groups": [g["group_name"] for g in skill_groups], "n_candidates": len(candidates_chunk)},
    )
    return out


def score_skill_groups(
    candidates: list[dict],
    skill_groups: list[dict],
    run_id: str,
    call_type: str = "agent_score_skill_group",
    user_id: str = "system",
) -> dict[str, dict[str, dict]]:
    """
    Scores all candidates against all skill groups, chunked into batched
    LLM calls. Each call covers a chunk of candidates x every skill group
    at once - see optimization_config.SCORE_SKILLS_PAIRS_PER_CALL for the
    chunk-size tuning knob.

    candidates: list of per-candidate dicts, each MUST include "document_id"
    plus whatever fields are relevant (typically just work_experience,
    since all skill groups are evaluated from the same underlying text).

    skill_groups: list of {"group_name": str, "member_skills": [str],
    "requirement_level": "must"|"nice"} - see agent.py's plan_evaluation
    for how these are built.

    Returns {f"skill:{group_name}": {document_id: {score, reasoning, evidence_snippet}}}
    - same key format ("skill:<name>") that callers already expect from the
    old one-call-per-skill approach, so merge_scores/finalize_response need
    no changes to consume this.
    """
    if not candidates or not skill_groups:
        return {}

    n = len(skill_groups)
    chunk_size = max(1, OPT.SCORE_SKILLS_PAIRS_PER_CALL // n)

    criterion_scores: dict[str, dict] = {f"skill:{g['group_name']}": {} for g in skill_groups}

    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i:i + chunk_size]
        batch_result = _score_skill_groups_one_call(chunk, skill_groups, run_id, call_type, user_id)
        for group_name, doc_scores in batch_result.items():
            criterion_scores.setdefault(f"skill:{group_name}", {}).update(doc_scores)

    # Backfill any (candidate, skill_group) pair the model silently dropped.
    for g in skill_groups:
        key = f"skill:{g['group_name']}"
        for c in candidates:
            doc_id = c["document_id"]
            if doc_id not in criterion_scores[key]:
                logger.warning("Skill-group scoring omitted a candidate - backfilling 0", group_name=g["group_name"], document_id=doc_id)
                criterion_scores[key][doc_id] = {
                    "score": 0.0,
                    "reasoning": "No score returned by the scoring model for this candidate/skill-group.",
                    "evidence_snippet": "",
                }

    return criterion_scores