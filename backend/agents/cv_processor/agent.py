"""
backend/agents/cv_processor/agent.py
----------------------------------------
CV Processor: ranks a batch of candidates (given as document_ids + optional
Extraction Helper CSV rows) against a free-text task/JD.

Stage sequence:
    plan_evaluation -> detect_duplicates_and_issues -> score_education
    -> score_skills -> score_experience -> merge_scores (initial ranking)
    -> verify_claims (subset, post-ranking) -> compute_adjusted_scores
    -> review_ranking -> finalize_response

plan_evaluation is the only stage that can pause the run (status
"needs_input") - see its docstring for how resume re-enters it.

CHANGED (performance optimization pass): verify_claims moved from before
scoring to after merge_scores, and now only runs on a subset of candidates
(chosen from the initial ranking) rather than everyone - see
optimization_config.py for the tunable knobs, and _select_verification_subset
below. score_skills now scores merged skill GROUPS (built once in
plan_evaluation) in chunked batched calls instead of one call per raw skill
covering every candidate. compute_adjusted_scores now batches multiple
candidates and multiple flagged criteria into single calls instead of one
call per flagged claim. All tunable batch sizes/caps live in
optimization_config.py, not hardcoded here.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from core.logger import get_logger
from llm.engine import call_llm
from db import insert_agent_flag
from agents.tools.candidate_data import build_candidate_dataset, get_field
from agents.tools.scoring import score_candidates, score_skill_groups
from agents.cv_processor import optimization_config as OPT

from agents.cv_processor.prompts import (
    PLAN_EVALUATION_SYSTEM,
    DETECT_DUPLICATES_SYSTEM,
    VERIFY_CLAIMS_SYSTEM,
    JUDGE_MERGE_SYSTEM,
    REVIEW_RANKING_SYSTEM,
    ADJUSTED_SCORES_SYSTEM,
    FINALIZE_SUMMARY_SYSTEM,
    PROJECT_HIGHLIGHT_SYSTEM,
)

logger = get_logger("agents.cv_processor")


# ---------------------------------------------------------------------------
# Structured models
# ---------------------------------------------------------------------------

class SkillRequirement(BaseModel):
    name: str
    requirement_level: str = Field(description="'must' or 'nice'")


class SkillGroupHint(BaseModel):
    group_name: str = Field(description="Short label for this merged skill group, e.g. 'Agentic AI'")
    member_skills: list[str] = Field(description="Exact skill names from the extracted skills list that belong to this group")


class Ambiguity(BaseModel):
    key: str
    question: str
    options: list[str] = []


class PlanDraft(BaseModel):
    education_preference: str
    skills: list[SkillRequirement]
    skill_group_hints: list[SkillGroupHint] = Field(default_factory=list)
    experience_notes: str
    other_preferences: str
    requested_count: int | None = None
    ambiguities: list[Ambiguity] = []


class RankedCandidate(BaseModel):
    document_id: str
    candidate_name: str = Field(description="Copy the candidate's name through exactly as given")
    rank: int = Field(description="1 = best")
    overall_score: float = Field(ge=0, le=10)
    summary_reasoning: str


class RankingResult(BaseModel):
    ranked: list[RankedCandidate]


class ProjectHighlight(BaseModel):
    document_id: str
    best_project: str = ""
    technologies: list[str] = []
    skill_context: str = ""


class ProjectHighlightBatch(BaseModel):
    highlights: list[ProjectHighlight]


class SuspiciousClaim(BaseModel):
    field: str = Field(description="Which field/claim this concerns, e.g. 'work_experience' or a specific skill mention")
    claimed_text: str = Field(description="The specific text/claim that looks suspicious")
    reasoning: str = Field(description="Why this looks inflated, inconsistent, or implausible")
    confidence: str = Field(description="'low', 'medium', or 'high' confidence that this is inflated/false")
    adjusted_text: str = Field(description="The same field's content with the suspicious claim removed/toned down, for a 'without this claim' comparison score")


class ClaimVerification(BaseModel):
    document_id: str
    suspicious_claims: list[SuspiciousClaim] = Field(default_factory=list)


class ClaimVerificationBatch(BaseModel):
    verifications: list[ClaimVerification]


class DuplicateGroup(BaseModel):
    document_ids: list[str] = Field(description="Every document_id believed to be the same person")
    reasoning: str = Field(description="Which specific signals (name, email, phone, education, work history) led to this conclusion")
    contradictions: list[str] = Field(default_factory=list, description="Fields where the group's resumes actually disagree with each other, if any")


class DuplicateDetectionResult(BaseModel):
    groups: list[DuplicateGroup] = Field(default_factory=list)


class RankSwap(BaseModel):
    document_id_higher: str = Field(description="document_id currently at the higher (better) of the two adjacent ranks being swapped")
    document_id_lower: str = Field(description="document_id currently at the lower (worse) of the two adjacent ranks being swapped")
    reasoning: str = Field(description="Specific per-criterion scores justifying why these two should swap")


class RankingReview(BaseModel):
    swaps: list[RankSwap] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="Non-adjacent inconsistencies noticed but not fixed - out of scope for this pass")


class AdjustedCriterionScore(BaseModel):
    document_id: str
    criterion: str = Field(description="Must exactly match one of the criterion keys given for this candidate")
    adjusted_score: float = Field(ge=0, le=10)


class AdjustedScoreBatch(BaseModel):
    adjustments: list[AdjustedCriterionScore]


# ---------------------------------------------------------------------------
# Stage: plan_evaluation
# ---------------------------------------------------------------------------

def _apply_answers_to_plan(plan_draft: dict, user_answers: dict) -> dict:
    """
    Folds resumed clarification answers into the plan as plain clarifying
    text on education_preference/other_preferences, rather than trying to
    re-parse each ambiguity type structurally - keeps this robust to any
    ambiguity shape the LLM produced.
    """
    plan = dict(plan_draft)
    clarifications = []
    for amb in plan_draft.get("ambiguities", []):
        key = amb["key"]
        if key in user_answers:
            clarifications.append(f"(User clarified - {amb['question']}): {user_answers[key]}")
    if clarifications:
        plan["other_preferences"] = (plan.get("other_preferences", "") + " " + " ".join(clarifications)).strip()
    plan.pop("ambiguities", None)
    return plan


def _build_skill_groups(skills: list[dict], hints: list[dict]) -> list[dict]:
    """
    Cross-references the LLM's skill_group_hints against the actual
    extracted skills list, and computes each group's requirement_level in
    CODE (not from the LLM) - "must" if any member skill is a must-have,
    else "nice" - so this is deterministic and can't drift from the raw
    per-skill tags. Also guarantees every skill ends up in exactly one
    group: skills the LLM didn't assign to any hint become singleton
    groups; skills a hint assigned that don't actually exist, or that got
    assigned twice, are dropped/de-duplicated defensively.
    """
    skill_by_name = {s["name"]: s for s in skills}
    assigned: set[str] = set()
    groups: list[dict] = []

    for hint in hints:
        members = [m for m in hint.get("member_skills", []) if m in skill_by_name and m not in assigned]
        if not members:
            continue
        assigned.update(members)
        requirement_level = "must" if any(skill_by_name[m]["requirement_level"] == "must" for m in members) else "nice"
        groups.append({
            "group_name": hint.get("group_name") or members[0],
            "member_skills": members,
            "requirement_level": requirement_level,
        })

    for name, skill in skill_by_name.items():
        if name not in assigned:
            groups.append({
                "group_name": name,
                "member_skills": [name],
                "requirement_level": skill["requirement_level"],
            })

    return groups


def plan_evaluation(run_id: str, state: dict) -> dict:
    if "plan" in state:
        # Shouldn't normally happen (next_stage moves past this once plan is
        # set) but guards against any accidental re-entry.
        return {"next_stage": "detect_duplicates_and_issues"}

    task = state["task"]
    user_id = state.get("_user_id", "system")
    n_candidates = len(state["input_data"].get("document_ids", []))

    plan_draft_raw = state.get("plan_draft_raw")

    if plan_draft_raw is None:
        # First entry for this run.
        try:
            draft: PlanDraft = call_llm(
                system=PLAN_EVALUATION_SYSTEM,
                user=(
                    f"{n_candidates} candidate(s) have already been selected for evaluation "
                    f"(candidate selection is not part of your task).\n\nTask:\n{task}"
                ),
                temperature=0.0,
                call_type="agent_plan",
                response_model=PlanDraft,
                structured_max_retries=1,
                user_id=user_id,
            )
        except Exception as exc:
            logger.error("Plan generation failed", run_id=run_id, error=str(exc))
            raise

        plan_draft_raw = draft.model_dump()

        if plan_draft_raw["ambiguities"]:
            questions = [
                {"key": a["key"], "question": a["question"], "type": "select" if a["options"] else "text", "options": a["options"]}
                for a in plan_draft_raw["ambiguities"]
            ]
            return {"state_update": {"plan_draft_raw": plan_draft_raw}, "questions": questions}

        plan = _apply_answers_to_plan(plan_draft_raw, {})
    else:
        # Resumed - state["user_answers"] now has the caller's answers.
        plan = _apply_answers_to_plan(plan_draft_raw, state.get("user_answers", {}))

    plan["skill_groups"] = _build_skill_groups(plan["skills"], plan_draft_raw.get("skill_group_hints", []))
    plan.pop("skill_group_hints", None)

    input_data = state["input_data"]
    candidates = build_candidate_dataset(input_data["document_ids"], input_data.get("csv_data", []))

    # Guardrail: flag any candidate whose resume text had hidden/invisible
    # characters stripped during sanitization (build_candidate_dataset()
    # marks these under "_sanitization_flag"). Popped off here so the raw
    # list of field names never rides along into a later LLM prompt.
    for c in candidates:
        modified_fields = c.pop("_sanitization_flag", None)
        if modified_fields:
            insert_agent_flag(
                run_id, "sanitization",
                {"fields": modified_fields, "note": "Hidden/invisible characters stripped from resume text before evaluation."},
                document_id=c["document_id"], severity="medium",
            )

    return {"state_update": {"plan": plan, "candidates": candidates}, "next_stage": "detect_duplicates_and_issues"}


# ---------------------------------------------------------------------------
# Stage: detect_duplicates_and_issues (runs on ALL candidates, cheap check)
# ---------------------------------------------------------------------------

def detect_duplicates_and_issues(run_id: str, state: dict) -> dict:
    candidates, user_id = state["candidates"], state.get("_user_id", "system")

    identity_fields = []
    for c in candidates:
        identity_fields.append({
            "document_id": c["document_id"],
            "candidate_name": c.get("candidate_name"),
            "email": c.get("email") or c.get("candidate_email"),
            "phone": c.get("phone") or c.get("candidate_phone"),
            "education": c.get("education") or c.get("candidate_education"),
            "work_experience": c.get("work_experience") or c.get("candidate_work_experience"),
        })

    resume_issues: list[dict] = []

    try:
        result: DuplicateDetectionResult = call_llm(
            system=DETECT_DUPLICATES_SYSTEM,
            user=f"Candidates in this batch:\n{identity_fields}",
            temperature=0.0,
            call_type="agent_detect_duplicates",
            response_model=DuplicateDetectionResult,
            structured_max_retries=1,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning("Duplicate detection failed - continuing without it", run_id=run_id, error=str(exc))
        return {"state_update": {"resume_issues": resume_issues}, "next_stage": "score_education"}

    name_by_doc = {c["document_id"]: c.get("candidate_name") or c["document_id"] for c in candidates}

    for group in result.groups:
        if len(group.document_ids) < 2:
            continue

        resume_issues.append({
            "issue_type": "duplicate",
            "document_ids": group.document_ids,
            "candidate_names": [name_by_doc.get(d, d) for d in group.document_ids],
            "detail": group.reasoning,
        })
        insert_agent_flag(
            run_id, "duplicate_resume",
            {"document_ids": group.document_ids, "reasoning": group.reasoning},
            severity="medium",
        )

        for contradiction in group.contradictions:
            resume_issues.append({
                "issue_type": "contradiction",
                "document_ids": group.document_ids,
                "candidate_names": [name_by_doc.get(d, d) for d in group.document_ids],
                "detail": contradiction,
            })
            insert_agent_flag(
                run_id, "contradiction",
                {"document_ids": group.document_ids, "detail": contradiction},
                severity="medium",
            )

    return {"state_update": {"resume_issues": resume_issues}, "next_stage": "score_education"}


# ---------------------------------------------------------------------------
# Stage: score_education
# ---------------------------------------------------------------------------

def score_education(run_id: str, state: dict) -> dict:
    plan, candidates, user_id = state["plan"], state["candidates"], state.get("_user_id", "system")

    trimmed = []
    for c in candidates:
        edu = get_field(
            c, "candidate_education",
            fallback_question="What is the candidate's educational background - institute, degree, branch, and years?",
            user_id=user_id,
        )
        trimmed.append({"document_id": c["document_id"], "education": edu["value"]})

    preference = plan["education_preference"] or (
        "No explicit institute/degree preference was stated - use general institute reputation and "
        "degree level as mild tie-breakers only."
    )
    scores = score_candidates(
        trimmed, "education",
        f"Education preference stated by the requester: {preference}",
        run_id,
        call_type="agent_score_education", user_id=user_id,
    )

    criterion_scores = {**state.get("criterion_scores", {}), "education": scores}
    return {"state_update": {"criterion_scores": criterion_scores}, "next_stage": "score_skills"}


# ---------------------------------------------------------------------------
# Stage: score_skills (chunked, grouped skill scoring)
# ---------------------------------------------------------------------------

def score_skills(run_id: str, state: dict) -> dict:
    plan, candidates, user_id = state["plan"], state["candidates"], state.get("_user_id", "system")

    trimmed = []
    for c in candidates:
        exp = get_field(
            c, "candidate_work_experience",
            fallback_question="List this candidate's key projects, tools, and technologies used, with concrete detail on what they built and their role.",
            user_id=user_id,
        )
        trimmed.append({"document_id": c["document_id"], "work_experience": exp["value"]})

    skill_scores = score_skill_groups(
        trimmed, plan["skill_groups"], run_id,
        call_type="agent_score_skill_group", user_id=user_id,
    )

    criterion_scores = {**state.get("criterion_scores", {}), **skill_scores}
    return {"state_update": {"criterion_scores": criterion_scores}, "next_stage": "score_experience"}


# ---------------------------------------------------------------------------
# Stage: score_experience
# ---------------------------------------------------------------------------

def score_experience(run_id: str, state: dict) -> dict:
    plan, candidates, user_id = state["plan"], state["candidates"], state.get("_user_id", "system")

    trimmed = []
    for c in candidates:
        exp = get_field(
            c, "candidate_work_experience",
            fallback_question="Summarize the candidate's total years of experience and seniority progression across roles.",
            user_id=user_id,
        )
        trimmed.append({"document_id": c["document_id"], "work_experience": exp["value"]})

    notes = plan["experience_notes"] or "No specific notes - evaluate overall relevant experience depth and career progression."
    scores = score_candidates(
        trimmed, "experience",
        f"General experience/seniority evaluation. Requester notes: {notes}",
        run_id,
        call_type="agent_score_experience", user_id=user_id,
    )

    criterion_scores = {**state.get("criterion_scores", {}), "experience": scores}
    return {"state_update": {"criterion_scores": criterion_scores}, "next_stage": "merge_scores"}


# ---------------------------------------------------------------------------
# Stage: merge_scores (the "judge" call - produces the INITIAL ranking,
# runs BEFORE claim verification now, on unverified data)
# ---------------------------------------------------------------------------

def merge_scores(run_id: str, state: dict) -> dict:
    plan, candidates, criterion_scores, user_id = (
        state["plan"], state["candidates"], state["criterion_scores"], state.get("_user_id", "system"),
    )

    per_candidate_summary = []
    for c in candidates:
        doc_id = c["document_id"]
        entry = {"document_id": doc_id, "candidate_name": c.get("candidate_name") or doc_id}
        for crit_name, crit_scores in criterion_scores.items():
            entry[crit_name] = crit_scores.get(doc_id, {"score": 0.0, "reasoning": "No score available."})
        per_candidate_summary.append(entry)

    skill_groups_text = ", ".join(f"{g['group_name']} ({g['requirement_level']})" for g in plan["skill_groups"])
    plan_text = (
        f"Education preference: {plan['education_preference'] or 'none stated'}\n"
        f"Skill groups evaluated: {skill_groups_text}\n"
        f"Experience notes: {plan['experience_notes'] or 'none stated'}\n"
        f"Other preferences: {plan['other_preferences'] or 'none stated'}"
    )

    try:
        result: RankingResult = call_llm(
            system=JUDGE_MERGE_SYSTEM,
            user=f"Requester's stated preferences:\n{plan_text}\n\nPer-criterion scores per candidate:\n{per_candidate_summary}",
            temperature=0.1,
            call_type="agent_merge",
            response_model=RankingResult,
            structured_max_retries=1,
            user_id=user_id,
        )
        ranked = [r.model_dump() for r in result.ranked]
    except Exception as exc:
        logger.error("Merge/judge call failed", run_id=run_id, error=str(exc))
        raise

    return {"state_update": {"ranked": ranked}, "next_stage": "verify_claims"}


# ---------------------------------------------------------------------------
# Stage: verify_claims (runs AFTER merge_scores, on a SUBSET chosen from
# the initial ranking; batched, parallel)
# ---------------------------------------------------------------------------

def _select_verification_subset(ranked_sorted: list[dict], plan: dict) -> list[str]:
    """
    Chooses which candidates get claim-verification, based on the initial
    ranking. If the requester asked for a specific top-N, verify roughly
    1.5x that many (some buffer above the cutoff); otherwise verify the
    top fraction of everyone (see optimization_config.py). This is a
    single-pass selection - no cascading re-check if verification later
    knocks someone out of the shown shortlist (accepted limitation).
    """
    total = len(ranked_sorted)
    requested_count = plan.get("requested_count")
    if requested_count:
        subset_size = math.ceil(requested_count * OPT.VERIFY_CLAIMS_TOP_N_MULTIPLIER)
    else:
        subset_size = math.ceil(total * OPT.VERIFY_CLAIMS_TOP_PERCENT)
    subset_size = max(1, min(subset_size, total))
    return [r["document_id"] for r in ranked_sorted[:subset_size]]


def _verify_claims_batch(candidates_batch: list[dict], user_id: str) -> list["ClaimVerification"]:
    """Runs one plausibility-check LLM call covering several candidates at once, evaluated independently. Returns [] on failure."""
    doc_ids = [c["document_id"] for c in candidates_batch]
    try:
        result: ClaimVerificationBatch = call_llm(
            system=VERIFY_CLAIMS_SYSTEM,
            user=f"Candidates in this batch:\n{candidates_batch}",
            temperature=0.0,
            call_type="agent_verify_claims",
            response_model=ClaimVerificationBatch,
            structured_max_retries=1,
            user_id=user_id,
        )
        return result.verifications
    except Exception as exc:
        logger.warning("Claim verification batch failed - skipping this batch", document_ids=doc_ids, error=str(exc))
        return []


def verify_claims(run_id: str, state: dict) -> dict:
    candidates, plan, ranked, user_id = (
        state["candidates"], state["plan"], state["ranked"], state.get("_user_id", "system"),
    )

    ranked_sorted = sorted(ranked, key=lambda r: r["rank"])
    subset_ids = _select_verification_subset(ranked_sorted, plan)

    by_doc = {c["document_id"]: c for c in candidates}
    subset_candidates = [by_doc[d] for d in subset_ids if d in by_doc]

    batches = [
        subset_candidates[i:i + OPT.VERIFY_CLAIMS_BATCH_SIZE]
        for i in range(0, len(subset_candidates), OPT.VERIFY_CLAIMS_BATCH_SIZE)
    ]

    claim_verifications: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_verify_claims_batch, batch, user_id) for batch in batches]
        for future in as_completed(futures):
            for cv in future.result():
                if not cv.suspicious_claims:
                    continue
                claims = [c.model_dump() for c in cv.suspicious_claims]
                claim_verifications[cv.document_id] = claims

                for claim in claims:
                    insert_agent_flag(
                        run_id, "suspicious_claim",
                        {
                            "field": claim["field"],
                            "claimed_text": claim["claimed_text"],
                            "reasoning": claim["reasoning"],
                        },
                        document_id=cv.document_id, severity=claim["confidence"],
                    )

    return {"state_update": {"claim_verifications": claim_verifications}, "next_stage": "compute_adjusted_scores"}


# ---------------------------------------------------------------------------
# Stage: compute_adjusted_scores (batched: multiple candidates AND multiple
# flagged criteria per candidate, in one call; capped total calls)
# ---------------------------------------------------------------------------

def _resolve_criterion_key(field: str, plan: dict) -> str | None:
    """
    Maps a suspicious-claim's free-text 'field' to an actual criterion_scores
    key ("education", "experience", or "skill:<group_name>"). Matches
    against both the skill group's own name and its member skill names,
    since a flagged claim's 'field' is likely to reference the original
    raw skill mention, not the merged group label. Returns None if nothing
    matches (claim is still noted in the issues table, just won't shift
    the score).
    """
    field_lower = field.lower()
    if "educat" in field_lower:
        return "education"
    if "experience" in field_lower:
        return "experience"
    for group in plan.get("skill_groups", []):
        if group["group_name"].lower() in field_lower:
            return f"skill:{group['group_name']}"
        for member in group.get("member_skills", []):
            if member.lower() in field_lower:
                return f"skill:{group['group_name']}"
    return None


def _adjust_batch(batch_payload: list[dict], user_id: str) -> list["AdjustedCriterionScore"]:
    """One LLM call re-scoring several (document_id, criterion) entries across possibly-multiple candidates. Returns [] on failure."""
    try:
        result: AdjustedScoreBatch = call_llm(
            system=ADJUSTED_SCORES_SYSTEM,
            user=f"Entries to re-score:\n{batch_payload}",
            temperature=0.0,
            call_type="agent_score_adjusted",
            response_model=AdjustedScoreBatch,
            structured_max_retries=1,
            user_id=user_id,
        )
        return result.adjustments
    except Exception as exc:
        logger.warning("Adjusted-score batch failed - skipping this batch", error=str(exc))
        return []


def compute_adjusted_scores(run_id: str, state: dict) -> dict:
    plan, criterion_scores, claim_verifications, ranked, user_id = (
        state["plan"], state["criterion_scores"], state.get("claim_verifications", {}),
        state["ranked"], state.get("_user_id", "system"),
    )

    if not claim_verifications:
        return {"state_update": {"adjusted_scores": {}}, "next_stage": "review_ranking"}

    # Build, per flagged candidate, the list of (criterion, adjusted_text,
    # original_score, reasoning) entries that need re-scoring.
    candidate_entries: dict[str, list[dict]] = {}
    for doc_id, claims in claim_verifications.items():
        entries = []
        for claim in claims:
            criterion_key = _resolve_criterion_key(claim["field"], plan)
            if criterion_key is None:
                continue
            original_score = criterion_scores.get(criterion_key, {}).get(doc_id, {}).get("score")
            if original_score is None:
                continue
            entries.append({
                "criterion_key": criterion_key,
                "adjusted_text": claim["adjusted_text"],
                "original_score": original_score,
                "reasoning": claim["reasoning"],
            })
        if entries:
            candidate_entries[doc_id] = entries

    if not candidate_entries:
        return {"state_update": {"adjusted_scores": {}}, "next_stage": "review_ranking"}

    ids_with_entries = list(candidate_entries.keys())
    n_flagged = len(ids_with_entries)
    batch_size = OPT.ADJUSTED_SCORES_BATCH_SIZE
    if n_flagged > batch_size * OPT.ADJUSTED_SCORES_MAX_CALLS:
        batch_size = math.ceil(n_flagged / OPT.ADJUSTED_SCORES_MAX_CALLS)

    n_criteria = 1 + len(plan["skill_groups"])  # education/experience share one bucket here; kept simple - approximation, not exact weighted math
    adjusted_scores: dict[str, dict] = {}

    for i in range(0, len(ids_with_entries), batch_size):
        batch_ids = ids_with_entries[i:i + batch_size]
        batch_payload = [
            {
                "document_id": doc_id,
                "entries": [
                    {"criterion": e["criterion_key"], "adjusted_text": e["adjusted_text"], "reasoning_for_removal": e["reasoning"]}
                    for e in candidate_entries[doc_id]
                ],
            }
            for doc_id in batch_ids
        ]

        adjustments = _adjust_batch(batch_payload, user_id)
        by_doc_adj: dict[str, dict[str, float]] = {}
        for a in adjustments:
            by_doc_adj.setdefault(a.document_id, {})[a.criterion] = a.adjusted_score

        for doc_id in batch_ids:
            total_delta = 0.0
            for entry in candidate_entries[doc_id]:
                adj_score = by_doc_adj.get(doc_id, {}).get(entry["criterion_key"], entry["original_score"])
                total_delta += (entry["original_score"] - adj_score)

            if total_delta > 0:
                overall = next((r["overall_score"] for r in ranked if r["document_id"] == doc_id), None)
                if overall is not None:
                    adjusted_scores[doc_id] = {
                        "score_without_flagged_claims": round(max(0.0, overall - (total_delta / max(n_criteria, 1))), 2),
                        "claims": claim_verifications[doc_id],
                    }

    return {"state_update": {"adjusted_scores": adjusted_scores}, "next_stage": "review_ranking"}


# ---------------------------------------------------------------------------
# Stage: review_ranking (consistency check, adjacent-swap corrections only,
# runs over the FULL ranking regardless of the verify_claims subset)
# ---------------------------------------------------------------------------

def review_ranking(run_id: str, state: dict) -> dict:
    ranked, criterion_scores, user_id = (
        state["ranked"], state["criterion_scores"], state.get("_user_id", "system"),
    )

    ranked_sorted = sorted(ranked, key=lambda r: r["rank"])

    try:
        review: RankingReview = call_llm(
            system=REVIEW_RANKING_SYSTEM,
            user=f"Ranked list:\n{ranked_sorted}\n\nPer-criterion scores per candidate:\n{criterion_scores}",
            temperature=0.0,
            call_type="agent_review_ranking",
            response_model=RankingReview,
            structured_max_retries=1,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning("Ranking review failed - keeping original ranking", run_id=run_id, error=str(exc))
        return {"state_update": {}, "next_stage": "finalize_response"}

    by_doc = {r["document_id"]: r for r in ranked_sorted}
    applied_swaps = []

    for swap in review.swaps:
        higher = by_doc.get(swap.document_id_higher)
        lower = by_doc.get(swap.document_id_lower)
        if higher is None or lower is None:
            continue
        # Only apply if they're actually adjacent in the current ranking -
        # protects against the model proposing a swap that isn't adjacent
        # despite the prompt instruction, since this is a hard safety
        # constraint, not just a request.
        if abs(higher["rank"] - lower["rank"]) != 1:
            logger.warning(
                "Ranking review proposed a non-adjacent swap - ignoring",
                run_id=run_id, higher=swap.document_id_higher, lower=swap.document_id_lower,
            )
            continue

        higher["rank"], lower["rank"] = lower["rank"], higher["rank"]
        applied_swaps.append({
            "document_id_higher": swap.document_id_higher,
            "document_id_lower": swap.document_id_lower,
            "reasoning": swap.reasoning,
        })

    if applied_swaps or review.notes:
        insert_agent_flag(
            run_id, "ranking_review",
            {"swaps_applied": applied_swaps, "notes": review.notes},
        )

    updated_ranked = sorted(by_doc.values(), key=lambda r: r["rank"])
    return {"state_update": {"ranked": updated_ranked}, "next_stage": "finalize_response"}


# ---------------------------------------------------------------------------
# Stage: finalize_response (fixed output contract)
# ---------------------------------------------------------------------------

def _get_project_highlights(candidates: list[dict], target_entries: list[dict], plan: dict, task: str, user_id: str = "system") -> dict:
    """
    One batched call extracting each shown candidate's best/most relevant
    project + technologies used + how the required skill(s) were applied.
    Only called for candidates actually shown to the requester (top +
    honorable mentions), not the full evaluated set - keeps this to a
    bounded, small number of extra LLM calls regardless of batch size.
    """
    if not target_entries:
        return {}

    by_doc = {c["document_id"]: c for c in candidates}
    skills_wanted = ", ".join(g["group_name"] for g in plan["skill_groups"])

    trimmed = []
    for e in target_entries:
        c = by_doc.get(e["document_id"], {"document_id": e["document_id"]})
        exp = get_field(
            c, "work_experience",
            fallback_question=f"List this candidate's key projects, the technologies/skills used in each, and how each relates to: {skills_wanted}.",
            user_id=user_id,
        )
        trimmed.append({"document_id": e["document_id"], "work_experience": exp["value"]})

    try:
        result: ProjectHighlightBatch = call_llm(
            system=PROJECT_HIGHLIGHT_SYSTEM,
            user=f"Requester's original request:\n{task}\n\nCandidates:\n{trimmed}",
            temperature=0.0,
            call_type="agent_project_highlight",
            response_model=ProjectHighlightBatch,
            structured_max_retries=1,
            user_id=user_id,
        )
        return {h.document_id: h.model_dump() for h in result.highlights}
    except Exception as exc:
        logger.warning("Project highlight extraction failed - continuing without it", error=str(exc))
        return {}


def finalize_response(run_id: str, state: dict) -> dict:
    plan, ranked, task, candidates, user_id = (
        state["plan"], state["ranked"], state["task"], state["candidates"], state.get("_user_id", "system"),
    )
    claim_verifications = state.get("claim_verifications", {})
    adjusted_scores = state.get("adjusted_scores", {})
    criterion_scores = state.get("criterion_scores", {})
    resume_issues = state.get("resume_issues", [])

    ranked_sorted = sorted(ranked, key=lambda r: r["rank"])
    requested_count = plan.get("requested_count")
    top = ranked_sorted[:requested_count] if requested_count else ranked_sorted

    # Honorable mentions: a small sample of candidates just outside the
    # shortlist, so the requester can see who almost made it (only
    # meaningful when a specific count was requested and more candidates
    # exist beyond it).
    honorable_mentions = []
    if requested_count and len(ranked_sorted) > len(top):
        honorable_mentions = ranked_sorted[len(top): len(top) + 3]

    highlights = _get_project_highlights(candidates, top + honorable_mentions, plan, task, user_id=user_id)

    def _enrich(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            h = highlights.get(e["document_id"], {})
            out.append({
                **e,
                "best_project": h.get("best_project", ""),
                "technologies": h.get("technologies", []),
                "skill_context": h.get("skill_context", ""),
                "flagged": e["document_id"] in claim_verifications,
            })
        return out

    top_enriched = _enrich(top)
    mentions_enriched = _enrich(honorable_mentions)

    try:
        summary = call_llm(
            system=FINALIZE_SUMMARY_SYSTEM,
            user=(
                f"Original request:\n{task}\n\n"
                f"Full ranking - ALL {len(ranked_sorted)} candidate(s) evaluated:\n{ranked_sorted}\n\n"
                f"Shown to requester as the shortlist (top {len(top)}):\n{[{'candidate_name': e['candidate_name'], 'rank': e['rank']} for e in top]}"
            ),
            temperature=0.3,
            call_type="agent_finalize_summary",
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning("Finalize summary call failed - using a plain fallback", run_id=run_id, error=str(exc))
        summary = f"Evaluated {len(ranked_sorted)} candidate(s); top {len(top)} shown below."

    # ---------------------------------------------------------------
    # Explainability: build findings with inline [n] citation markers,
    # backed by a single global citations list (one running counter
    # across the whole response, not reset per candidate). Citations
    # are sourced only from per-criterion evidence_snippets - the
    # judge's own comparative summary_reasoning stays uncited prose.
    # ---------------------------------------------------------------
    citations: list[dict] = []

    def _finding_line(e: dict) -> str:
        line = f"#{e['rank']} {e['candidate_name']} (document {e['document_id']}, score {e['overall_score']}): {e['summary_reasoning']}"

        for crit_name, crit_scores in criterion_scores.items():
            crit_entry = crit_scores.get(e["document_id"])
            if not crit_entry or not crit_entry.get("evidence_snippet"):
                continue
            citations.append({
                "number": len(citations) + 1,
                "candidate_name": e["candidate_name"],
                "criterion": crit_name,
                "snippet": crit_entry["evidence_snippet"],
            })
            line += f" [{len(citations)}]"

        if e.get("best_project"):
            line += f" Best project: {e['best_project']}"
            if e.get("technologies"):
                line += f" — technologies: {', '.join(e['technologies'])}"
            if e.get("skill_context"):
                line += f" ({e['skill_context']})"
        return line

    findings = [_finding_line(e) for e in top_enriched]

    data = [{"type": "table", "label": "Ranked candidates", "value": top_enriched}]
    if mentions_enriched:
        data.append({"type": "table", "label": "Honorable mentions (just outside the shortlist)", "value": mentions_enriched})

    if citations:
        data.append({"type": "table", "label": "Evidence citations", "value": citations})

    combined_issues = list(resume_issues)  # duplicates + contradictions, already built as issue rows

    if claim_verifications:
        name_by_doc = {c["document_id"]: c.get("candidate_name") or c["document_id"] for c in ranked_sorted}
        for doc_id, claims in claim_verifications.items():
            adjusted = adjusted_scores.get(doc_id, {})
            for claim in claims:
                combined_issues.append({
                    "issue_type": "suspicious_claim",
                    "document_ids": [doc_id],
                    "candidate_names": [name_by_doc.get(doc_id, doc_id)],
                    "detail": f"{claim['field']}: {claim['reasoning']} (confidence: {claim['confidence']})",
                    "score_without_flagged_claims": adjusted.get("score_without_flagged_claims"),
                })

    if combined_issues:
        data.append({"type": "table", "label": "Issues (duplicates, contradictions, suspicious claims)", "value": combined_issues})

    result = {
        "summary": summary,
        "findings": findings,
        "data": data,
        "extra": {"plan": plan, "criterion_scores": state.get("criterion_scores", {}), "all_ranked": ranked_sorted},
    }

    return {"state_update": {}, "next_stage": None, "result": result}


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------
STAGES = {
    "plan_evaluation":  plan_evaluation,
    "detect_duplicates_and_issues": detect_duplicates_and_issues,
    "score_education":  score_education,
    "score_skills":     score_skills,
    "score_experience": score_experience,
    "merge_scores":      merge_scores,
    "verify_claims":    verify_claims,
    "compute_adjusted_scores": compute_adjusted_scores,
    "review_ranking":    review_ranking,
    "finalize_response": finalize_response,
}

FIRST_STAGE = "plan_evaluation"