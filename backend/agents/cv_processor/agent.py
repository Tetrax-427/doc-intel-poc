"""
backend/agents/cv_processor/agent.py
----------------------------------------
CV Processor: ranks a batch of candidates (given as document_ids + optional
Extraction Helper CSV rows) against a free-text task/JD, following the
plan -> per-criterion score -> merge -> finalize pipeline.

Stage sequence:
    plan_evaluation -> score_education -> score_skills -> score_experience
    -> merge_scores -> finalize_response

plan_evaluation is the only stage that can pause the run (status
"needs_input") - see its docstring for how resume re-enters it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.logger import get_logger
from llm.engine import call_llm
from db import insert_agent_flag
from agents.tools.candidate_data import build_candidate_dataset, get_field
from agents.tools.scoring import score_candidates

from agents.cv_processor.prompts import (
    PLAN_EVALUATION_SYSTEM,
    DETECT_DUPLICATES_SYSTEM,
    VERIFY_CLAIMS_SYSTEM,
    JUDGE_MERGE_SYSTEM,
    REVIEW_RANKING_SYSTEM,
    FINALIZE_SUMMARY_SYSTEM,
    PROJECT_HIGHLIGHT_SYSTEM,
)

logger = get_logger("agents.cv_processor")


# ---------------------------------------------------------------------------
# Structured models for the two LLM calls that aren't per-criterion scoring
# (score_candidates in tools/scoring.py already defines its own model)
# ---------------------------------------------------------------------------

class SkillRequirement(BaseModel):
    name: str
    requirement_level: str = Field(description="'must' or 'nice'")


class Ambiguity(BaseModel):
    key: str
    question: str
    options: list[str] = []


class PlanDraft(BaseModel):
    education_preference: str
    skills: list[SkillRequirement]
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


def plan_evaluation(run_id: str, state: dict) -> dict:
    if "plan" in state:
        # Shouldn't normally happen (next_stage moves past this once plan is
        # set) but guards against any accidental re-entry.
        return {"next_stage": "score_education"}

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
# Stage: verify_claims
# ---------------------------------------------------------------------------

def _verify_one_candidate(candidate: dict, user_id: str) -> "ClaimVerification | None":
    """Runs one plausibility-check LLM call for a single candidate. Returns None on failure - skip, don't crash the batch."""
    doc_id = candidate["document_id"]
    try:
        result: ClaimVerification = call_llm(
            system=VERIFY_CLAIMS_SYSTEM,
            user=f"Candidate document_id={doc_id}:\n{ {k: v for k, v in candidate.items() if k != 'document_id'} }",
            temperature=0.0,
            call_type="agent_verify_claims",
            response_model=ClaimVerification,
            structured_max_retries=1,
            user_id=user_id,
        )
        return result
    except Exception as exc:
        logger.warning("Claim verification failed for a candidate - skipping", document_id=doc_id, error=str(exc))
        return None

# ---------------------------------------------------------------------------
# Stage: detect_duplicates_and_issues
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
        return {"state_update": {"resume_issues": resume_issues}, "next_stage": "verify_claims"}

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

    return {"state_update": {"resume_issues": resume_issues}, "next_stage": "verify_claims"}


def verify_claims(run_id: str, state: dict) -> dict:
    candidates, user_id = state["candidates"], state.get("_user_id", "system")

    claim_verifications: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_verify_one_candidate, c, user_id): c["document_id"] for c in candidates}
        for future in as_completed(futures):
            doc_id = futures[future]
            result = future.result()
            if result is None or not result.suspicious_claims:
                continue

            claims = [c.model_dump() for c in result.suspicious_claims]
            claim_verifications[doc_id] = claims

            for claim in claims:
                insert_agent_flag(
                    run_id, "suspicious_claim",
                    {
                        "field": claim["field"],
                        "claimed_text": claim["claimed_text"],
                        "reasoning": claim["reasoning"],
                    },
                    document_id=doc_id, severity=claim["confidence"],
                )

    return {"state_update": {"claim_verifications": claim_verifications}, "next_stage": "score_education"}

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
# Stage: score_skills (one score_candidates call per skill, all in one stage)
# ---------------------------------------------------------------------------

def score_skills(run_id: str, state: dict) -> dict:
    plan, candidates, user_id = state["plan"], state["candidates"], state.get("_user_id", "system")
    criterion_scores = dict(state.get("criterion_scores", {}))

    for skill in plan["skills"]:
        skill_name = skill["name"]
        trimmed = []
        for c in candidates:
            exp = get_field(
                c, "candidate_work_experience",
                fallback_question=f"What has this candidate specifically worked on related to '{skill_name}'? Include concrete projects/tools/duration if mentioned.",
                user_id=user_id,
            )
            trimmed.append({"document_id": c["document_id"], "work_experience": exp["value"]})

        if skill["requirement_level"] == "must":
            weight_note = (
                "This is a MUST-HAVE skill - a candidate with no evidence of it should score very low (0-2). "
                "This will be weighted heavily in the final ranking, though not as an automatic disqualifier."
            )
        else:
            weight_note = (
                "This is a NICE-TO-HAVE skill - lack of evidence should only mildly lower the score, not tank it."
            )

        scores = score_candidates(
            trimmed, f"skill: {skill_name}",
            f"Evaluate evidence of experience with '{skill_name}'. {weight_note}",
            run_id,
            call_type="agent_score_skill", user_id=user_id,
        )
        criterion_scores[f"skill:{skill_name}"] = scores

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
# Stage: merge_scores (the "judge" call)
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

    skills_text = ", ".join(f"{s['name']} ({s['requirement_level']})" for s in plan["skills"])
    plan_text = (
        f"Education preference: {plan['education_preference'] or 'none stated'}\n"
        f"Skills evaluated: {skills_text}\n"
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

    return {"state_update": {"ranked": ranked}, "next_stage": "compute_adjusted_scores"}


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
    skills_wanted = ", ".join(s["name"] for s in plan["skills"])

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


# ---------------------------------------------------------------------------
# Stage: compute_adjusted_scores (only runs for flagged candidates)
# ---------------------------------------------------------------------------

def _resolve_criterion_key(field: str, plan: dict) -> str | None:
    """
    Maps a suspicious-claim's free-text 'field' to an actual criterion_scores
    key ("education", "experience", or "skill:<name>"). Best-effort substring
    match against the plan's known skill names and the fixed education/
    experience labels - returns None if nothing matches (claim is noted in
    the flagged-claims table either way, it just won't shift the score).
    """
    field_lower = field.lower()
    if "educat" in field_lower:
        return "education"
    if "experience" in field_lower:
        return "experience"
    for skill in plan["skills"]:
        if skill["name"].lower() in field_lower:
            return f"skill:{skill['name']}"
    return None


def compute_adjusted_scores(run_id: str, state: dict) -> dict:
    plan, candidates, criterion_scores, claim_verifications, user_id = (
        state["plan"], state["candidates"], state["criterion_scores"],
        state.get("claim_verifications", {}), state.get("_user_id", "system"),
    )

    if not claim_verifications:
        return {"state_update": {"adjusted_scores": {}}, "next_stage": "finalize_response"}

    by_doc = {c["document_id"]: c for c in candidates}
    n_criteria = 1 + len(plan["skills"])  # education + experience share one bucket here; kept simple - see note below
    adjusted_scores: dict[str, dict] = {}

    for doc_id, claims in claim_verifications.items():
        candidate = by_doc.get(doc_id)
        if candidate is None:
            continue

        total_delta = 0.0
        for claim in claims:
            criterion_key = _resolve_criterion_key(claim["field"], plan)
            if criterion_key is None:
                continue

            original_score = criterion_scores.get(criterion_key, {}).get(doc_id, {}).get("score")
            if original_score is None:
                continue

            # Re-score this one criterion for this one candidate using the
            # adjusted (claim-removed) text instead of the original field value.
            adjusted_candidate = {"document_id": doc_id, criterion_key: claim["adjusted_text"]}
            adjusted_result = score_candidates(
                [adjusted_candidate], criterion_key,
                f"Re-scoring with a flagged claim removed: {claim['reasoning']}",
                run_id, call_type="agent_score_adjusted", user_id=user_id,
            )
            adjusted_score = adjusted_result.get(doc_id, {}).get("score", original_score)
            total_delta += (original_score - adjusted_score)

        if total_delta > 0:
            # Simple proportional shift: total criterion-count-weighted delta,
            # applied against the candidate's overall (0-10) score. Not a
            # full re-judge - see design note in the docstring above.
            overall = next((r["overall_score"] for r in state.get("ranked", []) if r["document_id"] == doc_id), None)
            if overall is not None:
                adjusted_scores[doc_id] = {
                    "score_without_flagged_claims": round(max(0.0, overall - (total_delta / max(n_criteria, 1))), 2),
                    "claims": claims,
                }

    return {"state_update": {"adjusted_scores": adjusted_scores}, "next_stage": "review_ranking"}


# ---------------------------------------------------------------------------
# Stage: review_ranking (consistency check, adjacent-swap corrections only)
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
    "verify_claims":    verify_claims,
    "score_education":  score_education,
    "score_skills":     score_skills,
    "score_experience": score_experience,
    "merge_scores":      merge_scores,
    "compute_adjusted_scores": compute_adjusted_scores,
    "review_ranking":    review_ranking,
    "finalize_response": finalize_response,
}

FIRST_STAGE = "plan_evaluation"