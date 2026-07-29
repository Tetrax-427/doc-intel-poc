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

from core.logger import get_logger
from llm.engine import call_llm
from agents.tools.candidate_data import build_candidate_dataset, get_field
from agents.tools.scoring import score_candidates
from agents.cv_processor.prompts import (
    PLAN_EVALUATION_SYSTEM,
    JUDGE_MERGE_SYSTEM,
    FINALIZE_SUMMARY_SYSTEM,
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
    rank: int = Field(description="1 = best")
    overall_score: float = Field(ge=0, le=10)
    summary_reasoning: str


class RankingResult(BaseModel):
    ranked: list[RankedCandidate]


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

    plan_draft_raw = state.get("plan_draft_raw")

    if plan_draft_raw is None:
        # First entry for this run.
        try:
            draft: PlanDraft = call_llm(
                system=PLAN_EVALUATION_SYSTEM,
                user=f"Task:\n{task}",
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

    return {"state_update": {"plan": plan, "candidates": candidates}, "next_stage": "score_education"}


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
        entry = {"document_id": doc_id}
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

    return {"state_update": {"ranked": ranked}, "next_stage": "finalize_response"}


# ---------------------------------------------------------------------------
# Stage: finalize_response (fixed output contract)
# ---------------------------------------------------------------------------

def finalize_response(run_id: str, state: dict) -> dict:
    plan, ranked, task, user_id = (
        state["plan"], state["ranked"], state["task"], state.get("_user_id", "system"),
    )

    ranked_sorted = sorted(ranked, key=lambda r: r["rank"])
    requested_count = plan.get("requested_count")
    top = ranked_sorted[:requested_count] if requested_count else ranked_sorted

    try:
        summary = call_llm(
            system=FINALIZE_SUMMARY_SYSTEM,
            user=f"Original request:\n{task}\n\nFinal ranking:\n{top}",
            temperature=0.3,
            call_type="agent_finalize_summary",
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning("Finalize summary call failed - using a plain fallback", run_id=run_id, error=str(exc))
        summary = f"Evaluated {len(ranked_sorted)} candidate(s); top {len(top)} shown below."

    findings = [f"#{r['rank']} (document {r['document_id']}): {r['summary_reasoning']}" for r in top]

    result = {
        "summary": summary,
        "findings": findings,
        "data": [{"type": "table", "label": "Ranked candidates", "value": top}],
        "extra": {"plan": plan, "criterion_scores": state.get("criterion_scores", {}), "all_ranked": ranked_sorted},
    }

    return {"state_update": {}, "next_stage": None, "result": result}


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------

STAGES = {
    "plan_evaluation":  plan_evaluation,
    "score_education":  score_education,
    "score_skills":     score_skills,
    "score_experience": score_experience,
    "merge_scores":      merge_scores,
    "finalize_response": finalize_response,
}
FIRST_STAGE = "plan_evaluation"