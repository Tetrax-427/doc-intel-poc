"""
backend/agents/itr_helper/agent.py
------------------------------------
Full ITR filing helper agent — stage pipeline built on agents/base.py's
stage machine.

Pipeline: extract -> stitch -> validate -> calculate -> summarize

  extract    Runs schema-tagged extraction (Form16/FD/stocks) per user-
             tagged document lists in input_data["extra"].
  stitch     Merges extraction results into one canonical taxpayer_profile
             (pure merge/normalise, no computation).
  validate   Basic completeness check (at least one Form16 present) +
             auto-detects ITR1 vs ITR2 from stock_transactions presence.
             Pauses (questions) if no Form16 found, resumable.
  calculate  Runs compute_tax_comparison() — pure Python, both regimes,
             no LLM.
  summarize  Terminal stage — single grounded LLM call narrating the
             already-computed numbers into agent_runs.result.

STAGES/FIRST_STAGE/STAGE_DESCRIPTIONS below are what agents/registry.py
imports to register this agent. stage_descriptions is used by
agents/base.py's plan_resume_stage() when a completed run gets new_input
(e.g. user uploads another document) — it picks the earliest stage whose
output could change given the new data.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base import StageOutcome
from agents.itr_helper.helpers import extract_tagged_documents, stitch_taxpayer_profile
from agents.itr_helper.calculator import compute_tax_comparison
from llm.engine import call_llm


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

def extract_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Reads the user-tagged document lists from input_data["extra"] and runs
    the correct schema extraction against each. See
    helpers.extract_tagged_documents() for per-document error handling (one
    bad doc doesn't abort the run here — the validate stage decides if a
    resulting gap is fatal).
    """
    extra = state.get("input_data", {}).get("extra", {})

    extractions = extract_tagged_documents(
        form16_doc_ids=extra.get("form16_doc_ids", []),
        fd_doc_ids=extra.get("fd_doc_ids", []),
        stocks_doc_ids=extra.get("stocks_doc_ids", []),
        user_id=state.get("_user_id", "system"),
    )

    return StageOutcome(
        state_update={"extractions": extractions},
        next_stage="stitch",
    )


# ---------------------------------------------------------------------------
# Stage: stitch
# ---------------------------------------------------------------------------

def stitch_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Merges state["extractions"] (set by extract_stage) into the canonical
    taxpayer_profile. Pure merge/normalise — no computation, per
    helpers.stitch_taxpayer_profile()'s docstring.
    """
    profile = stitch_taxpayer_profile(state.get("extractions", {}))

    return StageOutcome(
        state_update={"taxpayer_profile": profile},
        next_stage="validate",
    )


# ---------------------------------------------------------------------------
# Stage: validate
# ---------------------------------------------------------------------------

def validate_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Basic completeness check (per current decision: simple, not strict).
    Auto-detects ITR1 vs ITR2 from the profile itself — ITR2 is required
    if there are any capital gains (stock_transactions non-empty);
    otherwise ITR1 suffices. This detection is informational (stored in
    state, surfaced in the summary) — it does not gate anything in v1
    since the calculator handles both cases identically.

    Only hard requirement: at least one Form16 entry — there's no income
    to compute tax on otherwise. FD/stocks are optional additions, not
    required, per current decision. If a user already answered this via
    user_answers (resume path), skip re-asking.
    """
    profile = state.get("taxpayer_profile", {})
    form16_entries = profile.get("form16_entries", [])
    stock_transactions = profile.get("stock_transactions", [])

    itr_type = "ITR2" if stock_transactions else "ITR1"

    if not form16_entries:
        if state.get("user_answers", {}).get("form16_confirmed_absent"):
            # User explicitly confirmed they have no salary income this year —
            # proceed anyway rather than looping forever.
            return StageOutcome(
                state_update={"itr_type": itr_type},
                next_stage="calculate",
            )
        return StageOutcome(
            state_update={"itr_type": itr_type},
            questions=[{
                "key": "form16_confirmed_absent",
                "question": (
                    "No Form16 was found among the uploaded documents, so there's "
                    "no salary income to calculate tax on. Upload a Form16, or "
                    "confirm you have no salary income this year to proceed anyway."
                ),
                "type": "confirm",
            }],
        )

    return StageOutcome(
        state_update={"itr_type": itr_type},
        next_stage="calculate",
    )


# ---------------------------------------------------------------------------
# Stage: calculate
# ---------------------------------------------------------------------------

def calculate_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Runs compute_tax_comparison() (pure Python, both regimes) over the
    stitched profile. No LLM call here — see calculator.py's module
    docstring for why every number must be grounded/computed.
    """
    profile = state.get("taxpayer_profile", {})
    tax_calculation = compute_tax_comparison(profile)

    return StageOutcome(
        state_update={"tax_calculation": tax_calculation},
        next_stage="summarize",
    )


# ---------------------------------------------------------------------------
# Stage: summarize (terminal)
# ---------------------------------------------------------------------------

ITR_SUMMARY_SYSTEM = """\
You are summarizing a completed Indian income tax (ITR) calculation for the \
person who requested it. You will be given the exact computed figures for both \
the old and new tax regimes, the recommended regime, and the amount saved by \
choosing it — all already computed in Python, not by you.

Your job is narration and explanation only. Never state a rupee figure, \
percentage, or any other number that was not given to you directly in the \
provided data — do not calculate, estimate, or round anything yourself. If you \
want to explain WHY one regime is cheaper (e.g. "the new regime's lower slab \
rates outweigh the deductions you'd lose"), explain the mechanism in words \
without inventing supporting numbers.

Write:
  - summary: 2-3 sentences — which regime is recommended and the headline saving.
  - findings: 3-6 short bullet-point-style strings — key facts from the computed \
    data (e.g. taxable income under each regime, capital gains tax if any, \
    which deductions were counted). Every figure mentioned must come from the \
    provided data.
  - deduction_note: one or two sentences on Chapter VI-A deductions claimed \
    (only mention this if old-regime deductions data was provided) — do not \
    suggest a specific additional amount to invest, since that figure wasn't \
    computed for you; you may only note qualitatively that further deductions \
    could improve the old-regime comparison if the person wants to explore that.
"""


class ITRSummaryResult(BaseModel):
    summary: str = Field(description="2-3 sentence headline summary of the recommendation and saving.")
    findings: list[str] = Field(description="3-6 short factual findings, each grounded in the provided data.")
    deduction_note: str = Field(default="", description="Brief, qualitative note on deductions — no invented figures.")


def _format_regime(label: str, regime: dict) -> str:
    return (
        f"{label} regime:\n"
        f"  Taxable salary income: {regime['taxable_salary_income']}\n"
        f"  Taxable other-source income: {regime['taxable_other_source_income']}\n"
        f"  Slab-taxable income: {regime['slab_taxable_income']}\n"
        f"  Tax after rebate: {regime['tax_after_rebate']}\n"
        f"  Capital gains tax: {regime['capital_gains_tax']}\n"
        f"  Surcharge: {regime['surcharge']}\n"
        f"  Cess: {regime['cess']}\n"
        f"  Total tax payable: {regime['total_tax_payable']}"
    )


def summarize_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Terminal stage. Single LLM call, response_model-structured (Instructor,
    same pattern as everywhere else), strictly narrating numbers already
    computed by calculator.py — never inventing new figures.

    next_stage=None + result= makes this the terminal stage — base.py's
    _run_stage_loop() marks the run "completed" here.
    """
    tax_calc = state.get("tax_calculation", {})
    profile = state.get("taxpayer_profile", {})
    itr_type = state.get("itr_type", "ITR1")

    user_content = (
        f"Assessment year: {tax_calc.get('assessment_year')}\n"
        f"Detected ITR type: {itr_type}\n\n"
        f"{_format_regime('Old', tax_calc.get('old_regime', {}))}\n\n"
        f"{_format_regime('New', tax_calc.get('new_regime', {}))}\n\n"
        f"Recommended regime: {tax_calc.get('recommended_regime')}\n"
        f"Savings by choosing the recommended regime: {tax_calc.get('savings_amount')}\n\n"
        f"Number of Form16 entries (employers): {len(profile.get('form16_entries', []))}\n"
        f"Number of FD/interest certificates: {len(profile.get('fd_interest_entries', []))}\n"
        f"Number of stock transactions: {len(profile.get('stock_transactions', []))}\n"
        f"Extraction errors (if any, documents that failed to process): "
        f"{profile.get('extraction_errors', [])}"
    )

    result: ITRSummaryResult = call_llm(
        system=ITR_SUMMARY_SYSTEM,
        user=user_content,
        temperature=0.2,
        call_type="itr_summarize",
        response_model=ITRSummaryResult,
        user_id=state.get("_user_id", "system"),
    )

    data_rows = [
        {"type": "regime", "label": "Old regime total tax", "value": tax_calc.get("old_regime", {}).get("total_tax_payable")},
        {"type": "regime", "label": "New regime total tax", "value": tax_calc.get("new_regime", {}).get("total_tax_payable")},
        {"type": "recommendation", "label": "Recommended regime", "value": tax_calc.get("recommended_regime")},
        {"type": "savings", "label": "Savings amount", "value": tax_calc.get("savings_amount")},
    ]

    final_result = {
        "summary": result.summary,
        "findings": result.findings,
        "data": data_rows,
        "extra": {
            "itr_type": itr_type,
            "tax_calculation": tax_calc,
            "deduction_note": result.deduction_note,
            "taxpayer_profile": profile,
        },
    }

    return StageOutcome(
        state_update={},
        next_stage=None,
        result=final_result,
    )


# ---------------------------------------------------------------------------
# Registration surface — imported by agents/registry.py
# ---------------------------------------------------------------------------

STAGES = {
    "extract": extract_stage,
    "stitch": stitch_stage,
    "validate": validate_stage,
    "calculate": calculate_stage,
    "summarize": summarize_stage,
}

FIRST_STAGE = "extract"

# Used by agents/base.py's plan_resume_stage() when a completed run gets
# new_input (e.g. user adds another document) — the planner picks the
# earliest stage whose output could change given what's new.
STAGE_DESCRIPTIONS = {
    "extract": "Runs document extraction against the tagged Form16/FD/stocks documents.",
    "stitch": "Merges all extracted document data into one combined taxpayer profile.",
    "validate": "Checks the profile has at least a Form16 present and detects ITR1 vs ITR2.",
    "calculate": "Computes tax under both the old and new regimes from the taxpayer profile.",
    "summarize": "Writes the final summary and recommendation from the computed tax figures.",
}