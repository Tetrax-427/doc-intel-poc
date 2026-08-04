"""
backend/agents/itr_helper/agent.py (partial — extract + stitch stages only)
------------------------------------------------------------------------------
Stage functions for the ITR agent, built on agents/base.py's stage machine.
Each stage function: def stage_fn(run_id: str, state: dict) -> StageOutcome.

This file will eventually hold the full STAGES dict (extract, stitch,
validate, calculate, summarize) + FIRST_STAGE, registered in
agents/registry.py. Only extract + stitch are built so far — validate/
calculate/summarize come next.
"""
from __future__ import annotations

from agents.base import StageOutcome
from agents.itr_helper.helpers import extract_tagged_documents, stitch_taxpayer_profile
from agents.itr_helper.calculator import compute_tax_comparison

def extract_stage(run_id: str, state: dict) -> StageOutcome:
    """
    Reads the user-tagged document lists from input_data["extra"] and runs
    the correct schema extraction against each. See helpers.extract_tagged_documents()
    for per-document error handling (one bad doc doesn't abort the run here —
    the validate stage decides if a resulting gap is fatal).
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