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