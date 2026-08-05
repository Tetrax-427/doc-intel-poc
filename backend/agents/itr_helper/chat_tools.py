"""
backend/agents/itr_helper/chat_tools.py
------------------------------------------
ITR-specific chat tools for run_tool_loop() (llm/tool_orchestrator.py).

CORRECTION vs original Phase A design: chat_tools can't be a static list in
agents/registry.py, because ITR's tools need to know WHICH run they're
operating on (whose taxpayer_profile to recalculate, whose documents can be
looked up) — a module-level static list has no way to carry a run_id.
Instead, this module exposes a FACTORY: build_itr_chat_tools(run_id, user_id)
-> list[ToolSpec], called fresh by the router right before each chat turn
(see routers/agents.py's updated send_chat_message()). agents/registry.py
stores this factory under "chat_tools_factory", not a static "chat_tools"
list — see registry.py's updated comment.

Both tools are READ-ONLY with respect to the run itself — neither persists
anything back to agent_runs.state. This matches chat.py's existing
constraint (chat cannot mutate a run; only the resume endpoint can) —
recalculate_tax returns fresh numbers for the conversation, it doesn't
silently update the stored result. If the person wants the stored result
updated, that's what the resume endpoint (new_input path) is for.
"""
from __future__ import annotations

from db import get_agent_run
from db_extraction import get_latest_extraction_for_document
from llm.tool_orchestrator import ToolSpec
from agents.itr_helper.calculator import compute_tax_comparison


def _recalculate_tax_executor(run_id: str, user_id: str):
    def _run(**kwargs) -> dict:
        run = get_agent_run(run_id, user_id=user_id)
        if run is None:
            return {"error": "Run not found."}
        profile = (run.get("state") or {}).get("taxpayer_profile", {})
        if not profile:
            return {"error": "No taxpayer profile available on this run yet."}
        return compute_tax_comparison(profile)
    return _run


def _lookup_extracted_doc_executor(run_id: str, user_id: str):
    def _run(document_id: str = "") -> dict:
        if not document_id:
            return {"error": "document_id is required."}

        run = get_agent_run(run_id, user_id=user_id)
        if run is None:
            return {"error": "Run not found."}

        # Only allow lookup of documents actually tagged on THIS run — a
        # chat message's document_id arg is model-parsed from free text, so
        # this stops it from being used to fetch an unrelated document the
        # person doesn't own/isn't part of this run's context.
        extra = (run.get("input_data") or {}).get("extra", {})
        tagged_ids = (
            extra.get("form16_doc_ids", [])
            + extra.get("fd_doc_ids", [])
            + extra.get("stocks_doc_ids", [])
        )
        if document_id not in tagged_ids:
            return {"error": f"Document '{document_id}' is not part of this run."}

        result = get_latest_extraction_for_document(document_id)
        if result is None:
            return {"error": f"No extraction result found for document '{document_id}'."}
        return result.get("result", {})
    return _run


def build_itr_chat_tools(run_id: str, user_id: str) -> list[ToolSpec]:
    """Called fresh per chat turn by the router — see module docstring for why."""
    return [
        ToolSpec(
            name="recalculate_tax",
            description=(
                "Recomputes tax under both the old and new regimes from this run's "
                "current taxpayer profile. Use this whenever the person asks about "
                "their tax figures, in case the underlying data has changed since "
                "the run last completed — never quote the original summary's numbers "
                "from memory when this tool is available."
            ),
            executor=_recalculate_tax_executor(run_id, user_id),
        ),
        ToolSpec(
            name="lookup_extracted_doc",
            description=(
                "Fetches the raw extracted data for one of this run's tagged "
                "documents (a Form16, FD certificate, or stocks statement), by "
                "document_id. Use this when the person asks about a specific "
                "document's details beyond what's in the summary."
            ),
            executor=_lookup_extracted_doc_executor(run_id, user_id),
            args_schema="document_id (string, required) — the document's id, "
                        "as referenced in this run's tagged document lists.",
        ),
    ]