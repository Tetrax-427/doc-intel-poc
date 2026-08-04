"""
routers/agents.py
Agent invocation, status polling, and resume endpoints.

Mirrors the async pattern already used for heavy operations (see
core/queue.py): invoke/resume return a run_id (here: agent_runs.id, not a
task_queue task_id) immediately; the actual multi-stage work runs on a
task_queue worker thread. The UI polls GET /agents/runs/{run_id} - not
GET /tasks/{task_id} - since agent progress lives in agent_runs (status,
current_stage, pending_questions, result), which is richer than task_queue's
generic pending/running/done/failed and supports the pause/resume flow.

CHANGED (resume reshape): ResumeAgentRequest now accepts EITHER `answers`
(existing needs_input case) OR `new_input` (completed run + new data, e.g.
a newly uploaded document folded into an already-finished run — the ITR
"add another doc, recalculate" case, but generic to any agent). Exactly one
must be given; which one determines which branch of
agent_base.accept_resume_input() runs. See agents/base.py's module
docstring for the full behaviour of each branch.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator, model_validator

from core.auth import get_current_user_context, get_user_id, UserContext
from core.responses import bad_request, not_found, internal_error
from core.queue import task_queue
from db import get_agent_run, list_agent_runs, get_agent_chat_history
from agents import chat as agent_chat 
from agents.registry import get_agent_def
from agents import base as agent_base

router = APIRouter(prefix="/agents", tags=["Agents"])


# ── Input models ──────────────────────────────────────────────────────────────

class InvokeAgentRequest(BaseModel):
    task: str
    document_ids: list[str]
    csv_data: list[dict] | None = None   # rows from the Extraction Helper's table, for these documents
    extra: dict | None = None            # room for agent-specific extra input beyond task/document_ids/csv_data
    name: str | None = None
    
    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v):
        if not v.strip():
            raise ValueError("task cannot be empty")
        return v.strip()

    @field_validator("document_ids")
    @classmethod
    def document_ids_not_empty(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        return v

class ResumeAgentRequest(BaseModel):
    """
    Exactly one of `answers` / `new_input` must be given — which one is
    valid depends on the run's current status (needs_input vs completed),
    checked in resume_run() below against the live run row, not here
    (this model doesn't have access to the run's status).
    """
    answers: dict | None = None
    new_input: dict | None = None

    @model_validator(mode="after")
    def exactly_one_payload(self):
        if bool(self.answers) == bool(self.new_input):
            # covers both-empty and both-given — either is invalid
            raise ValueError(
                "Provide exactly one of 'answers' (to answer pending questions) "
                "or 'new_input' (to add data to a completed run) — not both, not neither."
            )
        return self

class ChatMessageRequest(BaseModel):
    message: str
 
    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()
# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
def list_available_agents():
    """Returns the names of agents that can be invoked - lets a UI build an agent picker without hardcoding names."""
    from agents.registry import AGENT_REGISTRY
    return {"agents": list(AGENT_REGISTRY.keys())}


@router.post("/{agent_name}/invoke")
def invoke_agent(
    agent_name: str,
    req: InvokeAgentRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    agent_def = get_agent_def(agent_name)
    if agent_def is None:
        return not_found(f"Agent '{agent_name}'")

    input_data = {
        "document_ids": req.document_ids,
        "csv_data":     req.csv_data or [],
        "extra":        req.extra or {},
    }

    run_id = agent_base.create_and_queue_run(
        agent_name, req.task, input_data, user_id=uid, name=req.name,
    )

    task_queue.submit(
        f"agent:{agent_name}",
        agent_base.execute_stages,
        run_id, agent_def["stages"], agent_def["first_stage"],
        user_id=uid,
    )

    return {"run_id": run_id, "status": "pending"}


@router.get("/runs/{run_id}")
def get_run_status(
    run_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Polled by the UI. Returns the full agent_runs row: status, current_stage,
    pending_questions (when status == "needs_input"), result (when
    status == "completed"), error (when status == "failed").
    """
    uid = get_user_id(user)
    run = get_agent_run(run_id, user_id=uid)
    if run is None:
        return not_found(f"Agent run '{run_id}'")
    return run


@router.post("/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    req: ResumeAgentRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Handles both resume triggers:
      - status == "needs_input", req.answers given -> answers pending
        questions, re-enters at current_stage (unchanged from before).
      - status == "completed", req.new_input given -> merges new data into
        the run, planner picks the re-entry stage (see agents/base.py).
    Any other status, or a payload that doesn't match the run's status
    (e.g. new_input on a needs_input run), is rejected with 400.
    """
    uid = get_user_id(user)

    run = get_agent_run(run_id, user_id=uid)
    if run is None:
        return not_found(f"Agent run '{run_id}'")
    if run["status"] not in ("needs_input", "completed"):
        return bad_request(
            f"Agent run '{run_id}' cannot be resumed from status={run['status']}.",
            code="AGENT_RUN_NOT_RESUMABLE",
        )
    if run["status"] == "needs_input" and not req.answers:
        return bad_request(
            f"Agent run '{run_id}' is awaiting input — 'answers' is required to resume it.",
            code="AGENT_RESUME_ANSWERS_REQUIRED",
        )
    if run["status"] == "completed" and not req.new_input:
        return bad_request(
            f"Agent run '{run_id}' has already completed — 'new_input' is required to resume it.",
            code="AGENT_RESUME_NEW_INPUT_REQUIRED",
        )

    agent_def = get_agent_def(run["agent_name"])
    if agent_def is None:
        return internal_error(f"Agent '{run['agent_name']}' is no longer registered.")

    try:
        agent_base.accept_resume_input(
            run_id,
            user_id=uid,
            answers=req.answers,
            new_input=req.new_input,
            stage_descriptions=agent_def.get("stage_descriptions", {}),
        )
    except agent_base.AgentRunError as exc:
        return bad_request(str(exc), code="AGENT_RESUME_FAILED")

    task_queue.submit(
        f"agent:{run['agent_name']}:resume",
        agent_base.resume_stages,
        run_id, agent_def["stages"],
        user_id=uid,
    )

    return {"run_id": run_id, "status": "running"}


@router.get("/runs")
def list_runs(
    agent_name: str | None = None,
    status: str | None = None,
    limit: int = 50,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)
    return list_agent_runs(user_id=uid, agent_name=agent_name, status=status, limit=limit)


@router.post("/runs/{run_id}/chat")
def send_chat_message(
    run_id: str,
    req: ChatMessageRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Single Q&A chat turn, grounded in a completed run's result/state.
 
    Deliberately synchronous — NOT routed through task_queue like
    invoke/resume above. v1 chat is exactly one call_llm() call (see
    agents/chat.py), so there's no multi-stage work to background; the
    reply comes back in this same response.
 
    Returns 400 (AGENT_CHAT_UNAVAILABLE) if the run doesn't exist / isn't
    owned by this user, or isn't status == "completed" yet — chat is
    intentionally unavailable while a run is pending/running/needs_input/failed.

    CHANGED (tool-calling chat): looks up the run's agent_def and passes its
    registered chat_tools (if any) through to handle_chat_message(). Agents
    that register none (e.g. cv_processor) get the exact same plain-chat
    behaviour as before — this lookup is a no-op for them.
    """
    uid = get_user_id(user)

    run = get_agent_run(run_id, user_id=uid)
    if run is None:
        return not_found(f"Agent run '{run_id}'")

    agent_def = get_agent_def(run["agent_name"]) or {}
    chat_tools = agent_def.get("chat_tools", [])

    try:
        return agent_chat.handle_chat_message(run_id, req.message, user_id=uid, chat_tools=chat_tools)
    except agent_chat.AgentChatError as exc:
        return bad_request(str(exc), code="AGENT_CHAT_UNAVAILABLE")
 
 
@router.get("/runs/{run_id}/chat")
def get_chat_history(
    run_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """Full message history for a run's chat, oldest first."""
    uid = get_user_id(user)
    run = get_agent_run(run_id, user_id=uid)
    if run is None:
        return not_found(f"Agent run '{run_id}'")
    return {"messages": get_agent_chat_history(run_id, user_id=uid)}