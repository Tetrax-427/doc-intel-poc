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
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

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
    answers: dict

    @field_validator("answers")
    @classmethod
    def answers_not_empty(cls, v):
        if not v:
            raise ValueError("answers cannot be empty")
        return v

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

    run_id = agent_base.create_and_queue_run(agent_name, req.task, input_data, user_id=uid)

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
    uid = get_user_id(user)

    run = get_agent_run(run_id, user_id=uid)
    if run is None:
        return not_found(f"Agent run '{run_id}'")
    if run["status"] != "needs_input":
        return bad_request(
            f"Agent run '{run_id}' is not awaiting input (status={run['status']}).",
            code="AGENT_RUN_NOT_PAUSED",
        )

    agent_def = get_agent_def(run["agent_name"])
    if agent_def is None:
        return internal_error(f"Agent '{run['agent_name']}' is no longer registered.")

    try:
        agent_base.accept_resume_answers(run_id, req.answers, user_id=uid)
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
    """
    uid = get_user_id(user)
    try:
        return agent_chat.handle_chat_message(run_id, req.message, user_id=uid)
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