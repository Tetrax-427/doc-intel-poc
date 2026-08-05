"""
backend/agents/registry.py
----------------------------
Single lookup point: agent_name (as used in POST /agents/{agent_name}/invoke)
-> {"stages": {...}, "first_stage": "...", "stage_descriptions": {...},
    "chat_tools": [...]}.

CHANGED (resume reshape): added an OPTIONAL third key, "stage_descriptions" —
{stage_name: "plain-English description of what this stage does"}. Used
only by agents/base.py's plan_resume_stage() when a completed run gets
new_input and there's no defined resume point (see base.py's module
docstring for why). Agents that never take new_input on a completed run
(cv_processor, so far) don't need to add this — get_agent_def() callers
that need it should read it defensively via
agent_def.get("stage_descriptions", {}), not agent_def["stage_descriptions"].

CHANGED (tool-calling chat): added an OPTIONAL fourth key, "chat_tools_factory" —
a callable(run_id, user_id) -> list[llm.tool_orchestrator.ToolSpec]. NOT a
static list — chat tools that need to know which run they're operating on
(e.g. ITR's recalculate_tax, scoped to one run's taxpayer_profile) can't be
built once at import time. Called fresh by routers/agents.py's chat route
right before each chat turn (see agents/chat.py's module docstring for how
the resulting list is then used). Agents that register none (cv_processor,
so far) get identical chat behaviour to before this change — read
defensively via agent_def.get("chat_tools_factory"), not
agent_def["chat_tools_factory"], and treat a missing/None factory as "no
tools" (empty list), not an error.

CHANGED (ITR agent): registered "itr_helper" — stages/first_stage/
stage_descriptions from agents.itr_helper.agent, chat_tools_factory from
agents.itr_helper.chat_tools.build_itr_chat_tools.

NOTE: cv_processor.agent doesn't exist yet (next thing to build) - this
import will fail until it does. Once agents/cv_processor/agent.py defines
STAGES and FIRST_STAGE, this file works as-is with no further changes.
"""
from agents.cv_processor.agent import STAGES as CV_PROCESSOR_STAGES, FIRST_STAGE as CV_PROCESSOR_FIRST_STAGE
from agents.itr_helper.agent import (
    STAGES as ITR_STAGES,
    FIRST_STAGE as ITR_FIRST_STAGE,
    STAGE_DESCRIPTIONS as ITR_STAGE_DESCRIPTIONS,
)
from agents.itr_helper.chat_tools import build_itr_chat_tools

AGENT_REGISTRY: dict[str, dict] = {
    "cv_processor": {
        "stages": CV_PROCESSOR_STAGES,
        "first_stage": CV_PROCESSOR_FIRST_STAGE,
        # No stage_descriptions — cv_processor runs never take new_input on a
        # completed run, so plan_resume_stage() is never called for it.
    },
    "itr_helper": {
        "stages": ITR_STAGES,
        "first_stage": ITR_FIRST_STAGE,
        "stage_descriptions": ITR_STAGE_DESCRIPTIONS,
        "chat_tools_factory": build_itr_chat_tools,
    },
}


def get_agent_def(agent_name: str) -> dict | None:
    return AGENT_REGISTRY.get(agent_name)