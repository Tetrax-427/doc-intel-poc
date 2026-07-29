"""
backend/agents/registry.py
----------------------------
Single lookup point: agent_name (as used in POST /agents/{agent_name}/invoke)
-> {"stages": {...}, "first_stage": "..."}.

NOTE: cv_processor.agent doesn't exist yet (next thing to build) - this
import will fail until it does. Once agents/cv_processor/agent.py defines
STAGES and FIRST_STAGE, this file works as-is with no further changes.
"""
from agents.cv_processor.agent import STAGES as CV_PROCESSOR_STAGES, FIRST_STAGE as CV_PROCESSOR_FIRST_STAGE

AGENT_REGISTRY: dict[str, dict] = {
    "cv_processor": {
        "stages": CV_PROCESSOR_STAGES,
        "first_stage": CV_PROCESSOR_FIRST_STAGE,
    },
    # "ca_helper": {...}  - added when built
}


def get_agent_def(agent_name: str) -> dict | None:
    return AGENT_REGISTRY.get(agent_name)