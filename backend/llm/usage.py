## DELETE THE FILES, ONCE EVERYTHING IS WORKING...

from datetime import datetime
from dotenv import load_dotenv
from db import supabase

load_dotenv()

# In-memory log (flushed to Supabase periodically)
_usage_log = []


def log_usage(call_type: str, model: str, prompt_len: int, response_len: int, latency: float):
    """Log an LLM call to memory and Supabase"""
    entry = {
        "call_type": call_type,
        "model": model,
        "prompt_chars": prompt_len,
        "response_chars": response_len,
        "estimated_tokens": (prompt_len + response_len) // 4,  # rough estimate
        "latency_ms": round(latency * 1000),
        "created_at": datetime.utcnow().isoformat()
    }
    _usage_log.append(entry)

    # Persist to Supabase
    try:
        supabase.table("usage_logs").insert(entry).execute()
    except Exception as e:
        pass  # fail silently — don't break the app over logging


def get_usage_summary() -> dict:
    """Return usage totals from in-memory log"""
    if not _usage_log:
        return {"total_calls": 0, "total_tokens": 0, "total_latency_ms": 0, "by_type": {}}

    total_tokens = sum(e["estimated_tokens"] for e in _usage_log)
    total_latency = sum(e["latency_ms"] for e in _usage_log)

    by_type = {}
    for e in _usage_log:
        t = e["call_type"]
        if t not in by_type:
            by_type[t] = {"calls": 0, "tokens": 0}
        by_type[t]["calls"] += 1
        by_type[t]["tokens"] += e["estimated_tokens"]

    return {
        "total_calls": len(_usage_log),
        "total_tokens": total_tokens,
        "avg_latency_ms": round(total_latency / len(_usage_log)),
        "by_type": by_type
    }