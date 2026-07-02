"""
core/quota_checker.py
Quota enforcement for DocIntel.

Quota resolution order (first match wins):
  1. User-specific quota
  2. Team quota (if user is in a team)
  3. Org quota (if user is in an org)
  4. System default (from config)

Quota types:
  max_documents        — total documents stored by user
  max_uploads_per_day  — uploads per calendar day
  max_llm_cost_month   — USD LLM spend per calendar month
  max_queries_per_day  — queries per calendar day

Hard limits  → block the action (HTTP 429)
Soft limits  → warn but allow (logged, not enforced)

Usage:
    from core.quota_checker import check_upload_quota, check_query_quota, check_llm_cost_quota

    # Before upload:
    check_upload_quota(user)

    # Before LLM call (in engine.py):
    check_llm_cost_quota(user, estimated_cost_usd=0.001)

    # Before query:
    check_query_quota(user)
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from supabase import create_client

from core.auth import UserContext
from core.config import config as app_config
from core.logger import get_logger

logger = get_logger("quota_checker")


# ---------------------------------------------------------------------------
# Supabase client (service role — reads all rows regardless of RLS)
# ---------------------------------------------------------------------------

def _get_sb():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Quota resolution
# ---------------------------------------------------------------------------

def _resolve_quota(
    user: UserContext,
    quota_type: str,
) -> tuple[float, bool]:
    """
    Resolve the effective quota limit for a user + quota_type.

    Returns (limit_value, is_hard_limit).
    Falls back to config defaults if no explicit quota is set.
    """
    try:
        sb = _get_sb()

        # Build OR filter: user_id match OR team_id match OR org_id match
        # Fetch all matching quotas and pick the most specific one
        rows = []

        # 1. User-specific
        resp = (
            sb.table("quotas")
            .select("limit_value, is_hard_limit, user_id, team_id, org_id")
            .eq("quota_type", quota_type)
            .eq("user_id", user.user_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            rows.append(("user", resp.data[0]))

        # 2. Team quota
        if user.team_id and not rows:
            resp = (
                sb.table("quotas")
                .select("limit_value, is_hard_limit, user_id, team_id, org_id")
                .eq("quota_type", quota_type)
                .eq("team_id", user.team_id)
                .is_("user_id", "null")
                .limit(1)
                .execute()
            )
            if resp.data:
                rows.append(("team", resp.data[0]))

        # 3. Org quota
        if user.org_id and not rows:
            resp = (
                sb.table("quotas")
                .select("limit_value, is_hard_limit, user_id, team_id, org_id")
                .eq("quota_type", quota_type)
                .eq("org_id", user.org_id)
                .is_("user_id", "null")
                .is_("team_id", "null")
                .limit(1)
                .execute()
            )
            if resp.data:
                rows.append(("org", resp.data[0]))

        if rows:
            _, row = rows[0]
            return float(row["limit_value"]), bool(row["is_hard_limit"])

    except Exception as e:
        logger.warning("Quota lookup failed — using defaults", error=str(e))

    # Fall back to config defaults
    defaults = {
        "max_documents":        (float(app_config.default_max_documents),       True),
        "max_uploads_per_day":  (float(app_config.default_max_uploads_per_day), True),
        "max_llm_cost_month":   (float(app_config.default_max_llm_cost_month),  True),
        "max_queries_per_day":  (float(app_config.default_max_queries_per_day), True),
    }
    return defaults.get(quota_type, (float("inf"), False))


# ---------------------------------------------------------------------------
# Usage counters
# ---------------------------------------------------------------------------

def _get_document_count(user_id: str) -> int:
    try:
        sb   = _get_sb()
        resp = (
            sb.table("documents")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        return resp.count or 0
    except Exception:
        return 0


def _get_uploads_today(user_id: str) -> int:
    try:
        sb    = _get_sb()
        today = date.today().isoformat()
        resp  = (
            sb.table("lineage_logs")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("event_type", "upload_received")
            .gte("created_at", f"{today}T00:00:00+00:00")
            .execute()
        )
        return resp.count or 0
    except Exception:
        return 0


def _get_llm_cost_this_month(user_id: str) -> float:
    try:
        sb    = _get_sb()
        month = datetime.now(timezone.utc).strftime("%Y-%m-01")
        resp  = (
            sb.table("llm_calls")
            .select("estimated_cost_usd")
            .eq("user_id", user_id)
            .gte("created_at", f"{month}T00:00:00+00:00")
            .execute()
        )
        return sum(
            float(r.get("estimated_cost_usd") or 0)
            for r in (resp.data or [])
        )
    except Exception:
        return 0.0


def _get_queries_today(user_id: str) -> int:
    try:
        sb    = _get_sb()
        today = date.today().isoformat()
        resp  = (
            sb.table("llm_calls")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("call_type", "query")
            .gte("created_at", f"{today}T00:00:00+00:00")
            .execute()
        )
        return resp.count or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Enforcement helpers
# ---------------------------------------------------------------------------

def _enforce(
    current: float,
    limit: float,
    is_hard: bool,
    quota_type: str,
    unit: str = "",
) -> None:
    if current < limit:
        return

    msg = (
        f"Quota exceeded: {quota_type}. "
        f"Current: {current}{unit}, Limit: {limit}{unit}."
    )

    if is_hard:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": msg, "code": "QUOTA_EXCEEDED", "quota_type": quota_type},
        )
    else:
        logger.warning("Soft quota exceeded — allowing through", quota_type=quota_type,
                       current=current, limit=limit)


# ---------------------------------------------------------------------------
# Public check functions
# ---------------------------------------------------------------------------

def check_upload_quota(user: UserContext) -> None:
    """
    Check upload quotas before accepting a file.
    Checks: max_documents, max_uploads_per_day.
    Raises HTTP 429 if a hard limit is exceeded.
    """
    if user.is_dev:
        return

    # max_documents
    doc_limit, doc_hard = _resolve_quota(user, "max_documents")
    doc_count = _get_document_count(user.user_id)
    _enforce(doc_count, doc_limit, doc_hard, "max_documents", " documents")

    # max_uploads_per_day
    up_limit, up_hard = _resolve_quota(user, "max_uploads_per_day")
    up_count = _get_uploads_today(user.user_id)
    _enforce(up_count, up_limit, up_hard, "max_uploads_per_day", " uploads today")


def check_query_quota(user: UserContext) -> None:
    """
    Check query quota before processing a query.
    Checks: max_queries_per_day.
    Raises HTTP 429 if a hard limit is exceeded.
    """
    if user.is_dev:
        return

    limit, is_hard = _resolve_quota(user, "max_queries_per_day")
    count = _get_queries_today(user.user_id)
    _enforce(count, limit, is_hard, "max_queries_per_day", " queries today")


def check_llm_cost_quota(user: UserContext, estimated_cost_usd: float = 0.0) -> None:
    """
    Check LLM cost quota before making an LLM call.
    Checks: max_llm_cost_month.
    Raises HTTP 429 if a hard limit is exceeded.

    Args:
        user:               Resolved UserContext.
        estimated_cost_usd: Estimated cost of the upcoming call in USD.
                            The check uses current spend + estimated cost.
    """
    if user.is_dev:
        return

    limit, is_hard  = _resolve_quota(user, "max_llm_cost_month")
    current_spend   = _get_llm_cost_this_month(user.user_id)
    projected_spend = current_spend + estimated_cost_usd

    _enforce(projected_spend, limit, is_hard, "max_llm_cost_month", " USD/month")


def get_quota_status(user: UserContext) -> dict:
    """
    Return full quota status for a user — used by GET /usage/me.
    Non-blocking: returns zeros on any DB error.
    """
    if user.is_dev:
        return {"quotas": [], "note": "Dev mode — no quota enforcement"}

    quotas = []
    for quota_type, current_fn, unit in [
        ("max_documents",       lambda: _get_document_count(user.user_id),    "documents"),
        ("max_uploads_per_day", lambda: _get_uploads_today(user.user_id),     "uploads"),
        ("max_llm_cost_month",  lambda: _get_llm_cost_this_month(user.user_id), "USD"),
        ("max_queries_per_day", lambda: _get_queries_today(user.user_id),     "queries"),
    ]:
        try:
            limit, is_hard = _resolve_quota(user, quota_type)
            current        = current_fn()
            quotas.append({
                "quota_type":  quota_type,
                "current":     current,
                "limit":       limit,
                "unit":        unit,
                "is_hard":     is_hard,
                "exceeded":    current >= limit,
                "pct_used":    round((current / limit * 100) if limit > 0 else 0, 1),
            })
        except Exception:
            quotas.append({"quota_type": quota_type, "error": "Could not retrieve"})

    return {"quotas": quotas}