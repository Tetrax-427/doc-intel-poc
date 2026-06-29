"""
db_usage.py
Usage aggregation queries for DocIntel.

Reads from the views created in 007_quotas.sql:
  user_daily_usage, user_monthly_usage, user_document_counts,
  user_daily_uploads, team_monthly_usage, org_monthly_usage

All queries use the service-role client — usage data is aggregated
server-side after permission checks in the router layer.

Usage:
    from db_usage import get_user_usage, get_team_usage, get_org_usage
"""

import os
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


def _sb():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


def _current_month() -> str:
    """Return first day of current month as ISO string e.g. '2025-01-01'."""
    now = datetime.now(timezone.utc)
    return f"{now.year}-{now.month:02d}-01"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# User usage
# ---------------------------------------------------------------------------

def get_user_usage(user_id: str) -> dict:
    """
    Return usage summary for a single user covering:
      - Current month LLM usage (cost, calls, cache hits)
      - Today's uploads
      - Total document count
      - Today's queries

    Non-blocking — returns zeros on any DB error.
    """
    sb = _sb()

    # Monthly LLM usage
    monthly = {"total_cost_usd": 0.0, "total_llm_calls": 0, "cache_hits": 0, "billable_cost_usd": 0.0}
    try:
        resp = (
            sb.table("user_monthly_usage")
            .select("*")
            .eq("user_id", user_id)
            .gte("usage_month", _current_month())
            .limit(1)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            monthly = {
                "total_cost_usd":    float(row.get("total_cost_usd") or 0),
                "total_llm_calls":   int(row.get("total_llm_calls") or 0),
                "cache_hits":        int(row.get("cache_hits") or 0),
                "billable_cost_usd": float(row.get("billable_cost_usd") or 0),
            }
    except Exception:
        pass

    # Daily uploads
    uploads_today = 0
    try:
        resp = (
            sb.table("user_daily_uploads")
            .select("uploads")
            .eq("user_id", user_id)
            .eq("upload_date", _today())
            .limit(1)
            .execute()
        )
        if resp.data:
            uploads_today = int(resp.data[0].get("uploads") or 0)
    except Exception:
        pass

    # Document count
    doc_count = 0
    try:
        resp = (
            sb.table("user_document_counts")
            .select("total_documents, private_documents, team_documents, org_documents")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            row       = resp.data[0]
            doc_count = int(row.get("total_documents") or 0)
    except Exception:
        pass

    # Queries today
    queries_today = 0
    try:
        resp = (
            sb.table("user_daily_usage")
            .select("chat_queries")
            .eq("user_id", user_id)
            .eq("usage_date", _today())
            .limit(1)
            .execute()
        )
        if resp.data:
            queries_today = int(resp.data[0].get("chat_queries") or 0)
    except Exception:
        pass

    return {
        "user_id":        user_id,
        "period":         "current_month",
        "monthly_llm":    monthly,
        "uploads_today":  uploads_today,
        "queries_today":  queries_today,
        "document_count": doc_count,
    }


def get_user_daily_breakdown(user_id: str, days: int = 30) -> list[dict]:
    """
    Return per-day usage breakdown for a user (last N days).
    Used for usage charts in the frontend.
    """
    try:
        sb = _sb()
        from datetime import timedelta, date
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = (
            sb.table("user_daily_usage")
            .select("usage_date, chat_queries, extractions, total_llm_calls, total_cost_usd, cache_hits")
            .eq("user_id", user_id)
            .gte("usage_date", cutoff)
            .order("usage_date", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Team usage
# ---------------------------------------------------------------------------

def get_team_usage(team_id: str) -> dict:
    """
    Return usage summary for a team for the current month.
    Includes per-member breakdown.
    """
    sb = _sb()

    # Team monthly rollup
    team_monthly = {
        "total_cost_usd": 0.0,
        "total_llm_calls": 0,
        "cache_hits": 0,
        "active_members": 0,
    }
    try:
        resp = (
            sb.table("team_monthly_usage")
            .select("*")
            .eq("team_id", team_id)
            .gte("usage_month", _current_month())
            .limit(1)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            team_monthly = {
                "total_cost_usd":  float(row.get("total_cost_usd") or 0),
                "total_llm_calls": int(row.get("total_llm_calls") or 0),
                "cache_hits":      int(row.get("cache_hits") or 0),
                "active_members":  int(row.get("active_members") or 0),
            }
    except Exception:
        pass

    # Per-member breakdown
    members = []
    try:
        resp = (
            sb.table("user_monthly_usage")
            .select("user_id, total_cost_usd, total_llm_calls, cache_hits")
            .eq("team_id", team_id)
            .gte("usage_month", _current_month())
            .execute()
        )
        members = resp.data or []
    except Exception:
        pass

    return {
        "team_id":       team_id,
        "period":        "current_month",
        "monthly_total": team_monthly,
        "members":       members,
    }


# ---------------------------------------------------------------------------
# Org usage
# ---------------------------------------------------------------------------

def get_org_usage(org_id: str) -> dict:
    """
    Return usage summary for an org for the current month.
    Includes per-team and per-member breakdown.
    """
    sb = _sb()

    # Org monthly rollup
    org_monthly = {
        "total_cost_usd": 0.0,
        "total_llm_calls": 0,
        "cache_hits": 0,
        "billable_cost_usd": 0.0,
        "active_members": 0,
        "active_teams": 0,
    }
    try:
        resp = (
            sb.table("org_monthly_usage")
            .select("*")
            .eq("org_id", org_id)
            .gte("usage_month", _current_month())
            .limit(1)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            org_monthly = {
                "total_cost_usd":    float(row.get("total_cost_usd") or 0),
                "total_llm_calls":   int(row.get("total_llm_calls") or 0),
                "cache_hits":        int(row.get("cache_hits") or 0),
                "billable_cost_usd": float(row.get("billable_cost_usd") or 0),
                "active_members":    int(row.get("active_members") or 0),
                "active_teams":      int(row.get("active_teams") or 0),
            }
    except Exception:
        pass

    # Per-team breakdown
    teams = []
    try:
        resp = (
            sb.table("team_monthly_usage")
            .select("team_id, total_cost_usd, total_llm_calls, active_members")
            .eq("org_id", org_id)
            .gte("usage_month", _current_month())
            .execute()
        )
        teams = resp.data or []
    except Exception:
        pass

    # Per-member breakdown
    members = []
    try:
        resp = (
            sb.table("user_monthly_usage")
            .select("user_id, total_cost_usd, total_llm_calls, cache_hits")
            .eq("org_id", org_id)
            .gte("usage_month", _current_month())
            .execute()
        )
        members = resp.data or []
    except Exception:
        pass

    return {
        "org_id":        org_id,
        "period":        "current_month",
        "monthly_total": org_monthly,
        "teams":         teams,
        "members":       members,
    }