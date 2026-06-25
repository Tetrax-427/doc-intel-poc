"""
F2 — DB helpers for API key management.

Separates raw DB access from the business logic in api_keys.py,
matching the pattern established by db_lineage.py.

Public API:
    get_api_key_by_id()       — fetch one key record by UUID
    mark_api_key_rotating()   — set status='rotating' + grace_expires_at
"""

from __future__ import annotations

from db import supabase
from core.logger import get_logger

logger = get_logger("db_apikeys")


def get_api_key_by_id(key_id: str) -> dict | None:
    """
    Fetch a single API key record by its UUID.

    Returns the full row dict, or None if not found.
    Callers are responsible for ownership checks (user_id comparison).
    """
    try:
        result = (
            supabase.table("api_keys")
            .select("*")
            .eq("id", key_id)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.warning("get_api_key_by_id failed", key_id=key_id, error=str(exc))
        return None


def mark_api_key_rotating(key_id: str, grace_expires_at: str) -> None:
    """
    Mark an API key as 'rotating'.

    Sets:
        status=rotating              — key in grace period after rotation
        grace_expires_at=<ts>     — key stays valid until this timestamp

    validate_api_key() in api_keys.py checks grace_expires_at to decide
    whether the key is still in its grace period.

    Args:
        key_id:            UUID of the key to mark as rotating.
        grace_expires_at:  ISO-8601 UTC timestamp string (e.g. from datetime.isoformat()).
    """
    try:
        supabase.table("api_keys").update({
            "status":              "rotating",
            "grace_expires_at":     grace_expires_at,
        }).eq("id", key_id).execute()

        logger.info(
            "API key marked as rotating",
            key_id=key_id,
            grace_expires_at=grace_expires_at,
        )
    except Exception as exc:
        logger.error(
            "mark_api_key_rotating failed",
            key_id=key_id,
            error=str(exc),
        )
        raise