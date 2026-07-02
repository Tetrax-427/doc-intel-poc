"""
api_keys.py
API key management for DocIntel.

Changes in this phase:
  - create_api_key() accepts scope ('personal' | 'org') and org_id
  - org-scoped keys stored with org_id + scope='org' columns
  - validate_api_key() unchanged — scope is metadata only, not enforced differently

F2 (unchanged):
  - Keys have status: 'active' | 'rotating' | 'deleted'
  - rotate_api_key(): new key created, old marked rotating with grace period
  - validate_api_key(): accepts active + rotating-within-grace-period keys
"""

import hashlib
import secrets
from datetime import date, datetime, timezone, timedelta
from dotenv import load_dotenv
from db import get_supabase_admin

load_dotenv()


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Generate API key — returns (full_key, prefix, hash)."""
    key      = f"dik_{secrets.token_urlsafe(32)}"
    prefix   = key[:10]
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, prefix, key_hash


def create_api_key(
    name: str,
    rate_limit: int = 100,
    scope: str = "personal",
    org_id: str | None = None,
) -> dict:
    """
    Create and store a new API key.

    Args:
        name:       Human-readable name for the key.
        rate_limit: Max calls per day.
        scope:      'personal' (default) or 'org'.
        org_id:     Required when scope='org'. Links key to an org.

    Returns key details including the full key (shown once only).
    """
    if scope not in ("personal", "org"):
        scope = "personal"

    if scope == "org" and not org_id:
        raise ValueError("org_id is required when scope='org'")

    key, prefix, key_hash = generate_api_key()

    row: dict = {
        "name":       name,
        "key_hash":   key_hash,
        "key_prefix": prefix,
        "rate_limit": rate_limit,
        "status":     "active",
        "scope":      scope,
    }
    if org_id:
        row["org_id"] = org_id

    get_supabase_admin().table("api_keys").insert(row).execute()

    return {
        "key":        key,
        "prefix":     prefix,
        "name":       name,
        "rate_limit": rate_limit,
        "scope":      scope,
        "org_id":     org_id,
        "message":    "Store this key safely — it will not be shown again.",
    }


# ---------------------------------------------------------------------------
# Validation — unchanged
# ---------------------------------------------------------------------------

def validate_api_key(key: str) -> tuple[bool, str]:
    """
    Validate an API key — returns (is_valid, reason).

    Valid if:
      - status = 'active', OR
      - status = 'rotating' AND grace_expires_at > now
    """
    try:
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        result = (
            get_supabase_admin().table("api_keys")
            .select("*")
            .eq("key_hash", key_hash)
            .execute()
        )

        if not result.data:
            return False, "Invalid API key"

        record = result.data[0]
        status = record.get("status", "active")
        now    = datetime.now(timezone.utc)

        if status == "active":
            today = date.today().isoformat()
            if record.get("last_reset") != today:
                get_supabase_admin().table("api_keys").update({
                    "calls_today": 0,
                    "last_reset":  today,
                }).eq("id", record["id"]).execute()
                record["calls_today"] = 0

            if record["calls_today"] >= record["rate_limit"]:
                return False, f"Rate limit exceeded ({record['rate_limit']} calls/day)"

            get_supabase_admin().table("api_keys").update({
                "calls_today": record["calls_today"] + 1,
            }).eq("id", record["id"]).execute()
            return True, "valid"

        if status == "rotating":
            grace_expires_at = record.get("grace_expires_at")
            if grace_expires_at:
                try:
                    expiry = datetime.fromisoformat(
                        grace_expires_at.replace("Z", "+00:00")
                    )
                    if now < expiry:
                        return True, "valid_rotating"
                except Exception:
                    pass
            return False, "API key rotation grace period expired"

        return False, "Invalid or inactive API key"

    except Exception as e:
        print(f"API key validation error: {e}")
        return False, "Validation error"


# ---------------------------------------------------------------------------
# Rotation (F2) — unchanged
# ---------------------------------------------------------------------------

def rotate_api_key(key_id: str) -> dict:
    """
    Rotate an API key.
    New key created (active), old key marked rotating with grace period.
    Raises ValueError if key not found.
    """
    from db import get_api_key_by_id, mark_api_key_rotating
    from core.config import config as app_config

    old_record = get_api_key_by_id(key_id)
    if not old_record:
        raise ValueError(f"API key '{key_id}' not found")

    if old_record.get("status") == "deleted":
        raise ValueError(f"API key '{key_id}' has been deleted and cannot be rotated")

    grace_secs   = app_config.api_key_rotation_grace_period_seconds
    grace_expiry = (
        datetime.now(timezone.utc) + timedelta(seconds=grace_secs)
    ).isoformat()

    new_key, new_prefix, new_hash = generate_api_key()
    new_result = get_supabase_admin().table("api_keys").insert({
        "name":       old_record["name"] + " (rotated)",
        "key_hash":   new_hash,
        "key_prefix": new_prefix,
        "rate_limit": old_record.get("rate_limit", 100),
        "status":     "active",
        "scope":      old_record.get("scope", "personal"),
        "org_id":     old_record.get("org_id"),
    }).execute()
    new_id = new_result.data[0]["id"] if new_result.data else None

    mark_api_key_rotating(key_id, grace_expires_at=grace_expiry)

    if new_id:
        get_supabase_admin().table("api_keys").update({
            "rotated_to": new_id,
        }).eq("id", key_id).execute()

    return {
        "new_key":                 new_key,
        "new_prefix":              new_prefix,
        "new_key_id":              new_id,
        "old_key_id":              key_id,
        "old_key_prefix":          old_record.get("key_prefix", ""),
        "grace_period_seconds":    grace_secs,
        "grace_period_expires_at": grace_expiry,
        "message": (
            f"Old key remains valid for {grace_secs // 3600} hour(s). "
            "Update your integrations before it expires."
        ),
    }


# ---------------------------------------------------------------------------
# List / revoke — unchanged
# ---------------------------------------------------------------------------

def list_api_keys() -> list[dict]:
    result = (
        get_supabase_admin().table("api_keys")
        .select(
            "id, name, key_prefix, status, scope, org_id, rate_limit, "
            "calls_today, created_at, grace_expires_at, rotated_to"
        )
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def revoke_api_key(key_id: str) -> None:
    """Hard-revoke a key — instantly invalid, no grace period."""
    get_supabase_admin().table("api_keys").update({
        "status":           "deleted",
        "grace_expires_at": None,
    }).eq("id", key_id).execute()