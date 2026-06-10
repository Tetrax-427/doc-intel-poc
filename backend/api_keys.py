import os
import hashlib
import secrets
from datetime import date
from dotenv import load_dotenv
from db import supabase

load_dotenv()


def generate_api_key() -> tuple[str, str, str]:
    """Generate API key — returns (full_key, prefix, hash)"""
    key = f"di_{secrets.token_urlsafe(32)}"
    prefix = key[:10]
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, prefix, key_hash


def create_api_key(name: str, rate_limit: int = 100) -> dict:
    """Create and store a new API key"""
    key, prefix, key_hash = generate_api_key()

    supabase.table("api_keys").insert({
        "name": name,
        "key_hash": key_hash,
        "key_prefix": prefix,
        "rate_limit": rate_limit
    }).execute()

    return {
        "key": key,  # only returned once
        "prefix": prefix,
        "name": name,
        "rate_limit": rate_limit,
        "message": "Store this key safely — it will not be shown again."
    }


def validate_api_key(key: str) -> tuple[bool, str]:
    """Validate an API key — returns (is_valid, reason)"""
    try:
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        result = supabase.table("api_keys")\
            .select("*")\
            .eq("key_hash", key_hash)\
            .eq("is_active", True)\
            .execute()

        if not result.data:
            return False, "Invalid or inactive API key"

        record = result.data[0]

        # Reset daily counter if needed
        today = date.today().isoformat()
        if record.get("last_reset") != today:
            supabase.table("api_keys")\
                .update({"calls_today": 0, "last_reset": today})\
                .eq("id", record["id"])\
                .execute()
            record["calls_today"] = 0

        # Check rate limit
        if record["calls_today"] >= record["rate_limit"]:
            return False, f"Rate limit exceeded ({record['rate_limit']} calls/day)"

        # Increment counter
        supabase.table("api_keys")\
            .update({"calls_today": record["calls_today"] + 1})\
            .eq("id", record["id"])\
            .execute()

        return True, "valid"

    except Exception as e:
        print(f"API key validation error: {e}")
        return False, "Validation error"


def list_api_keys() -> list[dict]:
    """List all API keys (without hashes)"""
    result = supabase.table("api_keys")\
        .select("id, name, key_prefix, is_active, rate_limit, calls_today, created_at")\
        .order("created_at", desc=True)\
        .execute()
    return result.data or []


def revoke_api_key(key_id: str):
    """Revoke an API key"""
    supabase.table("api_keys")\
        .update({"is_active": False})\
        .eq("id", key_id)\
        .execute()