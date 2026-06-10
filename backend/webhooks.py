import os
import hmac
import json
import hashlib
import httpx
from datetime import datetime
from dotenv import load_dotenv
from db import supabase

load_dotenv()


def sign_payload(payload: str, secret: str) -> str:
    """Generate HMAC signature for webhook payload"""
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()


def get_active_webhooks(event: str) -> list[dict]:
    """Get all active webhooks subscribed to an event"""
    try:
        result = supabase.table("webhooks")\
            .select("*")\
            .eq("is_active", True)\
            .contains("events", [event])\
            .execute()
        return result.data or []
    except Exception as e:
        print(f"Failed to fetch webhooks: {e}")
        return []


def log_webhook(webhook_id: str, event: str, payload: dict, status: int, success: bool):
    """Log webhook delivery attempt"""
    try:
        supabase.table("webhook_logs").insert({
            "webhook_id": webhook_id,
            "event": event,
            "payload": payload,
            "response_status": status,
            "success": success
        }).execute()

        # Update fail count
        if not success:
            supabase.table("webhooks")\
                .update({"fail_count": supabase.table("webhooks")
                         .select("fail_count")
                         .eq("id", webhook_id)
                         .execute().data[0]["fail_count"] + 1})\
                .eq("id", webhook_id)\
                .execute()
        else:
            supabase.table("webhooks")\
                .update({
                    "last_triggered": datetime.utcnow().isoformat(),
                    "fail_count": 0
                })\
                .eq("id", webhook_id)\
                .execute()
    except Exception as e:
        print(f"Failed to log webhook: {e}")


def send_webhook(webhook: dict, event: str, payload: dict) -> bool:
    """Send webhook with retry logic"""
    payload_str = json.dumps({
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "data": payload
    })

    headers = {
        "Content-Type": "application/json",
        "X-DocIntel-Event": event,
        "X-DocIntel-Timestamp": datetime.utcnow().isoformat()
    }

    # Sign payload if secret configured
    if webhook.get("secret"):
        signature = sign_payload(payload_str, webhook["secret"])
        headers["X-DocIntel-Signature"] = f"sha256={signature}"

    # Retry up to 3 times
    for attempt in range(3):
        try:
            response = httpx.post(
                webhook["url"],
                content=payload_str,
                headers=headers,
                timeout=10
            )
            success = response.status_code < 400
            log_webhook(webhook["id"], event, payload, response.status_code, success)
            if success:
                return True
            print(f"Webhook failed with status {response.status_code}, attempt {attempt+1}/3")
        except Exception as e:
            print(f"Webhook delivery error: {e}, attempt {attempt+1}/3")
            log_webhook(webhook["id"], event, payload, 0, False)

    return False


def trigger_webhooks(event: str, payload: dict):
    """Trigger all webhooks for an event — call this after extraction"""
    webhooks = get_active_webhooks(event)
    for webhook in webhooks:
        send_webhook(webhook, event, payload)