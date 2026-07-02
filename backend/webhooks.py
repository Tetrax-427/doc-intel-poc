"""
webhooks.py
Webhook delivery for DocIntel.

Changes in this phase:
  - SSRF protection: validate_webhook_url() blocks internal/private IPs
    called before any HTTP request to a webhook URL
"""

import os
import hmac
import json
import socket
import hashlib
import ipaddress
import httpx
from datetime import datetime
from dotenv import load_dotenv
from db import supabase

load_dotenv()


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

# Private/reserved IP ranges that webhook URLs must not resolve to
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("169.254.0.0/16"),    # link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("0.0.0.0/8"),         # unspecified
    ipaddress.ip_network("100.64.0.0/10"),     # shared address space (CGN)
    ipaddress.ip_network("198.18.0.0/15"),     # benchmark testing
    ipaddress.ip_network("240.0.0.0/4"),       # reserved
]

_BLOCKED_SCHEMES = {"file", "gopher", "dict", "ftp", "ldap", "ldaps"}


def validate_webhook_url(url: str) -> None:
    """
    Validate a webhook URL against SSRF attack vectors.

    Checks:
      1. Scheme must be http or https (blocks file://, gopher://, etc.)
      2. URL must not resolve to a private/internal IP address
      3. URL must not use localhost or common internal hostnames

    Raises ValueError with a safe message if the URL is blocked.
    Does NOT include the resolved IP in the error (prevents info leakage).

    Called before every outbound webhook request.
    """
    import urllib.parse

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise ValueError("Invalid webhook URL.")

    # Check scheme
    scheme = parsed.scheme.lower()
    if scheme in _BLOCKED_SCHEMES:
        raise ValueError(f"Webhook URL scheme '{scheme}' is not allowed.")
    if scheme not in ("http", "https"):
        raise ValueError("Webhook URL must use http or https.")

    # Check hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Webhook URL must have a valid hostname.")

    # Block common internal hostnames
    blocked_hostnames = {
        "localhost", "localhost.localdomain",
        "metadata", "metadata.google.internal",
        "169.254.169.254",  # AWS/GCP metadata
        "100.100.100.200",  # Alibaba Cloud metadata
    }
    if hostname.lower() in blocked_hostnames:
        raise ValueError("Webhook URL hostname is not allowed.")

    # Resolve hostname and check against blocked ranges
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Webhook URL hostname could not be resolved.")

    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise ValueError(
                    "Webhook URL resolves to a private or reserved IP address. "
                    "Only public URLs are allowed."
                )


# ---------------------------------------------------------------------------
# Existing webhook logic (unchanged except SSRF check added to send_webhook)
# ---------------------------------------------------------------------------

def sign_payload(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()


def get_active_webhooks(event: str) -> list[dict]:
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
    try:
        supabase.table("webhook_logs").insert({
            "webhook_id":      webhook_id,
            "event":           event,
            "payload":         payload,
            "response_status": status,
            "success":         success,
        }).execute()

        if not success:
            current = supabase.table("webhooks")\
                .select("fail_count").eq("id", webhook_id).execute()
            fail_count = current.data[0]["fail_count"] if current.data else 0
            supabase.table("webhooks")\
                .update({"fail_count": fail_count + 1})\
                .eq("id", webhook_id)\
                .execute()
        else:
            supabase.table("webhooks")\
                .update({
                    "last_triggered": datetime.utcnow().isoformat(),
                    "fail_count":     0,
                })\
                .eq("id", webhook_id)\
                .execute()
    except Exception as e:
        print(f"Failed to log webhook: {e}")


def send_webhook(webhook: dict, event: str, payload: dict) -> bool:
    """Send webhook with SSRF validation and retry logic."""

    # SSRF check before any outbound request
    try:
        validate_webhook_url(webhook["url"])
    except ValueError as e:
        print(f"Webhook SSRF validation failed for {webhook.get('id', '?')}: {e}")
        log_webhook(webhook["id"], event, payload, 0, False)
        return False

    payload_str = json.dumps({
        "event":     event,
        "timestamp": datetime.utcnow().isoformat(),
        "data":      payload,
    })

    headers = {
        "Content-Type":         "application/json",
        "X-DocIntel-Event":     event,
        "X-DocIntel-Timestamp": datetime.utcnow().isoformat(),
    }

    if webhook.get("secret"):
        signature = sign_payload(payload_str, webhook["secret"])
        headers["X-DocIntel-Signature"] = f"sha256={signature}"

    for attempt in range(3):
        try:
            response = httpx.post(
                webhook["url"],
                content=payload_str,
                headers=headers,
                timeout=10,
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
    """Trigger all webhooks for an event."""
    webhooks = get_active_webhooks(event)
    for webhook in webhooks:
        send_webhook(webhook, event, payload)