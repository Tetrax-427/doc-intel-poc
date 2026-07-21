"""
config_store.py
----------------
Two separate concerns, kept apart on purpose:

1. Connection settings the user can see/edit (base_url, timeout) - resolved
   from, in increasing priority: DEFAULTS < env vars/secrets < data/config.json
   (written when you click "Save settings" in the sidebar).

2. The auth token (JWT) - read ONLY from the DOCINTEL_JWT_TOKEN environment
   variable (or Streamlit secrets with the same key). It is never written to
   data/config.json, never echoed back to the UI, and there's no sidebar
   field for it. Set it in your shell / hosting platform's env vars before
   running the app, e.g.:

       export DOCINTEL_JWT_TOKEN="eyJhbGciOi..."
       streamlit run app.py

   This keeps a long-lived token out of a local plaintext file and off
   screen. Since it's a single shared token for whoever runs this app, there
   is intentionally no per-user auth flow here - swap in a real login later
   if this app grows beyond a personal tool.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Loads variables from a .env file (if present) in the current working
# directory or any parent directory into os.environ. Without this,
# os.environ.get() only sees variables actually exported into the shell -
# a .env file on disk does nothing on its own.
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CONFIG_FILE = DATA_DIR / "config.json"

DEFAULTS = {
    "base_url": "http://localhost:8000",
    "timeout": 180,
}

ENV_KEYS = {
    "base_url": "DOCINTEL_BASE_URL",
    "timeout": "DOCINTEL_TIMEOUT",
}

JWT_ENV_KEY = "DOCINTEL_JWT_TOKEN"


def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        # No secrets.toml present, or running outside a streamlit context -
        # both are fine, just means secrets aren't a source here.
        return ""


def _from_env_and_secrets() -> dict:
    values = {}
    for field, env_key in ENV_KEYS.items():
        raw = os.environ.get(env_key) or _get_secret(env_key)
        if not raw:
            continue
        if field == "timeout":
            try:
                values[field] = int(raw)
            except (TypeError, ValueError):
                continue
        else:
            values[field] = raw
    return values


def load_config() -> dict:
    """Base URL + timeout only. Never contains the auth token."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    merged = dict(DEFAULTS)
    merged.update(_from_env_and_secrets())

    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            merged.update({k: v for k, v in saved.items() if v and k in DEFAULTS})
        except json.JSONDecodeError:
            pass

    return merged


def save_config(cfg: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Only ever persist the known non-secret fields, even if a caller
    # accidentally passes more.
    to_save = {k: cfg.get(k) for k in DEFAULTS}
    CONFIG_FILE.write_text(json.dumps(to_save, indent=2))


def get_auth_token() -> str:
    """The JWT to authenticate with, sourced only from env/secrets."""
    return os.environ.get(JWT_ENV_KEY) or _get_secret(JWT_ENV_KEY) or ""