"""
features/doc_comparison/auth_client.py

Thin wrapper around POST /auth/login and POST /auth/refresh on your own
engine — never talks to Supabase directly, matching routers/auth.py.

Session shape stored in st.session_state["auth"]:
    {
        "access_token": str,
        "refresh_token": str,
        "user": {"id": str, "email": str},
    }
"""

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class AuthError(Exception):
    """Raised on login/refresh failure. .detail holds the backend's message."""
    def __init__(self, detail: str, status_code: int = None):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def _extract_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def login(email: str, password: str) -> dict:
    """
    Calls POST /auth/login. Returns the session dict on success.
    Raises AuthError on failure (matches routers/auth.py: 401 on bad
    creds, 403 if email verification is required and missing).
    """
    resp = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    if resp.status_code != 200:
        raise AuthError(_extract_detail(resp), resp.status_code)

    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "user": data["user"],
    }


def refresh(refresh_token: str) -> dict:
    """
    Calls POST /auth/refresh. Returns new {access_token, refresh_token}.
    Note: routers/auth.py's /refresh response does NOT include the user
    object, so callers should keep the existing user dict and only swap
    the tokens.
    """
    resp = requests.post(
        f"{BACKEND_URL}/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=30,
    )
    if resp.status_code != 200:
        raise AuthError(_extract_detail(resp), resp.status_code)

    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
    }


def logout():
    """Clears local session. Best-effort call to /auth/logout — if the
    access token is already dead, we don't want logout itself to fail."""
    session = st.session_state.get("auth")
    if session:
        try:
            requests.post(
                f"{BACKEND_URL}/auth/logout",
                headers=auth_headers(),
                timeout=10,
            )
        except Exception:
            pass  # logging out locally still succeeds even if this fails
    st.session_state.pop("auth", None)


def auth_headers() -> dict:
    """Authorization header for any authenticated call to the engine."""
    session = st.session_state.get("auth")
    if not session:
        return {}
    return {"Authorization": f"Bearer {session['access_token']}"}


def is_logged_in() -> bool:
    return "auth" in st.session_state and st.session_state["auth"] is not None


def ensure_fresh_session():
    """
    Call this before any authenticated API request. If the backend says
    the token is invalid/expired (401), attempt exactly one refresh, then
    give up and force re-login rather than looping silently.
    """
    if not is_logged_in():
        return

    session = st.session_state["auth"]
    try:
        new_tokens = refresh(session["refresh_token"])
        session["access_token"] = new_tokens["access_token"]
        session["refresh_token"] = new_tokens["refresh_token"]
        st.session_state["auth"] = session
    except AuthError:
        # Refresh token itself is dead — user must log in again.
        st.session_state.pop("auth", None)
        st.session_state["auth_expired_notice"] = True


def call_with_auto_refresh(request_fn):
    """
    Wraps a single API call. request_fn is a zero-arg callable that makes
    the request and returns the requests.Response. If it comes back 401,
    refresh once and retry once — never loop indefinitely.

    Usage:
        response = call_with_auto_refresh(
            lambda: requests.post(url, headers=auth_headers(), ...)
        )
    """
    response = request_fn()
    if response.status_code == 401 and is_logged_in():
        try:
            session = st.session_state["auth"]
            new_tokens = refresh(session["refresh_token"])
            session["access_token"] = new_tokens["access_token"]
            session["refresh_token"] = new_tokens["refresh_token"]
            st.session_state["auth"] = session
            response = request_fn()  # retry exactly once with the new token
        except AuthError:
            st.session_state.pop("auth", None)
            st.session_state["auth_expired_notice"] = True
    return response