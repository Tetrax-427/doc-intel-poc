"""tests/test_auth.py"""
import pytest
import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_no_auth_configured_allows_all(monkeypatch):
    """When CLERK_SECRET_KEY is not set, all requests get dev_user."""
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    import importlib
    import core.auth as auth_module
    importlib.reload(auth_module)
    assert auth_module.CLERK_SECRET_KEY == ""


def test_invalid_api_key_returns_401():
    """Invalid API key returns 401."""
    response = client.get(
        "/documents",
        headers={"X-API-Key": "di_invalid_key_that_does_not_exist"}
    )
    assert response.status_code == 401


def test_no_api_key_returns_200_in_dev_mode(monkeypatch):
    """No auth configured — request passes through as dev_user."""
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    response = client.get("/documents")
    assert response.status_code == 200


def test_get_user_id_with_none():
    """get_user_id(None) returns 'anonymous'."""
    from core.auth import get_user_id
    assert get_user_id(None) == "anonymous"


def test_get_user_id_with_user():
    """get_user_id returns user_id from UserContext."""
    from core.auth import get_user_id, UserContext
    user = UserContext(user_id="user_abc123")
    assert get_user_id(user) == "user_abc123"


def test_get_user_id_with_dict():
    """get_user_id returns user_id from dict."""
    from core.auth import get_user_id
    user = {"user_id": "user_abc123"}
    assert get_user_id(user) == "user_abc123"


def test_get_user_id_missing_key():
    """get_user_id falls back to 'anonymous' if user_id key missing."""
    from core.auth import get_user_id
    assert get_user_id({"email": "test@test.com"}) == "anonymous"