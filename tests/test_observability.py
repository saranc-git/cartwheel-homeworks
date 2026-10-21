"""Homework 2, Part D: authentication tests for the session endpoints.

These tests call the endpoint functions directly. They need no Langfuse,
Docker, or model provider key.
"""

import pytest
from fastapi import HTTPException

from server import app as server_app


@pytest.fixture(autouse=True)
def clear_sessions():
    server_app._SESSIONS.clear()
    yield
    server_app._SESSIONS.clear()


def test_create_session_rejects_claimed_role_mismatch():
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc.value.status_code == 403
    assert server_app._SESSIONS == {}


def test_token_for_one_session_cannot_authorize_another():
    shopper = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    merchant = server_app.create_session(server_app.SessionCreate(user_id=9002, role="merchant"))

    own = server_app._authorize(shopper["session_id"], f"Bearer {shopper['token']}")
    assert own.user_id == 1

    with pytest.raises(HTTPException) as exc:
        server_app._authorize(merchant["session_id"], f"Bearer {shopper['token']}")
    assert exc.value.status_code == 403
