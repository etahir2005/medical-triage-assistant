"""Tests for the FastAPI HTTP API. External calls are mocked — no
real Gemini/Pinecone credentials or network access required.

API_ACCESS_KEY is explicitly controlled per-test via monkeypatch
rather than relying on whatever happens to be set in the developer's
local .env, so these tests behave identically locally and in CI.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api
from api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_api_key(monkeypatch):
    """Run tests with API key auth disabled by default."""
    monkeypatch.setattr(api, "API_ACCESS_KEY", None)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.get_retriever")
@patch("api.process_user_message")
def test_chat_returns_answer(mock_process, mock_get_retriever):
    mock_get_retriever.return_value = MagicMock()
    mock_process.return_value = {
        "is_emergency": False,
        "answer": "Rest and stay hydrated.",
        "grade": "faithful",
        "sources": [],
    }

    response = client.post("/chat", json={"message": "mild headache"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_emergency"] is False
    assert body["answer"] == "Rest and stay hydrated."
    assert body["grade"] == "faithful"


@patch("api.get_retriever")
@patch("api.process_user_message")
def test_chat_returns_emergency_alert(mock_process, mock_get_retriever):
    mock_get_retriever.return_value = MagicMock()
    mock_process.return_value = {
        "is_emergency": True,
        "answer": "Call 1122 immediately.",
    }

    response = client.post("/chat", json={"message": "chest pain"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_emergency"] is True
    assert body["grade"] is None


def test_chat_rejects_empty_message():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_rejects_overlong_message():
    response = client.post("/chat", json={"message": "a" * 2000})
    assert response.status_code == 422


@patch("api.get_retriever")
@patch("api.process_user_message")
def test_chat_requires_api_key_when_configured(
    mock_process, mock_get_retriever, monkeypatch
):
    """When API_ACCESS_KEY is set, requests must include a matching header."""
    monkeypatch.setattr(api, "API_ACCESS_KEY", "test-secret-key")
    mock_get_retriever.return_value = MagicMock()
    mock_process.return_value = {
        "is_emergency": False,
        "answer": "Rest and stay hydrated.",
        "grade": "faithful",
        "sources": [],
    }

    no_header_response = client.post("/chat", json={"message": "mild headache"})
    assert no_header_response.status_code == 401

    wrong_key_response = client.post(
        "/chat",
        json={"message": "mild headache"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert wrong_key_response.status_code == 401

    correct_key_response = client.post(
        "/chat",
        json={"message": "mild headache"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert correct_key_response.status_code == 200
