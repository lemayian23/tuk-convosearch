"""
Tests for the chat HTTP layer. The pipeline itself is faked in conftest, so
these assert on the contract the student UI depends on.

Location: tests/test_api_chat.py
"""

import json

import pytest


def test_chat_returns_answer_and_sources(client):
    response = client.post("/api/chat", json={"message": "When do exams start?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].startswith("Second semester exams")
    assert body["sources"][0]["source"] == "2026 Academic Year Calendar.docx"
    assert len(body["session_id"]) == 32
    assert body["route"] == "internal_docs"
    assert body["grounded"] is True


def test_chat_reuses_session_and_stores_history(client):
    first = client.post("/api/chat", json={"message": "When do exams start?"})
    session_id = first.json()["session_id"]

    second = client.post(
        "/api/chat", json={"message": "And the postgraduate ones?", "session_id": session_id}
    )
    assert second.json()["session_id"] == session_id

    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_rejects_blank_message(client):
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_rejects_guessed_session_id_shape(client):
    response = client.post("/api/chat", json={"message": "hi", "session_id": "x"})
    assert response.status_code == 422


def test_rejects_unknown_pipeline(client):
    response = client.post("/api/chat", json={"message": "hi", "pipeline": "legal"})
    assert response.status_code == 422


def test_stream_emits_ordered_events(client):
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "When do exams start?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(chunk for chunk in response.iter_text())

    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in payload.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "start"
    assert "sources" in types and "token" in types
    assert types.index("sources") < types.index("token"), "sources must precede the answer"
    assert types[-1] == "end"

    streamed = "".join(e["content"] for e in events if e["type"] == "token")
    assert "exams start" in streamed


def test_no_route_lists_every_session(client):
    """
    Regression. Students have no login, so a route returning every session
    published other students' ids - and with them every transcript, since
    /api/sessions/{id}/messages trusts the id. The list is admin-only now and
    the browser keeps its own history (web/src/history.js).
    """
    assert client.get("/api/sessions").status_code == 404
    assert client.get("/api/admin/sessions").status_code == 401


def test_operator_can_list_sessions(client, auth_headers):
    session_id = client.post("/api/chat", json={"message": "fees?"}).json()["session_id"]

    listed = client.get("/api/admin/sessions", headers=auth_headers)
    assert listed.status_code == 200
    assert any(s["session_id"] == session_id for s in listed.json())
    assert {"session_id", "title", "message_count"} <= set(listed.json()[0])


def test_the_browser_holding_an_id_can_delete_that_session(client):
    session_id = client.post("/api/chat", json={"message": "fees?"}).json()["session_id"]

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}/messages").json() == []


def test_client_config_advertises_quick_prompts(client):
    body = client.get("/api/config").json()
    assert body["streaming"] is True
    assert "tuk" in body["pipelines"]
    assert len(body["quick_prompts"]) >= 4


def test_rate_limit_returns_retry_after(client, monkeypatch):
    from app.api import deps
    from app.services.rate_limit import RateLimiter

    monkeypatch.setattr(deps, "get_limiter", lambda: RateLimiter(per_minute=0, burst=0))

    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_busy_model_slots_degrade_cleanly(client, monkeypatch):
    from app.api import chat as chat_api
    from app.services.chat import ChatBusyError

    class RejectingService:
        async def answer(self, *args, **kwargs):
            raise ChatBusyError("All 2 model slots are busy.")

    monkeypatch.setattr(chat_api, "get_chat_service", lambda: RejectingService())

    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "busy" in response.json()["detail"].lower()


def test_pipeline_failure_is_not_leaked_as_200(client, monkeypatch):
    from app.api import chat as chat_api
    from app.services.chat import ChatResult

    class FailingService:
        async def answer(self, *args, **kwargs):
            return ChatResult(
                session_id="deadbeefdeadbeefdeadbeefdeadbeef",
                answer="",
                error="connection refused",
            )

    monkeypatch.setattr(chat_api, "get_chat_service", lambda: FailingService())

    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "connection refused" not in response.text  # detail is the error, not a traceback


def test_student_bundle_arrives_compressed(client):
    """
    The target user is a student on a metered Data bundle, so the SPA is only
    shippable if it is gzip-compressed on the wire. This asserts on the real
    hashed asset name out of index.html rather than a made-up path.
    """
    import re

    html = client.get("/").text
    match = re.search(r'src="(/assets/[^"]+\.js)"', html)
    if match is None:
        pytest.skip("web/dist not built; nothing to serve")

    bundle = client.get(match.group(1), headers={"Accept-Encoding": "gzip"})
    assert bundle.status_code == 200
    assert bundle.headers.get("content-encoding") == "gzip"


def test_responses_carry_the_security_headers(client):
    response = client.get("/api/config")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"
