"""
Tests for auth and the admin surface.

Location: tests/test_auth_admin.py
"""

from app.services import database
from app.services.auth import hash_password, password_policy_error


def test_login_rejects_unknown_email_and_wrong_password(client):
    database.create_user("Real Admin", "real@example.com", hash_password("correct-horse-battery"), "admin")

    wrong_password = client.post(
        "/api/auth/login", json={"email": "real@example.com", "password": "nope"}
    )
    unknown_email = client.post(
        "/api/auth/login", json={"email": "ghost@example.com", "password": "nope"}
    )

    assert wrong_password.status_code == 401
    assert unknown_email.status_code == 401
    # Neither response may reveal which of the two failed.
    assert unknown_email.json()["detail"] == wrong_password.json()["detail"]


def test_admin_routes_require_a_token(client):
    assert client.get("/api/admin/documents").status_code in (401, 403)
    assert client.get("/api/admin/stats").status_code in (401, 403)


def test_token_from_login_unlocks_admin_routes(client, auth_headers):
    response = client.get("/api/admin/documents", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_me_returns_the_jwt_subject(client, auth_headers):
    body = client.get("/api/auth/me", headers=auth_headers).json()
    assert body["email"] == "admin@example.com"
    assert body["role"] == "admin"


def test_token_tampering_is_rejected(client, admin_token):
    forged = admin_token[:-4] + ("aaaa" if not admin_token.endswith("aaaa") else "bbbb")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_password_policy_blocks_weak_admin_passwords():
    assert password_policy_error("short") is not None
    assert password_policy_error("a" * 100) is not None
    assert password_policy_error("long-enough-passphrase") is None


def test_upload_indexes_and_returns_chunk_count(client, auth_headers, settings):
    content = b"Examinations begin on 5 May 2026 in the main campus."
    response = client.post(
        "/api/admin/documents",
        headers=auth_headers,
        files={"file": ("notice.txt", content, "text/plain")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "indexed"
    assert response.json()["chunks"] == 3


def test_upload_rejects_non_document_types(client, auth_headers):
    response = client.post(
        "/api/admin/documents",
        headers=auth_headers,
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_file(client, auth_headers, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    response = client.post(
        "/api/admin/documents",
        headers=auth_headers,
        files={"file": ("big.txt", b"x" * 64, "text/plain")},
    )
    assert response.status_code == 413


def test_uploaded_filename_cannot_escape_corpus_dir(client, auth_headers, settings):
    response = client.post(
        "/api/admin/documents",
        headers=auth_headers,
        files={"file": ("../../../../etc/cron.d/evil.txt", b"text", "text/plain")},
    )
    assert response.status_code in (201, 400, 422)

    escape_target = settings.data_dir.parent / "etc" / "cron.d" / "evil.txt"
    assert not escape_target.exists()
    written = list(settings.corpus_dir.glob("*evil*"))
    assert written == [] or all(p.parent == settings.corpus_dir for p in written)


def test_delete_document_removes_chunks(client, auth_headers):
    database.init_db()
    document_id = database.upsert_document(
        filename="calendar.docx", title="Calendar", file_type=".docx",
        sha256="abc", chunk_count=2,
    )
    assert client.delete(f"/api/admin/documents/{document_id}", headers=auth_headers).status_code == 204
    remaining = [d for d in database.list_documents(active_only=True) if d["document_id"] == document_id]
    assert remaining == []


def test_admin_index_reports_ollama_and_chunks(client, auth_headers):
    body = client.get("/api/admin/index", headers=auth_headers).json()
    assert body["index"]["total_chunks"] == 12
    assert body["ollama"]["model_present"] is True


def test_health_endpoints_differ_in_strictness(client):
    assert client.get("/health").json()["status"] == "alive"
    readiness = client.get("/api/health").json()
    assert set(readiness["checks"]) == {
        "corpus_indexed",
        "index_matches_corpus",
        "ollama_reachable",
        "model_present",
    }
    assert readiness["status"] in ("ready", "degraded")
    # The stub index carries vectors the chunks table does not: that is
    # precisely the out-of-band-ingest divergence readiness has to expose.
    assert readiness["checks"]["index_matches_corpus"] is False
    assert readiness["stored_chunks"] == 0


def test_no_admin_is_created_without_the_bootstrap_pair(settings, monkeypatch):
    """A fresh instance must boot with zero accounts, not a printed default."""
    from app.services import database
    from app.services.auth import ensure_bootstrap_admin

    assert database.user_count() == 0
    monkeypatch.setattr(settings, "bootstrap_admin_email", "")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "")
    ensure_bootstrap_admin()
    assert database.user_count() == 0


def test_bootstrap_pair_creates_exactly_one_admin(settings, monkeypatch):
    from app.services import database
    from app.services.auth import authenticate_user, ensure_bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_email", "registrar@example.test")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "long-enough-passphrase")
    ensure_bootstrap_admin()
    # Twice is what happens across a restart; it must not add a second account.
    ensure_bootstrap_admin()

    assert database.user_count() == 1
    assert authenticate_user("registrar@example.test", "long-enough-passphrase")
    assert authenticate_user("registrar@example.test", "wrong-passphrase-here") is None


def test_weak_bootstrap_password_is_refused(settings, monkeypatch):
    from app.services import database
    from app.services.auth import ensure_bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_email", "registrar@example.test")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "Admin2026!")
    ensure_bootstrap_admin()
    assert database.user_count() == 0
