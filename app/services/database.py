"""
SQLite layer: users, documents, chunks, sessions, messages, query log.

Two things changed relative to the old backend/ version. Conversation state
used to live in a module-level dict, which meant a restart (or a second
uvicorn worker) wiped every student's context; it now lives in `sessions`
and `messages`. Chunk vectors used to live in a pickle beside the FAISS
file; they now live as blobs in `chunks`, which makes the database the only
artifact that needs backing up and lets a deleted document be removed
without re-embedding the corpus.

Location: app/services/database.py
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'admin',
    created_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    filename     TEXT    NOT NULL UNIQUE,
    file_type    TEXT    NOT NULL,
    sha256       TEXT    NOT NULL,
    upload_date  REAL    NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    is_active    INTEGER NOT NULL DEFAULT 1,
    indexed_at   REAL,
    uploaded_by  INTEGER REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    source      TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    title          TEXT NOT NULL DEFAULT 'New chat',
    created_at     REAL NOT NULL,
    last_active_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    created_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    query_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    answer        TEXT    NOT NULL,
    sources_json  TEXT    NOT NULL,
    route         TEXT,
    grounded      INTEGER,
    retry_count   INTEGER DEFAULT 0,
    chunks_found  INTEGER NOT NULL DEFAULT 0,
    response_time REAL    NOT NULL DEFAULT 0,
    model         TEXT,
    timestamp     REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_querylog_timestamp ON query_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_documents_active   ON documents (is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_document    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages (session_id, message_id);
"""


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------- #

def create_user(full_name: str, email: str, password_hash: str, role: str = "admin") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (full_name, email, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (full_name, email.lower().strip(), password_hash, role, time.time()),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def user_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]


# --------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------- #

def upsert_document(
    filename: str,
    title: str,
    file_type: str,
    sha256: str,
    chunk_count: int,
    uploaded_by: Optional[int] = None,
) -> int:
    now = time.time()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT document_id FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE documents SET title = ?, file_type = ?, sha256 = ?, "
                "chunk_count = ?, upload_date = ?, is_active = 1, indexed_at = ? "
                "WHERE filename = ?",
                (title, file_type, sha256, chunk_count, now, now, filename),
            )
            return existing["document_id"]
        cur = conn.execute(
            "INSERT INTO documents (title, filename, file_type, sha256, upload_date, "
            "chunk_count, uploaded_by, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, filename, file_type, sha256, now, chunk_count, uploaded_by, now),
        )
        return cur.lastrowid


def find_document_by_hash(sha256: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? AND is_active = 1", (sha256,)
        ).fetchone()
        return dict(row) if row else None


def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None


def get_document_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None


def mark_document_indexed(document_id: int, chunk_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET chunk_count = ?, indexed_at = ?, is_active = 1 "
            "WHERE document_id = ?",
            (chunk_count, time.time(), document_id),
        )


def list_documents(active_only: bool = True) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        query = "SELECT * FROM documents"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY upload_date DESC"
        return [dict(r) for r in conn.execute(query).fetchall()]


def deactivate_document(document_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE documents SET is_active = 0 WHERE document_id = ?", (document_id,)
        )
        return cur.rowcount > 0


# --------------------------------------------------------------------- #
# Chunks (vectors live here)
# --------------------------------------------------------------------- #

def insert_chunks(document_id: int, rows: List[Dict[str, Any]]) -> None:
    """rows: [{chunk_index, source, text, embedding(bytes)}]"""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO chunks (document_id, chunk_index, source, text, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (document_id, r["chunk_index"], r["source"], r["text"], r["embedding"])
                for r in rows
            ],
        )


def delete_chunks(document_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        return cur.rowcount


def iter_chunks() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT c.chunk_id, c.document_id, c.source, c.text, c.embedding "
            "FROM chunks c JOIN documents d ON d.document_id = c.document_id "
            "WHERE d.is_active = 1 ORDER BY c.chunk_id"
        ).fetchall()
        return [dict(r) for r in rows]


def chunk_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]


def document_chunk_count(document_id: int) -> int:
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()["c"]


def clear_all_chunks() -> int:
    """Used only by scripts/ingest_corpus.py --force."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM chunks")
        return cur.rowcount


# --------------------------------------------------------------------- #
# Sessions and messages
# --------------------------------------------------------------------- #

def touch_session(session_id: str, first_question: str = "") -> None:
    now = time.time()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT session_id, title FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            title = (first_question[:60] + "...") if len(first_question) > 60 else (
                first_question or "New chat"
            )
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, last_active_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        else:
            conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (now, session_id),
            )


def add_message(session_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def recent_messages(session_id: str, limit: int) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY message_id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT s.session_id, s.title, s.created_at, s.last_active_at, "
            "COUNT(m.message_id) AS message_count "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.session_id "
            "GROUP BY s.session_id ORDER BY s.last_active_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? "
            "ORDER BY message_id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0


# --------------------------------------------------------------------- #
# Query log
# --------------------------------------------------------------------- #

def log_query(
    session_id: str,
    question: str,
    answer: str,
    sources: list,
    chunks_found: int,
    response_time: float,
    route: str = "",
    grounded: bool = False,
    retry_count: int = 0,
    model: str = "",
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO query_log (session_id, question, answer, sources_json, route, "
            "grounded, retry_count, chunks_found, response_time, model, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                question,
                answer,
                json.dumps(sources),
                route,
                int(grounded),
                retry_count,
                chunks_found,
                response_time,
                model,
                time.time(),
            ),
        )


def get_recent_queries(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json"))
            results.append(item)
        return results


def get_query_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM query_log").fetchone()["c"]
        avg_time = conn.execute(
            "SELECT AVG(response_time) AS a FROM query_log"
        ).fetchone()["a"]
        unanswered = conn.execute(
            "SELECT COUNT(*) AS c FROM query_log WHERE chunks_found = 0"
        ).fetchone()["c"]
        ungrounded = conn.execute(
            "SELECT COUNT(*) AS c FROM query_log WHERE grounded = 0 AND route = 'internal_docs'"
        ).fetchone()["c"]
        top_questions = conn.execute(
            "SELECT question, COUNT(*) AS hits FROM query_log "
            "GROUP BY question ORDER BY hits DESC LIMIT 10"
        ).fetchall()
        return {
            "total_queries": total,
            "average_response_time": round(avg_time, 2) if avg_time else 0.0,
            "unanswered_queries": unanswered,
            "ungrounded_answers": ungrounded,
            "top_questions": [dict(r) for r in top_questions],
        }
