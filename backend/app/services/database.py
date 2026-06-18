"""
Database Service - SQLite layer for TUK-ConvoSearch
Location: backend/app/services/database.py

Adds persistent storage for:
  - documents   : metadata about each file ingested into FAISS
  - query_log   : every question asked and the answer/sources returned
  - users       : admin accounts for the React admin panel (Phase 2)

This is purely additive. FAISS remains the vector search engine and
metadata.pkl remains the source of truth for chunk-level retrieval data.
SQLite here is the system of record for *document-level* metadata,
*query history*, and *admin accounts* — the three things that currently
have nowhere durable to live.
"""

import sqlite3
import os
import time
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

# Database file lives in backend/data/tuk_convosearch.db
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "tuk_convosearch.db")


def _ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)


@contextmanager
def get_connection():
    """
    Context manager yielding a sqlite3 connection with foreign keys enabled
    and row factory set so rows behave like dicts.
    """
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Create all tables if they do not already exist. Safe to call on every
    application startup (in main.py) — CREATE TABLE IF NOT EXISTS is a no-op
    if the schema is already there.
    """
    _ensure_db_dir()
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name     TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'admin',
                created_at    REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT    NOT NULL,
                filename      TEXT    NOT NULL UNIQUE,
                file_type     TEXT    NOT NULL,
                upload_date   REAL    NOT NULL,
                chunk_count   INTEGER NOT NULL DEFAULT 0,
                is_active     INTEGER NOT NULL DEFAULT 1,
                uploaded_by   INTEGER,
                FOREIGN KEY (uploaded_by) REFERENCES users (user_id)
            );

            CREATE TABLE IF NOT EXISTS query_log (
                query_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                question        TEXT    NOT NULL,
                answer          TEXT    NOT NULL,
                sources_json    TEXT    NOT NULL,
                chunks_found    INTEGER NOT NULL DEFAULT 0,
                response_time   REAL    NOT NULL DEFAULT 0,
                vector_db       TEXT    NOT NULL DEFAULT 'FAISS',
                timestamp       REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_querylog_timestamp
                ON query_log (timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_documents_active
                ON documents (is_active);
        """)
    print(f"✓ Database initialised at {DB_PATH}")


# --------------------------------------------------------------------- #
# Document helpers
# --------------------------------------------------------------------- #

def upsert_document(filename: str, title: str, file_type: str, chunk_count: int,
                     uploaded_by: Optional[int] = None) -> int:
    """
    Insert a new document record, or update chunk_count/upload_date if a
    document with this filename already exists (e.g. rebuild_faiss.py was
    re-run after a file was replaced). Returns the document_id.
    """
    now = time.time()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT document_id FROM documents WHERE filename = ?", (filename,)
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE documents
                   SET title = ?, file_type = ?, chunk_count = ?, upload_date = ?, is_active = 1
                   WHERE filename = ?""",
                (title, file_type, chunk_count, now, filename)
            )
            return existing["document_id"]
        else:
            cur = conn.execute(
                """INSERT INTO documents (title, filename, file_type, upload_date, chunk_count, uploaded_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (title, filename, file_type, now, chunk_count, uploaded_by)
            )
            return cur.lastrowid


def list_documents(active_only: bool = True) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        query = "SELECT * FROM documents"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY upload_date DESC"
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]


def deactivate_document(document_id: int) -> bool:
    """Soft-delete: mark inactive rather than removing the row, preserving history."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE documents SET is_active = 0 WHERE document_id = ?", (document_id,)
        )
        return cur.rowcount > 0


def clear_all_documents():
    """Used by rebuild_faiss.py before a full re-index, to keep documents table
    in sync with a freshly rebuilt FAISS index."""
    with get_connection() as conn:
        conn.execute("DELETE FROM documents")


# --------------------------------------------------------------------- #
# Query log helpers
# --------------------------------------------------------------------- #

def log_query(session_id: str, question: str, answer: str, sources: list,
              chunks_found: int, response_time: float, vector_db: str = "FAISS"):
    import json
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO query_log
               (session_id, question, answer, sources_json, chunks_found, response_time, vector_db, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, question, answer, json.dumps(sources), chunks_found,
             response_time, vector_db, time.time())
        )


def get_recent_queries(limit: int = 50) -> List[Dict[str, Any]]:
    import json
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for r in rows:
            row_dict = dict(r)
            row_dict["sources"] = json.loads(row_dict.pop("sources_json"))
            results.append(row_dict)
        return results


def get_query_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM query_log").fetchone()["c"]
        avg_time = conn.execute("SELECT AVG(response_time) AS a FROM query_log").fetchone()["a"]
        unanswered = conn.execute(
            "SELECT COUNT(*) AS c FROM query_log WHERE chunks_found = 0"
        ).fetchone()["c"]
        return {
            "total_queries": total,
            "average_response_time": round(avg_time, 2) if avg_time else 0.0,
            "unanswered_queries": unanswered
        }


# --------------------------------------------------------------------- #
# User helpers (used by Phase 2 auth)
# --------------------------------------------------------------------- #

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


def create_user(full_name: str, email: str, password_hash: str, role: str = "admin") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO users (full_name, email, password_hash, role, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (full_name, email, password_hash, role, time.time())
        )
        return cur.lastrowid


def user_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]