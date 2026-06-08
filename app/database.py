import sqlite3
from pathlib import Path

DB_PATH = Path("logs/requests.db")


def init_db():
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint         TEXT NOT NULL,
                model            TEXT,
                prompt_preview   TEXT,
                response_preview TEXT,
                response_time_ms INTEGER,
                status_code      INTEGER,
                temperature      REAL,
                max_tokens       INTEGER,
                top_p            REAL,
                top_k            INTEGER,
                repeat_penalty   REAL,
                seed             INTEGER,
                num_ctx          INTEGER
            )
        """)
        # Safe migration: add new columns to existing databases
        for col, typ in [
            ("response_preview", "TEXT"),
            ("temperature", "REAL"),
            ("max_tokens", "INTEGER"),
            ("top_p", "REAL"),
            ("top_k", "INTEGER"),
            ("repeat_penalty", "REAL"),
            ("seed", "INTEGER"),
            ("num_ctx", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass


def log_request(
    endpoint: str,
    model: str,
    prompt_preview: str,
    response_time_ms: int,
    status_code: int,
    temperature=None,
    max_tokens=None,
    top_p=None,
    top_k=None,
    repeat_penalty=None,
    seed=None,
    num_ctx=None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO requests
               (endpoint, model, prompt_preview, response_time_ms, status_code,
                temperature, max_tokens, top_p, top_k, repeat_penalty, seed, num_ctx)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (endpoint, model, prompt_preview, response_time_ms, status_code,
             temperature, max_tokens, top_p, top_k, repeat_penalty, seed, num_ctx),
        )
        return cur.lastrowid


def update_response(log_id: int, response_preview: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE requests SET response_preview = ? WHERE id = ?",
            (response_preview[:1000], log_id),
        )


def clear_requests():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM requests")


def get_requests(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
