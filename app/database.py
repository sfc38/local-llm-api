import json
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
                ttft_ms          INTEGER,
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                messages   TEXT NOT NULL
            )
        """)
        # Safe migration: add new columns to existing databases
        for col, typ in [
            ("response_preview", "TEXT"),
            ("ttft_ms", "INTEGER"),
            ("response_time_ms", "INTEGER"),
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
    ttft_ms: int,
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
               (endpoint, model, prompt_preview, ttft_ms, status_code,
                temperature, max_tokens, top_p, top_k, repeat_penalty, seed, num_ctx)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (endpoint, model, prompt_preview, ttft_ms, status_code,
             temperature, max_tokens, top_p, top_k, repeat_penalty, seed, num_ctx),
        )
        return cur.lastrowid


def update_response(log_id: int, response_preview: str,
                    response_time_ms: int | None = None, ttft_ms: int | None = None):
    cols = ["response_preview = ?"]
    params: list = [response_preview[:1000]]
    if response_time_ms is not None:
        cols.append("response_time_ms = ?")
        params.append(response_time_ms)
    if ttft_ms is not None:
        cols.append("ttft_ms = ?")
        params.append(ttft_ms)
    params.append(log_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"UPDATE requests SET {', '.join(cols)} WHERE id = ?", params)


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


# ── Conversations ─────────────────────────────────────────────────────────────

def save_conversation(name: str, messages: list) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO conversations (name, messages) VALUES (?, ?)",
            (name.strip(), json.dumps(messages)),
        )
        return cur.lastrowid


def list_conversations() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def load_conversation(conv_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    if row:
        d = dict(row)
        d["messages"] = json.loads(d["messages"])
        return d
    return None


def update_conversation(conv_id: int, name: str, messages: list):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE conversations SET name = ?, messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (name.strip(), json.dumps(messages), conv_id),
        )


def delete_conversation(conv_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
