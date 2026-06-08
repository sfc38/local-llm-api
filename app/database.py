import sqlite3
from pathlib import Path

DB_PATH = Path("logs/requests.db")


def init_db():
    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint    TEXT NOT NULL,
                model       TEXT,
                prompt_preview TEXT,
                response_time_ms INTEGER,
                status_code INTEGER
            )
        """)


def log_request(endpoint: str, model: str, prompt_preview: str, response_time_ms: int, status_code: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO requests (endpoint, model, prompt_preview, response_time_ms, status_code) "
            "VALUES (?, ?, ?, ?, ?)",
            (endpoint, model, prompt_preview, response_time_ms, status_code),
        )


def get_requests(limit: int = 500) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
