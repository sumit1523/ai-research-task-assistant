"""Local persistence. Keeping it separate makes the AI workflow easier to read."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("research_tasks.db")


def initialise_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                summary TEXT NOT NULL,
                tasks TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_session(question: str, summary: str, tasks: str) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO sessions (question, summary, tasks) VALUES (?, ?, ?)",
            (question, summary, tasks),
        )


def recent_sessions(limit: int = 5) -> list[tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(
            "SELECT question, created_at FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
