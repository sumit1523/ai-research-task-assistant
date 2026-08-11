"""Local persistence. Keeping it separate makes the AI workflow easier to read."""

import sqlite3
import re
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                task_text TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )


def extract_task_items(tasks: str) -> list[str]:
    """Keep numbered task-plan lines; fall back to non-empty lines for flexible model output."""
    numbered = re.findall(r"^\s*\d+[.)]\s+(.+)$", tasks, flags=re.MULTILINE)
    return numbered or [line.strip("- ") for line in tasks.splitlines() if line.strip()]


def save_session(question: str, summary: str, tasks: str) -> int:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO sessions (question, summary, tasks) VALUES (?, ?, ?)",
            (question, summary, tasks),
        )
        session_id = cursor.lastrowid
        connection.executemany(
            "INSERT INTO task_items (session_id, task_text) VALUES (?, ?)",
            [(session_id, task) for task in extract_task_items(tasks)],
        )
        return session_id


def task_items(session_id: int) -> list[tuple[int, str, bool]]:
    with sqlite3.connect(DB_PATH) as connection:
        return [
            (task_id, task_text, bool(completed))
            for task_id, task_text, completed in connection.execute(
                "SELECT id, task_text, completed FROM task_items WHERE session_id = ? ORDER BY id", (session_id,)
            )
        ]


def set_task_completed(task_id: int, completed: bool) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE task_items SET completed = ? WHERE id = ?", (int(completed), task_id))


def recent_sessions(limit: int = 5) -> list[tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(
            "SELECT question, created_at FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
