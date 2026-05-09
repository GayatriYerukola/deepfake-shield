"""
SQLite Database Layer
=====================
Stores every scan result persistently so the history survives server restarts.
Uses SQLAlchemy Core (no ORM) for simplicity.

Schema
------
scans
  id           TEXT PRIMARY KEY   (e.g. "SCAN-0001")
  filename     TEXT
  media_type   TEXT               "image" | "video"
  verdict      TEXT
  ai_probability REAL
  confidence   REAL
  model_used   INTEGER            0 | 1
  model_id     TEXT
  report_json  TEXT               full JSON report
  created_at   TEXT               ISO timestamp
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("reports") / "scans.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the scans table if it doesn't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id             TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                media_type     TEXT NOT NULL,
                verdict        TEXT,
                ai_probability REAL,
                confidence     REAL,
                model_used     INTEGER DEFAULT 0,
                model_id       TEXT,
                report_json    TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
        """)
        conn.commit()


def insert_scan(scan_id: str, report: dict) -> None:
    """Persist a completed scan report."""
    td = report.get("technical_details", {})
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scans
              (id, filename, media_type, verdict, ai_probability,
               confidence, model_used, model_id, report_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                scan_id,
                report.get("filename", ""),
                report.get("media_type", ""),
                report.get("verdict"),
                report.get("ai_probability"),
                report.get("confidence_level"),
                int(td.get("model_used", False)),
                td.get("model_id"),
                json.dumps(report),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def get_all_scans(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return a list of scan summaries (no full report JSON)."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, filename, media_type, verdict, ai_probability,
                      confidence, model_used, model_id, created_at
               FROM scans
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_scan(scan_id: str) -> Optional[dict]:
    """Return the full report for one scan, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT report_json FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["report_json"])


def count_scans() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]


def next_scan_id() -> str:
    n = count_scans() + 1
    return f"SCAN-{n:05d}"
