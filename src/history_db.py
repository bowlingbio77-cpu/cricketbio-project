"""
History database for saved bowling analyses.

Persists every saved analysis (feature vector + ML predictions + coaching
notes + timings) into a local SQLite database so results can be recalled,
compared over time, and used to track performance across sessions.

Uses only the Python standard library (sqlite3) -- no extra dependencies.
"""
import json
import os
import sqlite3
import time

from . import config

DB_PATH = os.path.join(config.DATA_DIR, "bowling_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT NOT NULL,
    label                TEXT,
    input_mode           TEXT,
    bowling_arm          TEXT,
    model                TEXT,
    performance_score    REAL,
    risk_level           TEXT,
    feature_vector       TEXT NOT NULL,
    injury_risk          TEXT,
    shap_performance     TEXT,
    shap_injury          TEXT,
    coaching_notes       TEXT,
    stage_times          TEXT
);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the database/table if it doesn't exist yet. Safe to call often."""
    conn = _connect()
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def save_analysis(result, *, label="", input_mode="", bowling_arm="", model=""):
    """Persist an AnalysisResult (or its dict) into history. Returns the new row id."""
    init_db()
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    feature_vector = data.get("feature_vector") or {}
    injury_risk = data.get("injury_risk") or {}
    performance_score = data.get("performance_score")
    shap_perf = data.get("shap_contributions_performance")
    shap_injury = data.get("shap_contributions_injury")
    coaching_notes = data.get("coaching_notes") or []
    stage_times = data.get("stage_times") or {}

    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO analyses (
                created_at, label, input_mode, bowling_arm, model,
                performance_score, risk_level, feature_vector, injury_risk,
                shap_performance, shap_injury, coaching_notes, stage_times
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                label or "",
                input_mode or "",
                bowling_arm or "",
                model or "",
                performance_score,
                injury_risk.get("risk_level"),
                json.dumps(feature_vector),
                json.dumps(injury_risk) if injury_risk else None,
                json.dumps(shap_perf) if shap_perf else None,
                json.dumps(shap_injury) if shap_injury else None,
                json.dumps(coaching_notes),
                json.dumps(stage_times),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _parse_row(row):
    d = dict(row)
    for key in ("feature_vector", "injury_risk", "shap_performance", "shap_injury",
                "coaching_notes", "stage_times"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def load_all():
    """All saved analyses, newest first. Each row is a dict with JSON fields parsed."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
        return [_parse_row(r) for r in rows]
    finally:
        conn.close()


def load_by_ids(ids):
    """Load specific analyses by id, in ascending id order."""
    if not ids:
        return []
    init_db()
    conn = _connect()
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM analyses WHERE id IN ({placeholders}) ORDER BY id ASC",
            list(ids)).fetchall()
        return [_parse_row(r) for r in rows]
    finally:
        conn.close()


def delete_analysis(ids):
    """Delete analyses by id. No-op for an empty list."""
    if not ids:
        return
    init_db()
    conn = _connect()
    try:
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM analyses WHERE id IN ({placeholders})", list(ids))
        conn.commit()
    finally:
        conn.close()


def clear_all():
    """Delete every saved analysis."""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM analyses")
        conn.commit()
    finally:
        conn.close()


def count():
    """Number of saved analyses."""
    init_db()
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    finally:
        conn.close()
