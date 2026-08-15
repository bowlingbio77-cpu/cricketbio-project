"""
History database for saved bowling analyses.

Persists every saved analysis (feature vector + ML predictions + coaching
notes + timings) into a local SQLite database so results can be recalled,
compared over time, and used to track performance across sessions.

Idempotency: each save carries a content `fingerprint` (athlete + label + tags
+ input mode + model + arm + feature vector + score). Saving the identical
result again returns the existing row instead of inserting a duplicate, so
accidental double-clicks on "Save" don't create duplicates.

Athlete identity: saves are tagged with the bowler's name so the history page
can separate one athlete's trend from another's.

Uses only the Python standard library (sqlite3) -- no extra dependencies.
"""
import hashlib
import json
import os
import sqlite3
import time

from . import config

DB_PATH = os.environ.get(
    "PACEAI_HISTORY_DB", os.path.join(config.DATA_DIR, "bowling_history.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT NOT NULL,
    label                TEXT,
    input_mode           TEXT,
    bowling_arm          TEXT,
    model                TEXT,
    athlete              TEXT,
    tags                 TEXT,
    fingerprint          TEXT,
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

# Columns added after the original schema -- migrated in place for existing DBs.
_ADDED_COLUMNS = {
    "athlete": "TEXT",
    "tags": "TEXT",
    "fingerprint": "TEXT",
}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the database/table if it doesn't exist yet, and migrate old
    databases that predate the athlete/tags/fingerprint columns. Safe to call
    often."""
    conn = _connect()
    try:
        conn.execute(_SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        for col, typ in _ADDED_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {typ}")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_analyses_fingerprint "
            "ON analyses (fingerprint)")
        conn.commit()
    finally:
        conn.close()


def _fingerprint(*, athlete, label, tags, input_mode, model, bowling_arm,
                 feature_vector, performance_score):
    """Content hash identifying a unique saved result. Re-saving identical
    content (same athlete, tags, mode, model, arm, features and score) yields
    the same fingerprint and is treated as idempotent."""
    canonical = json.dumps([
        athlete or "", label or "", tags or "", input_mode or "", model or "",
        bowling_arm or "", feature_vector, performance_score,
    ], sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def save_analysis(result, *, label="", input_mode="", bowling_arm="", model="",
                  athlete="", tags=""):
    """Persist an AnalysisResult (or its dict) into history.

    Idempotent: if an identical result (same content fingerprint) already
    exists, no new row is inserted and the existing row id is returned.

    Returns `(row_id, inserted)` where `inserted` is True only when a new row
    was created.
    """
    init_db()
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    feature_vector = data.get("feature_vector") or {}
    injury_risk = data.get("injury_risk") or {}
    performance_score = data.get("performance_score")
    shap_perf = data.get("shap_contributions_performance")
    shap_injury = data.get("shap_contributions_injury")
    coaching_notes = data.get("coaching_notes") or []
    stage_times = data.get("stage_times") or {}

    fingerprint = _fingerprint(
        athlete=athlete, label=label, tags=tags, input_mode=input_mode,
        model=model, bowling_arm=bowling_arm, feature_vector=feature_vector,
        performance_score=performance_score)

    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO analyses (
                created_at, label, input_mode, bowling_arm, model,
                athlete, tags, fingerprint,
                performance_score, risk_level, feature_vector, injury_risk,
                shap_performance, shap_injury, coaching_notes, stage_times
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                label or "",
                input_mode or "",
                bowling_arm or "",
                model or "",
                athlete or "",
                tags or "",
                fingerprint,
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
        if cur.rowcount > 0:
            return cur.lastrowid, True
        row = conn.execute(
            "SELECT id FROM analyses WHERE fingerprint = ?", (fingerprint,)).fetchone()
        return (row["id"] if row else None), False
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
        _vacuum(conn)
    finally:
        conn.close()


def clear_all():
    """Delete every saved analysis."""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM analyses")
        conn.commit()
        _vacuum(conn)
    finally:
        conn.close()


def _vacuum(conn):
    """Reclaim free pages left behind by deletes (no-op-safe). VACUUM cannot
    run inside a transaction, so callers must commit before invoking it."""
    try:
        conn.execute("VACUUM")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def count():
    """Number of saved analyses."""
    init_db()
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    finally:
        conn.close()
