"""History database tests: idempotent saves, athlete identity, delete, export."""
import json
import sqlite3

import pytest

from src import history_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(history_db, "DB_PATH", str(tmp_path / "bowling_history_test.db"))
    history_db.init_db()
    return history_db


def _result(score=70.0, features=None):
    return {
        "feature_vector": features or {"elbow_flexion_deg": 8.0, "knee_flexion_deg": 12.0},
        "injury_risk": {"risk_level": "low", "probabilities": [0.9, 0.08, 0.02]},
        "performance_score": score,
        "shap_contributions_performance": {"elbow_flexion_deg": 1.2},
        "shap_contributions_injury": {},
        "coaching_notes": ["keep it up"],
        "stage_times": {"total": 1.0},
    }


def test_save_and_load(db):
    rid, inserted = db.save_analysis(
        _result(), label="net 1", input_mode="sim", bowling_arm="right",
        model="rf", athlete="Usman", tags="nets")
    assert inserted is True
    rows = db.load_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["athlete"] == "Usman"
    assert row["tags"] == "nets"
    assert row["feature_vector"]["elbow_flexion_deg"] == 8.0
    assert row["injury_risk"]["risk_level"] == "low"


def test_save_is_idempotent(db):
    rid1, inserted1 = db.save_analysis(_result(), athlete="Usman", label="net 1")
    rid2, inserted2 = db.save_analysis(_result(), athlete="Usman", label="net 1")
    assert inserted1 is True
    assert inserted2 is False
    assert rid1 == rid2
    assert db.count() == 1


def test_changed_identity_creates_new_row(db):
    db.save_analysis(_result(), athlete="Usman", label="net 1")
    _, inserted = db.save_analysis(_result(), athlete="Ali", label="net 1")
    assert inserted is True
    _, inserted = db.save_analysis(_result(), athlete="Usman", label="match 2")
    assert inserted is True
    assert db.count() == 3


def test_athletes_stay_separate(db):
    db.save_analysis(_result(score=70), athlete="Usman")
    db.save_analysis(_result(score=45), athlete="Ali")
    rows = db.load_all()
    by_athlete = {r["athlete"]: r["performance_score"] for r in rows}
    assert by_athlete == {"Usman": 70.0, "Ali": 45.0}


def test_delete_and_clear(db):
    r1, _ = db.save_analysis(_result(), athlete="Usman")
    r2, _ = db.save_analysis(_result(score=40), athlete="Usman")
    db.delete_analysis([r1])
    assert [r["id"] for r in db.load_all()] == [r2]
    db.clear_all()
    assert db.count() == 0


def test_export_roundtrip(db):
    db.save_analysis(_result(), athlete="Usman")
    db.save_analysis(_result(score=45), athlete="Ali")
    exported = json.dumps(db.load_all(), default=str)
    loaded = json.loads(exported)
    assert len(loaded) == 2
    assert {r["athlete"] for r in loaded} == {"Usman", "Ali"}


def test_migration_adds_new_columns(tmp_path, monkeypatch):
    """A pre-existing database (without athlete/tags/fingerprint) must be
    migrated in place and its rows preserved."""
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(history_db, "DB_PATH", str(db_path))
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
        label TEXT, input_mode TEXT, bowling_arm TEXT, model TEXT,
        performance_score REAL, risk_level TEXT, feature_vector TEXT NOT NULL,
        injury_risk TEXT, shap_performance TEXT, shap_injury TEXT,
        coaching_notes TEXT, stage_times TEXT)""")
    conn.execute(
        "INSERT INTO analyses (created_at, label, feature_vector) VALUES (?, ?, ?)",
        ("2026-01-01 10:00:00", "legacy row", json.dumps({"elbow_flexion_deg": 9.0})))
    conn.commit()
    conn.close()

    history_db.init_db()

    conn = sqlite3.connect(str(db_path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(analyses)")}
    conn.close()
    assert {"athlete", "tags", "fingerprint"} <= cols

    rows = history_db.load_all()
    assert len(rows) == 1
    assert rows[0]["label"] == "legacy row"
    assert rows[0]["feature_vector"] == {"elbow_flexion_deg": 9.0}
