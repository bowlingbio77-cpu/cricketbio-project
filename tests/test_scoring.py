"""Regression tests for performance scoring.

Guards the biomechanics-grounded target formula AND the bundled models: a
"pro" delivery must score high, an injury-prone one low, and the default
Random Forest bundle must agree so the app never shows a fake 24/100 for an
elite setup again.
"""
import os

import pandas as pd
import pytest

from src import config, ml_models
from src.synthetic_data import compute_performance_score

ELITE = {
    "shoulder_rotation_deg": 18.0, "elbow_flexion_deg": 8.0, "wrist_angle_deg": 165.0,
    "hip_rotation_deg": 45.0, "knee_flexion_deg": 10.0, "trunk_lean_deg": 25.0,
    "stride_length_norm": 1.05, "release_angle_deg": 78.0,
    "angular_velocity_deg_s": 1100.0, "ground_contact_time_s": 0.11,
}
RISKY = {
    "shoulder_rotation_deg": 28.0, "elbow_flexion_deg": 22.0, "wrist_angle_deg": 135.0,
    "hip_rotation_deg": 58.0, "knee_flexion_deg": 38.0, "trunk_lean_deg": 45.0,
    "stride_length_norm": 0.72, "release_angle_deg": 60.0,
    "angular_velocity_deg_s": 460.0, "ground_contact_time_s": 0.28,
}
BENT_KNEE = dict(ELITE, knee_flexion_deg=41.9, elbow_flexion_deg=2.4)


def _as_df(feats):
    return pd.DataFrame([feats])


def test_formula_scores_elite_high():
    assert compute_performance_score(_as_df(ELITE)).iloc[0] >= 75


def test_formula_scores_risky_low():
    assert compute_performance_score(_as_df(RISKY)).iloc[0] <= 45


def test_formula_penalises_weak_knee_brace():
    elite = compute_performance_score(_as_df(ELITE)).iloc[0]
    bent = compute_performance_score(_as_df(BENT_KNEE)).iloc[0]
    assert bent < elite - 10  # 41.9° knee brace must hurt the score


@pytest.mark.skipif(
    not os.path.exists(os.path.join(config.MODEL_DIR, "performance_random_forest.joblib")),
    reason="bundled Random Forest model not present")
def test_bundled_rf_scores_elite_high():
    perf = ml_models.load_bundle(os.path.join(config.MODEL_DIR, "performance_random_forest.joblib"))
    assert ml_models.predict(perf, ELITE) >= 70


@pytest.mark.skipif(
    not os.path.exists(os.path.join(config.MODEL_DIR, "injury_random_forest.joblib")),
    reason="bundled Random Forest model not present")
def test_bundled_rf_elite_is_low_injury_risk():
    inj = ml_models.load_bundle(os.path.join(config.MODEL_DIR, "injury_random_forest.joblib"))
    pred = ml_models.predict(inj, ELITE)
    probs = pred.get("probabilities", []) if isinstance(pred, dict) else []
    # label order is {0: low, 1: moderate, 2: high}
    assert len(probs) == 3
    assert probs[0] > probs[2]  # P(low) > P(high)
    assert probs[0] > 0.5       # elite delivery is clearly LOW risk
