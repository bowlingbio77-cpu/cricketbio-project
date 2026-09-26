"""Harness app: render the result page for states the UI cannot reach on its own.

The scoring-withheld and partially-measured states only occur for real video
runs, which the AppTest cannot supply. This builds the AnalysisResult directly
so those trust states are still exercised.

Controlled by session_state["pai_state"]: withheld | partial | flagged | clean.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src import config, ml_models, pipeline, result_view

FEATURES = {
    "elbow_flexion_deg": 8.0,
    "shoulder_rotation_deg": 22.0,
    "trunk_lean_deg": 4.0,
    "wrist_angle_deg": -12.0,
    "knee_flexion_deg": 20.0,
    "hip_rotation_deg": 70.0,
    "stride_length_norm": 1.05,
    "angular_velocity_deg_s": 120.0,
    "vertical_ground_foot_angle_deg": 6.0,
}
FLAGGED = dict(FEATURES, elbow_flexion_deg=25.0, shoulder_rotation_deg=140.0,
               trunk_lean_deg=35.0, wrist_angle_deg=-70.0, stride_length_norm=0.6)

MEASURED = [m[0] for m in result_view.MEASUREMENTS]
PROV_3D = {k: {"source": "world_3d", "confidence": "high",
               "mean_landmark_visibility": 0.93} for k in MEASURED}
PROV_2D = {k: {"source": "normalized_2d", "confidence": "medium",
               "mean_landmark_visibility": 0.55} for k in MEASURED}

state = st.session_state.get("pai_state", "clean")

if state == "withheld":
    result = pipeline.AnalysisResult(
        feature_vector=dict(FEATURES),
        performance_score=None,
        injury_risk={},
        shap_contributions_performance={},
        shap_contributions_injury={},
        coaching_notes=[],
        feature_provenance=dict(PROV_3D),
        subject_verified=False,
        delivery_reliable=True,
        scoring_blocked_reason="the bowler could not be confirmed in this clip",
    )
elif state == "partial":
    result = pipeline.AnalysisResult(
        feature_vector=dict(FEATURES),
        performance_score=68.0,
        injury_risk={"risk_level": "moderate", "probabilities": [0.1, 0.6, 0.3]},
        shap_contributions_performance={},
        shap_contributions_injury={},
        coaching_notes=["Shorten the ground contact before release."],
        feature_provenance=dict(PROV_2D),
        subject_verified=True,
        delivery_reliable=True,
    )
elif state == "flagged":
    result = pipeline.AnalysisResult(
        feature_vector=dict(FLAGGED),
        performance_score=41.0,
        injury_risk={"risk_level": "high", "probabilities": [0.02, 0.13, 0.85]},
        shap_contributions_performance={},
        shap_contributions_injury={},
        coaching_notes=["Reduce trunk lean through the release window."],
        feature_provenance=dict(PROV_3D),
        subject_verified=True,
        delivery_reliable=True,
        bowler_track_id=2,
        bowler_confidence=0.88,
    )
else:
    result = pipeline.AnalysisResult(
        feature_vector=dict(FEATURES),
        performance_score=78.0,
        injury_risk={"risk_level": "low", "probabilities": [0.08, 0.82, 0.1]},
        shap_contributions_performance={},
        shap_contributions_injury={},
        coaching_notes=["Hold your current release window."],
        feature_provenance=dict(PROV_3D),
        subject_verified=True,
        delivery_reliable=True,
        bowler_track_id=2,
        bowler_confidence=0.9,
    )

try:
    perf_bundle = ml_models.load_bundle(config.PERFORMANCE_MODEL_PATH)
except Exception:
    perf_bundle = None
try:
    injury_bundle = ml_models.load_bundle(config.INJURY_MODEL_PATH)
except Exception:
    injury_bundle = None

result_view.render_result_page(
    result,
    perf_bundle=perf_bundle,
    injury_bundle=injury_bundle,
    is_video=True,
    ball_stats={},
)
