"""
Stage 10-11: Streamlit Web Dashboard + Coaching Recommendation UI

Run with:
    streamlit run app.py

Two input modes:
  1. "Manual feature entry" -- move sliders for the 10 biomechanical features
     directly. Works immediately, no video/model downloads needed. Best for
     exploring the model and demoing the dashboard.
  2. "Upload bowling video" -- runs the full CV pipeline (preprocessing ->
     YOLOv11 detection -> ByteTrack -> MediaPipe pose -> feature engineering).
     Requires the extra model downloads described in src/pose_estimation.py
     and src/detection.py docstrings (needs internet on first run).
"""
import os
import tempfile
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src import config, ml_models, explainability, coaching, pipeline
from src.synthetic_data import generate_synthetic_dataset

st.set_page_config(page_title="Cricket Bowling Biomechanics AI", layout="wide", page_icon="🏏")


# ---------- Cached resource loading ----------
@st.cache_resource
def load_or_train_models(model_name: str = "random_forest"):
    perf_path = os.path.join(config.MODEL_DIR, f"performance_{model_name}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{model_name}.joblib")

    if os.path.exists(perf_path) and os.path.exists(injury_path):
        return ml_models.load_bundle(perf_path), ml_models.load_bundle(injury_path)

    st.info(f"No saved '{model_name}' models found -- training on synthetic demo data now "
            f"(run `python train_demo_model.py` once to cache this).")
    df = generate_synthetic_dataset()
    X = df[config.FEATURE_NAMES].values
    perf_bundle = ml_models.train_performance_model(X, df[config.PERFORMANCE_TARGET].values, model_name)
    injury_bundle = ml_models.train_injury_model(X, df[config.INJURY_TARGET].values, model_name)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    ml_models.save_bundle(perf_bundle, perf_path)
    ml_models.save_bundle(injury_bundle, injury_path)
    return perf_bundle, injury_bundle


@st.cache_data
def get_background_data():
    df = generate_synthetic_dataset(n_samples=200)
    return df[config.FEATURE_NAMES].values


FEATURE_LABELS = {
    "shoulder_rotation_deg": ("Shoulder rotation", "deg", 0, 90, 40),
    "elbow_flexion_deg": ("Elbow flexion (extension from straight)", "deg", 0, 45, 10),
    "wrist_angle_deg": ("Wrist angle", "deg", 90, 180, 150),
    "hip_rotation_deg": ("Hip rotation", "deg", 0, 80, 35),
    "knee_flexion_deg": ("Front-knee flexion", "deg", 0, 60, 18),
    "trunk_lean_deg": ("Trunk lateral lean", "deg", 0, 60, 27),
    "stride_length_norm": ("Stride length (normalized)", "x height", 0.3, 1.6, 0.95),
    "release_angle_deg": ("Release angle", "deg", 30, 90, 75),
    "angular_velocity_deg_s": ("Peak angular velocity", "deg/s", 100, 1500, 700),
    "ground_contact_time_s": ("Front-foot ground contact time", "s", 0.05, 0.35, 0.15),
}


def render_gauge(value, title, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": "#2E7D32" if value >= 70 else "#F9A825" if value >= 50 else "#C62828"},
            "steps": [
                {"range": [0, 50], "color": "#FFCDD2"},
                {"range": [50, 70], "color": "#FFF9C4"},
                {"range": [70, max_val], "color": "#C8E6C9"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_shap_bar(contributions: dict, title: str):
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]))
    names = [FEATURE_LABELS.get(k, (k,))[0] for k, _ in items]
    values = [v for _, v in items]
    colors = ["#C62828" if v > 0 else "#1565C0" for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(title=title, height=350, margin=dict(l=10, r=10, t=40, b=10),
                       xaxis_title="Contribution")
    return fig


def render_timings(stage_times: dict):
    if not stage_times:
        st.caption("No timing data available for this run.")
        return
    ordered = sorted(stage_times.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in ordered]
    values = [v for _, v in ordered]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=["#1565C0" if k != "total" else "#2E7D32"
                                         for k, _ in ordered]))
    fig.update_layout(title="Stage timing", height=320,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Seconds")
    st.plotly_chart(fig, use_container_width=True)
    total = stage_times.get("total")
    if total is not None:
        st.caption(f"**Total pipeline time: {total:.2f}s**")


# ---------- Sidebar ----------
st.sidebar.title("🏏 Bowling Biomechanics AI")
st.sidebar.caption("Cricket bowling action analysis, injury-risk screening, and coaching feedback.")

model_choice = st.sidebar.selectbox(
    "ML model", ["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"],
    help="XGBoost/CatBoost/CNN-LSTM/Transformer fall back to a scikit-learn "
         "equivalent automatically if their library isn't installed.")

input_mode = st.sidebar.radio("Input mode", ["Manual feature entry", "Upload bowling video"])
bowling_arm = st.sidebar.selectbox("Bowling arm", ["right", "left"])

st.sidebar.markdown("**Processing speed / accuracy**")
processing_preset = st.sidebar.selectbox(
    "Preset", ["Balanced", "Fast", "Maximum accuracy"],
    help="Balanced (default): denoise off, 640×360 @ 20 fps -- most speed with little "
         "accuracy loss. Fast: same resolution @ 15 fps. Maximum accuracy: denoise on, "
         "960×540 @ 30 fps (slowest).")
if processing_preset == "Fast":
    target_fps, resize_choice, denoise = 15, (640, 360), False
elif processing_preset == "Maximum accuracy":
    target_fps, resize_choice, denoise = 30, (960, 540), True
else:
    target_fps, resize_choice, denoise = 20, (640, 360), False

with st.sidebar.expander("Fine-tune"):
    target_fps = st.slider("Target FPS", 5, 30, target_fps, step=5,
                           help="Lower = fewer frames for pose estimation = faster.")
    resize_choice = st.selectbox(
        "Frame resolution",
        [(640, 360), (960, 540), (1280, 720)],
        format_func=lambda d: f"{d[0]}×{d[1]}", index=[(640, 360), (960, 540), (1280, 720)].index(resize_choice))
    denoise = st.checkbox("Denoise frames", value=denoise,
                          help="On = more accurate on noisy footage but much slower.")

perf_bundle, injury_bundle = load_or_train_models(model_choice)

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend status**")
st.sidebar.text(f"XGBoost installed:  {ml_models.BACKEND_INFO['xgboost_available']}")
st.sidebar.text(f"CatBoost installed: {ml_models.BACKEND_INFO['catboost_available']}")
st.sidebar.text(f"PyTorch installed:  {ml_models.BACKEND_INFO['torch_available']}")
st.sidebar.text(f"SHAP installed:     {explainability.SHAP_AVAILABLE}")


# ---------- Main ----------
st.title("Cricket Bowling Action Analysis Dashboard")

feature_vector = {}
stage_times = {}

if input_mode == "Manual feature entry":
    st.subheader("1. Biomechanical Feature Input")
    st.caption("Enter (or simulate) the 10 release-point biomechanical features for one delivery.")
    cols = st.columns(2)
    for i, (feat, (label, unit, lo, hi, default)) in enumerate(FEATURE_LABELS.items()):
        with cols[i % 2]:
            feature_vector[feat] = st.slider(f"{label} ({unit})", float(lo), float(hi), float(default))

else:
    st.subheader("1. Upload Bowling Video")
    st.caption("Runs the full CV pipeline: preprocessing → YOLOv11 detection → ByteTrack → "
               "MediaPipe pose (33 landmarks) → biomechanical feature extraction.")
    uploaded = st.file_uploader("Upload a single-delivery clip", type=["mp4", "mov", "avi"])

    if uploaded is not None:
        upload_t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
        upload_time = time.perf_counter() - upload_t0

        if not os.path.exists(config.POSE_MODEL_PATH):
            st.error(
                "Pose model not found. Download it once (needs internet):\n\n"
                "`wget -O models/pose_landmarker_heavy.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`"
            )
        else:
            with st.spinner("Running detection, tracking, and pose estimation..."):
                try:
                    result = pipeline.analyze_video(video_path, bowling_arm=bowling_arm,
                                                      performance_bundle=perf_bundle,
                                                      injury_bundle=injury_bundle,
                                                      target_fps=target_fps,
                                                      resize_dim=resize_choice,
                                                      denoise=denoise)
                    feature_vector = result.feature_vector
                    stage_times = dict(result.stage_times or {})
                    stage_times["upload"] = upload_time
                    st.success("Pose extracted and features computed.")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

if feature_vector:
    st.subheader("2. Extracted / Input Features")
    feat_df = pd.DataFrame([
        {"Feature": FEATURE_LABELS.get(k, (k,))[0], "Value": round(v, 2),
         "Unit": FEATURE_LABELS.get(k, ("", ""))[1]}
        for k, v in feature_vector.items()
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.subheader("3. Machine Learning Predictions")
    result = pipeline.analyze_feature_vector(feature_vector, perf_bundle, injury_bundle)
    stage_times = dict(result.stage_times or {})

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(render_gauge(result.performance_score, "Performance Score"),
                         use_container_width=True)
    with c2:
        risk = result.injury_risk
        risk_num = {"low": 25, "moderate": 60, "high": 90}[risk["risk_level"]]
        st.plotly_chart(render_gauge(risk_num, f"Injury Risk: {risk['risk_level'].upper()}"),
                         use_container_width=True)
        if risk["probabilities"]:
            st.caption(f"P(low)={risk['probabilities'][0]:.2f}  "
                       f"P(moderate)={risk['probabilities'][1]:.2f}  "
                       f"P(high)={risk['probabilities'][2]:.2f}")

    st.subheader("4. Run Timing")
    render_timings(stage_times)

    st.subheader("5. Explainable AI — Why this prediction?")
    tab1, tab2 = st.tabs(["Performance drivers", "Injury-risk drivers"])
    with tab1:
        if result.shap_contributions_performance:
            st.plotly_chart(render_shap_bar(result.shap_contributions_performance,
                                             "Feature contribution to performance score"),
                             use_container_width=True)
        if not explainability.SHAP_AVAILABLE:
            st.caption("SHAP not installed -- showing finite-difference sensitivity as a fallback "
                       "(`pip install shap` for full SHAP explanations).")
    with tab2:
        if result.shap_contributions_injury:
            st.plotly_chart(render_shap_bar(result.shap_contributions_injury,
                                             "Feature contribution to injury-risk score"),
                             use_container_width=True)

    st.subheader("6. Coaching Recommendations")
    for note in result.coaching_notes:
        st.markdown(f"- {note}")

else:
    st.info("Enter features manually or upload a video to run the analysis.")

st.markdown("---")
st.caption(
    "Demo models are trained on synthetic data (src/synthetic_data.py) grounded in published "
    "fast-bowling biomechanics ranges -- retrain on real labeled data "
    "(`python train_demo_model.py --data your_dataset.csv`) before using for real coaching/medical decisions."
)
