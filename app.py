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

from src import config, ml_models, explainability, coaching, pipeline, history_db
from src.synthetic_data import generate_synthetic_dataset

st.set_page_config(page_title="Cricket Bowling Biomechanics AI", layout="wide", page_icon="🏏")


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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


def _session_name(row):
    parts = [f"#{row['id']}", row["created_at"]]
    if row.get("label"):
        parts.append(f"— {row['label']}")
    return " ".join(parts)


def _risk_of(row):
    risk = row.get("injury_risk")
    if isinstance(risk, dict):
        return risk.get("risk_level", "—")
    return row.get("risk_level") or "—"


def render_history_page():
    records = history_db.load_all()
    st.title("History & Comparison")
    st.caption("Every delivery you saved from the Analyze page lives here -- browse past "
               "results, compare sessions side by side, and track performance over time.")

    if not records:
        st.info("No saved results yet. Go to **Analyze**, run a delivery, and press "
                "**Save this result to history**.")
        return

    # --- Summary metrics ---
    perfs = [r["performance_score"] for r in records if r.get("performance_score") is not None]
    risks = [_risk_of(r) for r in records]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sessions", len(records))
    c2.metric("Avg performance", f"{np.mean(perfs):.1f}" if perfs else "—")
    c3.metric("Best performance", f"{max(perfs):.1f}" if perfs else "—")
    c4.metric("High-risk sessions", risks.count("high"))

    # --- Full table ---
    st.subheader("All saved sessions")
    table = pd.DataFrame([
        {
            "ID": r["id"],
            "Date": r["created_at"],
            "Label": r.get("label") or "",
            "Mode": r.get("input_mode") or "",
            "Arm": r.get("bowling_arm") or "",
            "Model": r.get("model") or "",
            "Performance": round(r["performance_score"], 1) if r.get("performance_score") is not None else None,
            "Risk": _risk_of(r),
        }
        for r in records
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # --- Performance over time ---
    chrono = sorted([r for r in records if r.get("performance_score") is not None],
                    key=lambda r: r["created_at"])
    if len(chrono) >= 2:
        st.subheader("Performance over time")
        fig = go.Figure(go.Scatter(
            x=[r["created_at"] for r in chrono],
            y=[r["performance_score"] for r in chrono],
            mode="lines+markers+text",
            text=[f"#{r['id']}" for r in chrono],
            textposition="top center",
            line=dict(color="#2E7D32", width=2),
        ))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Date", yaxis_title="Performance score",
                          yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)
    elif chrono:
        st.subheader("Performance over time")
        st.caption("Save more results to see a performance trend chart.")

    # --- Comparison ---
    options = {r["id"]: _session_name(r) for r in records}
    st.subheader("Compare sessions")
    st.caption("Pick two or more sessions to compare side by side.")
    selected = st.multiselect("Sessions", list(options.keys()),
                              format_func=lambda i: options[i],
                              key="compare_select")
    if len(selected) >= 1:
        sel = history_db.load_by_ids(selected)
        sel_names = [_session_name(r) for r in sel]

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=sel_names, y=[r.get("performance_score") for r in sel],
                marker_color="#2E7D32", text=[f"{r.get('performance_score'):.0f}" if r.get('performance_score') is not None else "—" for r in sel],
                textposition="outside"))
            fig.update_layout(title="Performance score", height=320,
                              margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=sel_names, y=[_risk_of(r) for r in sel],
                marker_color=["#C62828" if _risk_of(r) == "high" else "#F9A825"
                              if _risk_of(r) == "moderate" else "#2E7D32" for r in sel],
                text=[_risk_of(r).title() for r in sel], textposition="outside"))
            fig.update_layout(title="Injury risk", height=320,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Feature-by-feature comparison**")
        feat_rows = []
        for feat, (label, unit, *_rest) in FEATURE_LABELS.items():
            vals = [r["feature_vector"].get(feat) for r in sel]
            row = {"Feature": label, "Unit": unit}
            for name, v in zip(sel_names, vals):
                row[name] = round(v, 2) if v is not None else None
            if len(vals) > 1 and all(v is not None for v in vals):
                row["Δ last vs first"] = round(vals[-1] - vals[0], 2)
            feat_rows.append(row)
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        if len(selected) >= 2:
            fig = go.Figure()
            for name, r in zip(sel_names, sel):
                x = [FEATURE_LABELS[k][0] for k in config.FEATURE_NAMES]
                y = [r["feature_vector"].get(k) for k in config.FEATURE_NAMES]
                fig.add_trace(go.Bar(x=x, y=y, name=name))
            fig.update_layout(title="Feature values by session", barmode="group",
                              height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # --- Detail view ---
        st.subheader("Session details")
        detail_id = st.selectbox("Pick a session to inspect", list(options.keys()),
                                 format_func=lambda i: options[i], key="detail_select")
        detail = history_db.load_by_ids([detail_id])[0]
        c1, c2 = st.columns(2)
        with c1:
            perf = detail.get("performance_score")
            st.plotly_chart(render_gauge(perf if perf is not None else 0,
                                         "Performance Score" if perf is not None else "Performance (n/a)"),
                            use_container_width=True)
        with c2:
            risk = detail.get("injury_risk")
            if isinstance(risk, dict) and risk.get("probabilities"):
                probs = risk["probabilities"]
                risk_num = {"low": 25, "moderate": 60, "high": 90}[risk.get("risk_level", "low")]
                st.plotly_chart(render_gauge(risk_num, f"Injury Risk: {risk['risk_level'].upper()}"),
                                use_container_width=True)
                if len(probs) >= 3:
                    st.caption(f"P(low)={probs[0]:.2f}  P(moderate)={probs[1]:.2f}  P(high)={probs[2]:.2f}")
            else:
                st.info("No injury-risk prediction stored for this session.")
        with st.expander("Features"):
            feat_df = pd.DataFrame([
                {"Feature": FEATURE_LABELS.get(k, (k,))[0],
                 "Value": round(v, 2) if v is not None else None,
                 "Unit": FEATURE_LABELS.get(k, ("", ""))[1]}
                for k, v in detail["feature_vector"].items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        with st.expander("Coaching recommendations"):
            notes = detail.get("coaching_notes") or []
            for note in notes:
                st.markdown(f"- {note}")
            if not notes:
                st.caption("No coaching notes stored for this session.")
        with st.expander("Explainable AI — feature contributions"):
            tab1, tab2 = st.tabs(["Performance drivers", "Injury-risk drivers"])
            with tab1:
                shap_perf = detail.get("shap_performance")
                if shap_perf:
                    st.plotly_chart(render_shap_bar(shap_perf,
                                                    "Feature contribution to performance score"),
                                    use_container_width=True)
                else:
                    st.caption("No performance SHAP data stored for this session.")
            with tab2:
                shap_injury = detail.get("shap_injury")
                if shap_injury:
                    st.plotly_chart(render_shap_bar(shap_injury,
                                                    "Feature contribution to injury-risk score"),
                                    use_container_width=True)
                else:
                    st.caption("No injury SHAP data stored for this session.")
        with st.expander("Run timing"):
            render_timings(detail.get("stage_times") or {})

    # --- Manage history ---
    st.subheader("Manage history")
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        del_opts = st.multiselect("Select sessions to delete", list(options.keys()),
                                  format_func=lambda i: options[i], key="delete_select")
    with m2:
        if st.button("Delete selected", use_container_width=True):
            history_db.delete_analysis(del_opts)
            st.toast(f"Deleted {len(del_opts)} session(s).")
            rerun()
    with m3:
        if st.button("Clear all history", use_container_width=True):
            history_db.clear_all()
            st.toast("History cleared.")
            rerun()


# ---------- Sidebar ----------
st.sidebar.title("🏏 Bowling Biomechanics AI")
st.sidebar.caption("Cricket bowling action analysis, injury-risk screening, and coaching feedback.")

page = st.sidebar.radio("Navigation", ["Analyze", "History & Compare"],
                        help="Analyze: run a new delivery. History & Compare: browse saved results and track progress.")

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

if page == "Analyze":
    perf_bundle, injury_bundle = load_or_train_models(model_choice)
else:
    perf_bundle, injury_bundle = None, None

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend status**")
st.sidebar.text(f"XGBoost installed:  {ml_models.BACKEND_INFO['xgboost_available']}")
st.sidebar.text(f"CatBoost installed: {ml_models.BACKEND_INFO['catboost_available']}")
st.sidebar.text(f"PyTorch installed:  {ml_models.BACKEND_INFO['torch_available']}")
st.sidebar.text(f"SHAP installed:     {explainability.SHAP_AVAILABLE}")


# ---------- Main ----------
if page == "History & Compare":
    render_history_page()
    st.stop()

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

    st.subheader("7. Save to History")
    st.caption("Persist this delivery's results to the local history database so you can "
               "compare it against future sessions and track your performance over time.")
    save_label = st.text_input("Label (optional)", value=st.session_state.get("save_label", ""),
                               placeholder="e.g. Net session 1, match 3 over 4",
                               key="save_label")
    if st.button("💾 Save this result to history", type="primary"):
        saved_id = history_db.save_analysis(
            result, label=save_label, input_mode=input_mode,
            bowling_arm=bowling_arm, model=model_choice)
        st.success(f"Saved to history (id #{saved_id}). Open **History & Compare** in the "
                   f"sidebar to view and compare your saved results.")

else:
    st.info("Enter features manually or upload a video to run the analysis.")

st.markdown("---")
st.caption(
    "Demo models are trained on synthetic data (src/synthetic_data.py) grounded in published "
    "fast-bowling biomechanics ranges -- retrain on real labeled data "
    "(`python train_demo_model.py --data your_dataset.csv`) before using for real coaching/medical decisions."
)
