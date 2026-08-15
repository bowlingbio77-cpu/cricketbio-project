"""
PaceAI — Cricket Bowling Biomechanics AI
Pro Coaching & Injury Analytics Dashboard (dark theme)

Run with:
    streamlit run app.py

Two analysis modes:
  1. Interactive Bio-Simulator -- slider-based kinematic entry with elite presets.
  2. Video Motion Capture -- full CV pipeline (preprocessing -> YOLOv11
     detection -> ByteTrack -> MediaPipe pose -> feature engineering).

Also includes:
  - History & Compare: local SQLite persistence of every saved delivery.
  - Model quality & validity honesty panel (synthetic-data disclaimer).
"""
import os
import json
import tempfile
import time
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src import config, ml_models, explainability, pipeline, history_db, injury_knowledge_base as injury_kb
from src.synthetic_data import generate_synthetic_dataset

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="PaceAI | Cricket Bowling Biomechanics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background & headers */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Card Containers */
    .metric-card {
        background: linear-gradient(145deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-low { background-color: rgba(46, 125, 50, 0.2); color: #4caf50; border: 1px solid #4caf50; }
    .badge-moderate { background-color: rgba(249, 168, 37, 0.2); color: #fbc02d; border: 1px solid #fbc02d; }
    .badge-high { background-color: rgba(198, 40, 40, 0.2); color: #ef5350; border: 1px solid #ef5350; }
    .badge-legal { background-color: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid #00e676; }
    .badge-illegal { background-color: rgba(255, 23, 68, 0.15); color: #ff1744; border: 1px solid #ff1744; }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 25px;
        border: 1px solid #38444d;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Drill card */
    .drill-card {
        border-left: 4px solid #00e676;
        background: #161b22;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOADING OVERLAY ----------------
@st.cache_data
def _loader_html() -> str:
    path = os.path.join(config.ASSETS_DIR, "loading_overlay.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


def render_loader(message: str = "Extracting 33 3D Pose Landmarks...",
                  fps: str = "120", knee: str = "OPTIMAL", risk: str = "CALC..."):
    """Render the animated Biomech AI loading overlay (self-playing CSS/JS)."""
    html = (_loader_html()
            .replace("{{MESSAGE}}", message)
            .replace("{{FPS}}", fps)
            .replace("{{KNEE}}", knee)
            .replace("{{RISK}}", risk))
    components.html(html, height=560, scrolling=False)


def render_fullscreen_splash(message: str = "Booting PaceAI Engine...",
                             fps: str = "—", knee: str = "BOOTING", risk: str = "BOOT"):
    """Full-viewport splash used during app/model startup (CSS animates, JS skipped)."""
    html = (_loader_html()
            .replace("{{MESSAGE}}", message)
            .replace("{{FPS}}", fps)
            .replace("{{KNEE}}", knee)
            .replace("{{RISK}}", risk))
    st.markdown(
        '<div style="position:fixed;top:0;left:0;right:0;bottom:0;z-index:99999;'
        'background:#0b0f19;display:flex;align-items:center;justify-content:center;'
        'overflow:hidden;">' + html + "</div>",
        unsafe_allow_html=True,
    )


# ---------------- HELPERS ----------------
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------------- CACHED MODEL LOADING ----------------
@st.cache_resource
def load_or_train_models(model_name: str = "random_forest"):
    perf_path = os.path.join(config.MODEL_DIR, f"performance_{model_name}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{model_name}.joblib")

    if os.path.exists(perf_path) and os.path.exists(injury_path):
        return ml_models.load_bundle(perf_path), ml_models.load_bundle(injury_path)

    st.info(f"No saved '{model_name}' models found -- training on synthetic demo data now "
            f"(run `python train_demo_model.py` once to cache this).")
    render_loader(message=f"Training {model_name} models on synthetic demo data...",
                  fps="—", knee="BOOTING", risk="TRAIN")
    df = generate_synthetic_dataset()
    X = df[config.FEATURE_NAMES].values
    perf_bundle = ml_models.train_performance_model(X, df[config.PERFORMANCE_TARGET].values, model_name)
    injury_bundle = ml_models.train_injury_model(X, df[config.INJURY_TARGET].values, model_name)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    ml_models.save_bundle(perf_bundle, perf_path)
    ml_models.save_bundle(injury_bundle, injury_path)
    return perf_bundle, injury_bundle


# ---------------- BOOT SPLASH ----------------
# On the very first run, show a full-screen loading overlay while the default
# model bundle is loaded/cached, then re-run into the real dashboard.
if not st.session_state.get("booted", False):
    st.session_state["booted"] = True
    render_fullscreen_splash("Booting PaceAI Biomechanics Engine...")
    _ = load_or_train_models("random_forest")
    rerun()


FEATURE_LABELS = {
    "shoulder_rotation_deg": ("Shoulder Rotation", "deg", 0, 90, 42.0),
    "elbow_flexion_deg": ("Elbow Flexion", "deg", 0, 45, 11.5),
    "wrist_angle_deg": ("Wrist Angle", "deg", 90, 180, 155.0),
    "hip_rotation_deg": ("Hip Rotation", "deg", 0, 80, 36.0),
    "knee_flexion_deg": ("Front-Knee Flexion", "deg", 0, 60, 16.0),
    "trunk_lean_deg": ("Trunk Lateral Lean", "deg", 0, 60, 24.0),
    "stride_length_norm": ("Stride Length (norm)", "x H", 0.3, 1.6, 0.98),
    "release_angle_deg": ("Release Angle", "deg", 30, 90, 76.0),
    "angular_velocity_deg_s": ("Peak Angular Velocity", "deg/s", 100, 1500, 780.0),
    "ground_contact_time_s": ("Front Foot Contact Time", "s", 0.05, 0.35, 0.14),
}

# Elite Fast Bowler Benchmark for comparison
ELITE_BENCHMARK = {
    "shoulder_rotation_deg": 48.0,
    "elbow_flexion_deg": 10.0,
    "wrist_angle_deg": 160.0,
    "hip_rotation_deg": 42.0,
    "knee_flexion_deg": 12.0,
    "trunk_lean_deg": 22.0,
    "stride_length_norm": 1.05,
    "release_angle_deg": 78.0,
    "angular_velocity_deg_s": 880.0,
    "ground_contact_time_s": 0.12,
}

PRESETS = {
    "Custom / Manual": None,
    "⚡ Elite Express Pace (145+ km/h)": {
        "shoulder_rotation_deg": 52.0, "elbow_flexion_deg": 8.5, "wrist_angle_deg": 165.0,
        "hip_rotation_deg": 46.0, "knee_flexion_deg": 10.0, "trunk_lean_deg": 20.0,
        "stride_length_norm": 1.12, "release_angle_deg": 80.0, "angular_velocity_deg_s": 950.0,
        "ground_contact_time_s": 0.11
    },
    "🚨 High Lumbar Injury Risk Action": {
        "shoulder_rotation_deg": 28.0, "elbow_flexion_deg": 22.0, "wrist_angle_deg": 135.0,
        "hip_rotation_deg": 58.0, "knee_flexion_deg": 38.0, "trunk_lean_deg": 45.0,
        "stride_length_norm": 0.72, "release_angle_deg": 60.0, "angular_velocity_deg_s": 460.0,
        "ground_contact_time_s": 0.28
    },
    "🎯 Seam & Swing Specialist": {
        "shoulder_rotation_deg": 40.0, "elbow_flexion_deg": 12.0, "wrist_angle_deg": 172.0,
        "hip_rotation_deg": 34.0, "knee_flexion_deg": 18.0, "trunk_lean_deg": 26.0,
        "stride_length_norm": 0.95, "release_angle_deg": 74.0, "angular_velocity_deg_s": 720.0,
        "ground_contact_time_s": 0.15
    },
    "🌀 Mystery Spin Action": {
        "shoulder_rotation_deg": 65.0, "elbow_flexion_deg": 14.0, "wrist_angle_deg": 120.0,
        "hip_rotation_deg": 25.0, "knee_flexion_deg": 25.0, "trunk_lean_deg": 15.0,
        "stride_length_norm": 0.65, "release_angle_deg": 68.0, "angular_velocity_deg_s": 520.0,
        "ground_contact_time_s": 0.22
    }
}


# ---------------- CHART BUILDERS ----------------
def render_modern_gauge(value, title, subtitle="", max_val=100, is_risk=False):
    if is_risk:
        bar_color = "#ef5350" if value >= 70 else "#fbc02d" if value >= 40 else "#00e676"
    else:
        bar_color = "#00e676" if value >= 70 else "#fbc02d" if value >= 50 else "#ef5350"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:#8b949e'>{subtitle}</span>"},
        number={"font": {"size": 42, "color": "#ffffff"}, "suffix": "%" if is_risk else ""},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "borderwidth": 1,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 40], "color": "rgba(46, 125, 50, 0.15)" if not is_risk else "rgba(0, 230, 118, 0.15)"},
                {"range": [40, 70], "color": "rgba(249, 168, 37, 0.15)"},
                {"range": [70, max_val], "color": "rgba(0, 230, 118, 0.2)" if not is_risk else "rgba(239, 83, 80, 0.25)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"},
        height=240,
        margin=dict(l=25, r=25, t=60, b=20)
    )
    return fig


def render_radar_comparison(current_feats: dict):
    categories = [
        "Shoulder Rot.", "Arm Extension", "Wrist Cocking", "Hip Rotation",
        "Knee Brace", "Upright Trunk", "Stride Prowess", "Release Velocity"
    ]

    def normalize(val, lo, hi):
        return max(0, min(100, ((val - lo) / (hi - lo)) * 100))

    user_vals = [
        normalize(current_feats["shoulder_rotation_deg"], 0, 90),
        normalize(45 - current_feats["elbow_flexion_deg"], 0, 45),
        normalize(current_feats["wrist_angle_deg"], 90, 180),
        normalize(current_feats["hip_rotation_deg"], 0, 80),
        normalize(60 - current_feats["knee_flexion_deg"], 0, 60),  # Braced = lower flexion
        normalize(60 - current_feats["trunk_lean_deg"], 0, 60),
        normalize(current_feats["stride_length_norm"], 0.3, 1.6),
        normalize(current_feats["angular_velocity_deg_s"], 100, 1500),
    ]

    bench_vals = [
        normalize(ELITE_BENCHMARK["shoulder_rotation_deg"], 0, 90),
        normalize(45 - ELITE_BENCHMARK["elbow_flexion_deg"], 0, 45),
        normalize(ELITE_BENCHMARK["wrist_angle_deg"], 90, 180),
        normalize(ELITE_BENCHMARK["hip_rotation_deg"], 0, 80),
        normalize(60 - ELITE_BENCHMARK["knee_flexion_deg"], 0, 60),
        normalize(60 - ELITE_BENCHMARK["trunk_lean_deg"], 0, 60),
        normalize(ELITE_BENCHMARK["stride_length_norm"], 0.3, 1.6),
        normalize(ELITE_BENCHMARK["angular_velocity_deg_s"], 100, 1500),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_vals, theta=categories, fill='toself',
        name='Current Bowler',
        line=dict(color='#00e676', width=2),
        fillcolor='rgba(0, 230, 118, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=bench_vals, theta=categories, fill='toself',
        name='Elite Benchmark (145 km/h)',
        line=dict(color='#29b6f6', width=2, dash='dot'),
        fillcolor='rgba(41, 182, 246, 0.1)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#6e7681", gridcolor="#30363d"),
            angularaxis=dict(color="#c9d1d9", gridcolor="#30363d")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=380,
        margin=dict(l=40, r=40, t=30, b=40)
    )
    return fig


def render_shap_bar(contributions: dict, title: str):
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    items = sorted(items, key=lambda kv: kv[1])
    names = [FEATURE_LABELS.get(k, (k,))[0] for k, _ in items]
    values = [v for _, v in items]
    colors = ["#ef5350" if v > 0 else "#29b6f6" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color="#ffffff", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="Relative Model Impact (SHAP value)", gridcolor="#30363d", zerolinecolor="#8b949e"),
        yaxis=dict(gridcolor="#21262d"),
        height=340,
        margin=dict(l=10, r=20, t=40, b=20)
    )
    return fig


def render_timings(stage_times: dict):
    if not stage_times:
        st.caption("No timing data available for this run.")
        return
    ordered = sorted(stage_times.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in ordered]
    values = [v for _, v in ordered]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=["#29b6f6" if k != "total" else "#00e676"
                                         for k, _ in ordered]))
    fig.update_layout(title="Stage timing", height=320,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d1d9"),
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Seconds")
    st.plotly_chart(fig, use_container_width=True)
    total = stage_times.get("total")
    if total is not None:
        st.caption(f"**Total pipeline time: {total:.2f}s**")


# ---------------- MODEL HONESTY / VALIDITY PANEL ----------------
def data_source_of(bundle):
    if bundle is None:
        return "n/a"
    return getattr(bundle, "data_source", "unknown")


def render_model_quality_expander(perf_bundle, injury_bundle):
    """Honesty panel: data provenance, CV metrics, and baseline comparison."""
    with st.expander("Model quality & validity (read this)", expanded=False):
        src = data_source_of(perf_bundle)
        if src == "synthetic":
            st.warning(
                "Models are trained on **SYNTHETIC demo data** whose labels are generated from "
                "the features themselves (src/synthetic_data.py). Near-perfect metrics below are "
                "expected -- the model is effectively re-learning the generator's formula and they "
                "say **nothing** about real-world accuracy. Retrain on real labeled data "
                "(`python train_demo_model.py --data your_data.csv`) before using for coaching "
                "or medical decisions.")
        elif src == "real":
            st.info("Trained on a real labeled dataset -- but still validate on a fresh holdout "
                    "population before using it in production.")
        else:
            st.caption("Model provenance unknown (bundle saved by an older version).")

        if perf_bundle is not None:
            cv = getattr(perf_bundle, "cv_metrics", None) or {}
            bl = getattr(perf_bundle, "baseline_metrics", None) or {}
            folds = cv.get("folds", 0)
            st.markdown(f"**Performance model** (`{perf_bundle.model_name}`)"
                        + (f" — {folds}-fold cross-validation" if folds else ""))
            if cv:
                st.markdown(f"- MAE: **{cv.get('mae_mean', 0):.2f}** ± {cv.get('mae_std', 0):.2f} "
                            f"points / 100")
                st.markdown(f"- RMSE: **{cv.get('rmse_mean', 0):.2f}**")
                st.markdown(f"- R²: **{cv.get('r2_mean', 0):.3f}** ± {cv.get('r2_std', 0):.3f} "
                            f"— baseline (always predict mean): **{bl.get('r2', 0):.3f}**")
            else:
                st.caption("No cross-validation data stored in this bundle.")

        if injury_bundle is not None:
            cv = getattr(injury_bundle, "cv_metrics", None) or {}
            bl = getattr(injury_bundle, "baseline_metrics", None) or {}
            folds = cv.get("folds", 0)
            st.markdown(f"**Injury-risk model** (`{injury_bundle.model_name}`)"
                        + (f" — {folds}-fold cross-validation" if folds else ""))
            if cv:
                st.markdown(f"- Accuracy: **{cv.get('accuracy_mean', 0):.3f}** ± "
                            f"{cv.get('accuracy_std', 0):.3f} — baseline (always majority class): "
                            f"**{bl.get('accuracy', 0):.3f}**")
                st.markdown(f"- F1 (macro): **{cv.get('f1_mean', 0):.3f}** ± {cv.get('f1_std', 0):.3f}")
            else:
                st.caption("No cross-validation data stored in this bundle.")

        if src == "synthetic":
            st.caption("These numbers tell you the model fits the demo generator, not that it "
                       "predicts bowling outcomes. Treat all scores on the dashboard as illustrative.")


def render_ood_warnings(feature_vector, bundle):
    ood = ml_models.out_of_distribution_warnings(feature_vector, bundle)
    if ood:
        lines = [f"- **{FEATURE_LABELS.get(f, (f,))[0]}** = {v:.2f} "
                 f"(training range {lo:.2f}–{hi:.2f})" for f, v, lo, hi in ood]
        st.warning("Some input features fall outside the model's training range -- predictions "
                   "extrapolate beyond what the model has seen and may be unreliable:\n"
                   + "\n".join(lines))


# ---------------- HISTORY & COMPARE PAGE ----------------
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
                "**Save to History**.")
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
            line=dict(color="#00e676", width=2),
        ))
        fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=30, b=10),
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
                marker_color="#00e676", text=[f"{r.get('performance_score'):.0f}" if r.get('performance_score') is not None else "—" for r in sel],
                textposition="outside"))
            fig.update_layout(title="Performance score", height=320,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=40, b=10),
                              yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=sel_names, y=[_risk_of(r) for r in sel],
                marker_color=["#ef5350" if _risk_of(r) == "high" else "#fbc02d"
                              if _risk_of(r) == "moderate" else "#00e676" for r in sel],
                text=[_risk_of(r).title() for r in sel], textposition="outside"))
            fig.update_layout(title="Injury risk", height=320,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=40, b=10))
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
                              height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # --- Detail view ---
        st.subheader("Session details")
        detail_id = st.selectbox("Pick a session to inspect", list(options.keys()),
                                 format_func=lambda i: options[i], key="detail_select")
        detail = history_db.load_by_ids([detail_id])[0]
        c1, c2 = st.columns(2)
        with c1:
            perf = detail.get("performance_score")
            st.plotly_chart(render_modern_gauge(perf if perf is not None else 0,
                                                "Performance Score" if perf is not None else "Performance (n/a)"),
                            use_container_width=True)
        with c2:
            risk = detail.get("injury_risk")
            if isinstance(risk, dict) and risk.get("probabilities"):
                probs = risk["probabilities"]
                risk_num = {"low": 25, "moderate": 60, "high": 90}[risk.get("risk_level", "low")]
                st.plotly_chart(render_modern_gauge(risk_num, f"Injury Risk: {risk['risk_level'].upper()}",
                                                    is_risk=True), use_container_width=True)
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


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🏏 PaceAI Biomechanics")
    st.caption("AI Motion Capture, Kinetics & Injury Prevention")
    st.markdown("---")

    page = st.radio("🧭 Navigation", ["⚡ Analyze", "📚 History & Compare"],
                    help="Analyze: run a new delivery. History: browse saved results and compare.")

    if page == "⚡ Analyze":
        input_mode = st.radio("📥 Analysis Mode", ["🎛️ Interactive Bio-Simulator", "📹 Video Motion Capture"], index=0)
    else:
        input_mode = "🎛️ Interactive Bio-Simulator"

    st.markdown("#### ⚙️ Model & Bowling Setup")
    bowling_arm = st.selectbox("Bowling Arm", ["Right-Arm", "Left-Arm"])
    model_choice = st.selectbox(
        "AI Prediction Backbone",
        ["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"],
        help="Selects the ML architecture for kinetic scoring. XGBoost/CatBoost/CNN-LSTM/Transformer "
             "fall back to a scikit-learn equivalent automatically if their library isn't installed."
    )

    # Video-only processing controls (kept for CV speed/accuracy tuning)
    camera_view = "behind"
    target_fps, resize_choice, denoise = 20, (640, 360), False
    with st.expander("🎥 Video Processing (CV speed/accuracy)"):
        camera_view = st.selectbox(
            "Camera view", ["behind", "side"],
            help="Recording orientation. 'Behind' (rear of the bowler) is assumed by the "
                 "2D fallback features; 'side' is supported for world-landmark metrics.")
        processing_preset = st.selectbox(
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
        target_fps = st.slider("Target FPS", 5, 30, target_fps, step=5,
                               help="Lower = fewer frames for pose estimation = faster.")
        resize_choice = st.selectbox(
            "Frame resolution",
            [(640, 360), (960, 540), (1280, 720)],
            format_func=lambda d: f"{d[0]}×{d[1]}",
            index=[(640, 360), (960, 540), (1280, 720)].index(resize_choice))
        denoise = st.checkbox("Denoise frames", value=denoise,
                              help="On = more accurate on noisy footage but much slower.")

    st.markdown("---")
    st.markdown("#### 🔍 System Status")
    st.caption(f"⚡ XGBoost: **{'Active' if ml_models.BACKEND_INFO['xgboost_available'] else 'Scikit Fallback'}**")
    st.caption(f"🐱 CatBoost: **{'Active' if ml_models.BACKEND_INFO['catboost_available'] else 'Scikit Fallback'}**")
    st.caption(f"🧠 PyTorch: **{'Active' if ml_models.BACKEND_INFO['torch_available'] else 'Disabled'}**")
    st.caption(f"🔬 SHAP Engine: **{'Active' if explainability.SHAP_AVAILABLE else 'Finite Diff'}**")
    st.caption(f"📚 History entries: **{history_db.count()}**")

# ---------------- PAGE DISPATCH ----------------
if page == "📚 History & Compare":
    render_history_page()
    st.stop()

perf_bundle, injury_bundle = load_or_train_models(model_choice)

# ---------------- HERO BANNER ----------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">⚡ Fast-Bowling Biomechanics AI</h1>
            <p class="hero-subtitle">Kinematic Chain Profiling • ICC Arm Legality Check • Lumbar & Knee Injury Prevention</p>
        </div>
        <div>
            <span class="status-badge badge-legal" style="margin-right: 8px;">● AI Engine Ready</span>
            <span class="status-badge badge-low">● YOLOv11 + MediaPipe</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


feature_vector = {}
stage_times = {}

# ---------------- INPUT SECTION ----------------
if input_mode == "🎛️ Interactive Bio-Simulator":
    st.markdown("### 1. Delivery Kinematic Parameters")
    
    # Preset Selector
    selected_preset = st.selectbox("⚡ Quick Load Action Preset:", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset]

    with st.expander("🛠️ Fine-Tune Biomechanical Sliders (Release & Impact Points)", expanded=True):
        cols = st.columns(2)
        for i, (feat, (label, unit, lo, hi, default_val)) in enumerate(FEATURE_LABELS.items()):
            active_val = preset_data[feat] if preset_data else default_val
            with cols[i % 2]:
                feature_vector[feat] = st.slider(
                    f"{label} ({unit})",
                    float(lo), float(hi), float(active_val),
                    help=f"Acceptable range: {lo} - {hi} {unit}"
                )

else:
    st.markdown("### 1. Upload Bowling Video")
    st.info("Runs Computer Vision pipeline: Detection (YOLOv11) ➔ Multi-object tracking (ByteTrack) ➔ 3D Pose (MediaPipe) ➔ Biomechanical extraction.")
    uploaded = st.file_uploader("Upload bowling delivery video clip", type=["mp4", "mov", "avi"])

    if uploaded is not None:
        upload_t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
        upload_time = time.perf_counter() - upload_t0

        if not os.path.exists(config.POSE_MODEL_PATH):
            st.warning("MediaPipe pose task model missing. Run pose downloader or manual entry.")
        else:
            loader = st.empty()
            with loader.container():
                render_loader(message=f"Analyzing '{uploaded.name}'...",
                              fps=str(target_fps),
                              knee="ANALYZING",
                              risk="CALC...")
            try:
                result = pipeline.analyze_video(
                    video_path,
                    bowling_arm=bowling_arm.lower().split("-")[0],
                    performance_bundle=perf_bundle,
                    injury_bundle=injury_bundle,
                    target_fps=target_fps,
                    resize_dim=resize_choice,
                    denoise=denoise,
                    camera_view=camera_view,
                )
                loader.empty()
                feature_vector = result.feature_vector
                st.session_state["video_stage_times"] = dict(result.stage_times or {})
                st.session_state["video_upload_time"] = upload_time
                st.session_state["last_warnings"] = list(result.warnings or [])
                st.success("✅ Delivery motion capture completed successfully!")
            except Exception as e:
                loader.empty()
                st.error(f"Pipeline error: {e}")


# ---------------- ANALYSIS & VISUALIZATION ----------------
if feature_vector:
    result = pipeline.analyze_feature_vector(feature_vector, perf_bundle, injury_bundle)
    risk = result.injury_risk or {}
    risk_level = str(risk.get("risk_level", "low")).lower()
    risk_score = {"low": 22, "moderate": 58, "high": 88}.get(risk_level, 22)
    risk_probs = risk.get("probabilities") or []

    # Merge video pipeline timings (if this run came from a video)
    if input_mode.startswith("📹") and st.session_state.get("video_stage_times"):
        stage_times = dict(st.session_state["video_stage_times"])
        stage_times.update(dict(result.stage_times or {}))
        upload = st.session_state.get("video_upload_time")
        if upload:
            stage_times["upload"] = upload
    else:
        stage_times = dict(result.stage_times or {})

    elbow_flex = feature_vector.get("elbow_flexion_deg", 10.0)
    is_icc_legal = elbow_flex <= 15.0

    st.markdown("---")

    for warning in st.session_state.get("last_warnings", []):
        st.warning(warning)

    # Key Summary Metric Header
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">PERFORMANCE RATING</span>
            <h2 style="margin:4px 0; color:#00e676;">{result.performance_score:.1f}<span style="font-size:1rem;color:#8b949e"> / 100</span></h2>
            <span style="color:#8b949e; font-size:0.78rem;">Pace Potential Index</span>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        badge_cls = f"badge-{risk_level}"
        p_high = f"{risk_probs[2]:.2f}" if len(risk_probs) > 2 else "n/a"
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">INJURY RISK LEVEL</span>
            <div style="margin:8px 0;"><span class="status-badge {badge_cls}">{risk_level.upper()} RISK</span></div>
            <span style="color:#8b949e; font-size:0.78rem;">P(High Risk) = {p_high}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        icc_badge = "badge-legal" if is_icc_legal else "badge-illegal"
        icc_text = "LEGAL (≤15°)" if is_icc_legal else "SUSPECT (>15°)"
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">ICC ACTION LEGALITY</span>
            <div style="margin:8px 0;"><span class="status-badge {icc_badge}">{icc_text}</span></div>
            <span style="color:#8b949e; font-size:0.78rem;">Flexion: <b>{elbow_flex:.1f}°</b></span>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">FRONT KNEE BRACE</span>
            <h2 style="margin:4px 0; color:#29b6f6;">{feature_vector.get('knee_flexion_deg', 0):.1f}°</h2>
            <span style="color:#8b949e; font-size:0.78rem;">Ideal: &lt; 15° for lever efficiency</span>
        </div>
        """, unsafe_allow_html=True)

    render_ood_warnings(feature_vector, perf_bundle)

    # ---------------- TABBED DETAILED BREAKDOWN ----------------
    tab_summary, tab_radar, tab_shap, tab_coaching, tab_clinical, tab_export = st.tabs([
        "📊 Gauges & Joint Stress",
        "🕸️ Kinetic Radar vs Pro Benchmark",
        "🧠 Explainable AI (SHAP)",
        "🏋️ Coaching & Rehab Drills",
        "🏥 Clinical Risk",
        "📑 Biomechanical Report"
    ])

    with tab_summary:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(
                render_modern_gauge(result.performance_score, "Performance & Pace Potential", "Kinematic Energy Transfer Score"),
                use_container_width=True
            )
            interval = ml_models.prediction_interval_performance(perf_bundle, feature_vector)
            if interval:
                st.caption(f"68% prediction interval: **{interval[0]:.0f}–{interval[1]:.0f}** "
                           f"(model uncertainty)")
        with col_g2:
            st.plotly_chart(
                render_modern_gauge(risk_score, "Injury Hazard Index", f"Overall Risk: {risk_level.upper()}", is_risk=True),
                use_container_width=True
            )
            if len(risk_probs) >= 3:
                st.caption(f"P(low)={risk_probs[0]:.2f}  P(moderate)={risk_probs[1]:.2f}  "
                           f"P(high)={risk_probs[2]:.2f}")

        st.markdown("#### 🦴 Joint & Segment Kinetic Stress Levels")
        # Estimate stress indexes based on biomechanics
        trunk_stress = min(100, int((feature_vector['trunk_lean_deg'] / 50.0) * 100))
        knee_stress = min(100, int((feature_vector['knee_flexion_deg'] / 40.0) * 100))
        shoulder_stress = min(100, int((feature_vector['angular_velocity_deg_s'] / 1200.0) * 100))

        stress_cols = st.columns(3)
        with stress_cols[0]:
            st.markdown(f"**Lumbar Spine Lateral Shear**: `{trunk_stress}%`")
            st.progress(trunk_stress / 100.0)
        with stress_cols[1]:
            st.markdown(f"**Front Knee Impact Load**: `{knee_stress}%`")
            st.progress(knee_stress / 100.0)
        with stress_cols[2]:
            st.markdown(f"**Rotator Cuff Dynamic Strain**: `{shoulder_stress}%`")
            st.progress(shoulder_stress / 100.0)

    with tab_radar:
        st.markdown("#### 🕸️ Biomechanical Signature vs Elite Fast Bowlers")
        st.caption("A wider, balanced polygon indicates closer alignment with ideal aerodynamic and kinematic levers.")
        st.plotly_chart(render_radar_comparison(feature_vector), use_container_width=True)

    with tab_shap:
        st.markdown("#### 🧠 Model Explainability Breakdown")
        st.caption("Identifies which exact kinematic variables pushed performance up or triggered injury alerts.")
        shap_c1, shap_c2 = st.columns(2)
        with shap_c1:
            if result.shap_contributions_performance:
                st.plotly_chart(
                    render_shap_bar(result.shap_contributions_performance, "Performance Contributors (Blue = Positive, Red = Drag)"),
                    use_container_width=True
                )
        with shap_c2:
            if result.shap_contributions_injury:
                st.plotly_chart(
                    render_shap_bar(result.shap_contributions_injury, "Injury Hazard Risk Drivers (Red = Elevates Risk)"),
                    use_container_width=True
                )

    with tab_coaching:
        st.markdown("### 🏋️ AI Coaching & Prescriptive Drills")
        
        # Categorized recommendations
        if result.coaching_notes:
            for note in result.coaching_notes:
                st.markdown(f"""
                <div class="drill-card">
                    <b>🎯 Biomechanics Note:</b> {note}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Action mechanics are well within optimal ranges.")

        st.markdown("#### 📋 Recommended Corrective Exercise Protocols")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("""
            **1. Front Knee Block Stability (Brace Reinforcement)**
            - *Drill:* Single-leg box deceleration landings + resistance band knee blocks.
            - *Target:* Prevent collapse of the front knee at Front Foot Contact (FFC).
            """)
        with d2:
            st.markdown("""
            **2. Anti-Lateral Flexion Core Bracing**
            - *Drill:* Half-kneeling Pallof presses & suitcase carries.
            - *Target:* Minimize excessive trunk lateral flexion to prevent L4/L5 lumbar stress fractures.
            """)

    with tab_clinical:
        st.markdown("### 🏥 Clinical Injury-Risk Benchmarks")
        st.caption("Literature-derived trigger thresholds (data/cricket_injury_recovery_benchmarks.json) "
                   "evaluated against this delivery. Screening reference only -- not a medical diagnosis.")

        clinical_feats = injury_kb.map_from_pipeline_features(feature_vector)
        risks = injury_kb.assess_biomechanical_risks(clinical_feats)

        if risks:
            for r in risks:
                badge_cls = "badge-high" if r["severity"] == "High" else "badge-moderate"
                trig_text = "".join(f"<li>{t}</li>" for t in r["trigger_detected"])
                st.markdown(f"""
                <div class="drill-card">
                    <b>🩺 {r['injury']}</b>
                    &nbsp;<span class="status-badge {badge_cls}">{r['severity'].upper()} RISK</span>
                    <div style="margin-top:6px; color:#8b949e; font-size:0.85rem;">
                        <b>Site:</b> {r['anatomical_site']} &nbsp;•&nbsp;
                        <b>Incidence:</b> {r['clinical_incidence']}
                    </div>
                    <div style="margin-top:4px; font-size:0.9rem;">
                        <b>Trigger:</b>
                        <ul style="margin:4px 0 4px 18px; color:#e6edf3;">{trig_text}</ul>
                    </div>
                    <div style="color:#8b949e; font-size:0.85rem;">
                        <b>Recovery:</b> {r['est_recovery_timeline']}
                        (median <b style="color:#ef5350">{r['median_days_to_match']}</b> days to match fitness)
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No clinical benchmark trigger thresholds exceeded for this delivery.")

        st.markdown("#### 🧬 Workload & ACWR (optional inputs)")
        w1, w2, w3 = st.columns(3)
        with w1:
            acwr = st.number_input("ACWR (acute:chronic workload ratio)", 0.0, 3.0, 1.0, 0.05,
                                   help="Sweet spot 0.80-1.30; >1.50 = 2.5x-3.3x injury likelihood.")
        with w2:
            seven_day_load = st.number_input("7-day bowling load (balls)", 0, 2000, 180, 1,
                                             help=">234 balls in 7 days ≈ 11x lumbar stress-fracture risk vs <197.")
        with w3:
            rest_days = st.number_input("Rest days between spells", 0, 14, 3, 1,
                                        help="<2 rest days between spells = 2.4x higher injury rate.")
        for check in injury_kb.workload_risk(acwr=acwr, seven_day_load=seven_day_load, rest_days=rest_days):
            icon = {"at_risk": "🚨", "warning": "⚠️", "ok": "✅"}.get(check["status"], "•")
            st.markdown(f"{icon} **{check['check']}** — {check['detail']}")

        with st.expander("Full clinical benchmark table"):
            bench_df = pd.DataFrame(injury_kb.all_benchmarks())
            bench_df = bench_df.rename(columns={
                "injury": "Injury", "anatomical_site": "Anatomical Site",
                "clinical_incidence": "Clinical Incidence",
                "primary_triggers": "Primary Triggers",
                "avg_days_to_return": "Avg Days to Return",
                "median_days_to_match": "Median Days to Match",
                "recovery_window": "Recovery Window",
            })
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

    with tab_export:
        st.markdown("### 📑 Biomechanical Delivery Report")
        feat_df = pd.DataFrame([
            {
                "Kinematic Feature": FEATURE_LABELS.get(k, (k,))[0],
                "Measured Value": f"{v:.2f} {FEATURE_LABELS.get(k, ('', ''))[1]}",
                "Benchmark Range": f"{FEATURE_LABELS.get(k, ('','',0,0,0))[2]} - {FEATURE_LABELS.get(k, ('','',0,0,0))[3]} {FEATURE_LABELS.get(k, ('', ''))[1]}"
            }
            for k, v in feature_vector.items()
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        report_json = json.dumps({
            "performance_score": result.performance_score,
            "injury_risk": result.injury_risk,
            "icc_legal": is_icc_legal,
            "kinematics": feature_vector,
            "coaching_feedback": result.coaching_notes
        }, indent=2)

        st.download_button(
            label="📥 Download Full Delivery Analysis (JSON)",
            data=report_json,
            file_name="bowling_biomechanics_report.json",
            mime="application/json"
        )

    # ---------------- MODEL QUALITY HONESTY PANEL ----------------
    render_model_quality_expander(perf_bundle, injury_bundle)

    # ---------------- RUN TIMING ----------------
    if stage_times:
        st.markdown("### ⏱️ Run Timing")
        render_timings(stage_times)

    # ---------------- SAVE TO HISTORY ----------------
    st.markdown("### 💾 Save to History")
    st.caption("Persist this delivery's results to the local history database so you can "
               "compare it against future sessions and track your performance over time.")
    save_label = st.text_input("Label (optional)", placeholder="e.g. Net session 1, match 3 over 4",
                               key="save_label")
    if st.button("💾 Save this result", type="primary"):
        saved_id = history_db.save_analysis(
            result, label=save_label, input_mode=input_mode,
            bowling_arm=bowling_arm.lower().split("-")[0], model=model_choice)
        st.success(f"Saved to history (id #{saved_id}). Open **History & Compare** in the sidebar "
                   f"to view and compare your saved results.")

else:
    st.info("Enter features manually or upload a video to run the analysis.")

st.markdown("---")
st.caption(
    "⚡ **PaceAI Biomechanics Engine** • Demo models are trained on synthetic data "
    "(src/synthetic_data.py) -- retrain on real labeled data (`python train_demo_model.py "
    "--data your_dataset.csv`) before using for real coaching/medical decisions. "
    "Designed for elite high-performance cricket centers, coaches, and sports physiotherapists."
)
