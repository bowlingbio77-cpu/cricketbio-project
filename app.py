"""
PaceAI - Cricket Bowling Biomechanics AI
Pro Coaching & Biomechanics Screening Dashboard (dark theme)

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
import html as _html_mod
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import importlib


class _LazyModule:
    """Defer a heavy module's import until its first attribute is accessed.

    Used for the GPU/CV/ML stack (ml_models, explainability, pipeline) so the
    dashboard's first paint doesn't wait for torch/xgboost/catboost/mediapipe.
    """
    def __init__(self, name: str):
        self._name = name
        self._mod = None

    def _get(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return self._mod

    def __getattr__(self, item):
        return getattr(self._get(), item)


from src import config, history_db, injury_knowledge_base as injury_kb
from src import analysis_ui
from src import kinetic_ui
from src import result_view
from src.synthetic_data import generate_clinical_synthetic_dataset

# Heavy modules loaded lazily (only on first use), see _LazyModule above.
ml_models = _LazyModule("src.ml_models")
explainability = _LazyModule("src.explainability")
pipeline = _LazyModule("src.pipeline")
from src.auth_login import render_login_page, is_authenticated
from chat_assistant import render_chat_widget


def _esc(text) -> str:
    """HTML-escape a value for safe injection into an HTML template.

    Every ``unsafe_allow_html=True`` call that interpolates user-controlled
    or model-output strings (feature values, coaching notes, athlete names,
    risk levels) MUST pass through this helper first.  Static / trusted HTML
    (CSS blocks, hero banners with no dynamic content) does NOT need escaping.
    """
    if text is None:
        return ""
    return _html_mod.escape(str(text))

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="PaceAI | Cricket Bowling Biomechanics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- AUTH GATE (disabled for now) ----------------
# if not is_authenticated():
#     render_login_page()
#     st.stop()

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- Skip to content (visible on focus for keyboard nav) --- */
    .skip-link {
        position: absolute; left: -9999px; top: auto;
        width: 1px; height: 1px; overflow: hidden;
        z-index: 999999; padding: 12px 20px; margin: 8px;
        background: #2b3442; color: #a9cdec; font-weight: 700;
        border-radius: 8px; text-decoration: none; font-size: 0.95rem;
        border: 1px solid #63d4cf;
    }
    .skip-link:focus {
        position: fixed; left: 12px; top: 12px;
        width: auto; height: auto; overflow: visible;
    }

    /* Main background & headers */
    .stApp {
        background-color: #212833;
        color: #e9eef5;
    }

    /* Card Containers */
    .metric-card {
        background: linear-gradient(145deg, #2e3949 0%, #232b38 100%);
        border: 1px solid #3d4859;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
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
    .badge-low { background-color: rgba(67, 217, 163, 0.14); color: #5fe0b0; border: 1px solid #3cb98c; }
    .badge-moderate { background-color: rgba(232, 179, 74, 0.14); color: #f0c169; border: 1px solid #c99c42; }
    .badge-high { background-color: rgba(255, 112, 134, 0.14); color: #ff8699; border: 1px solid #e05a6f; }
    .badge-legal { background-color: rgba(67, 217, 163, 0.16); color: #5fe0b0; border: 1px solid #3cb98c; }
    .badge-illegal { background-color: rgba(255, 112, 134, 0.16); color: #ff8699; border: 1px solid #e05a6f; }
    .badge-demo { background-color: rgba(232, 179, 74, 0.16); color: #f0c169; border: 1px solid #c99c42; }
    .badge-video { background-color: rgba(142, 193, 238, 0.14); color: #9cc8ee; border: 1px solid #5f93c8; }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(90deg, #2e394b 0%, #27313f 50%, #222c3a 100%);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 25px;
        border: 1px solid #3d4859;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #a9cdec;
        margin: 0;
    }
    .hero-subtitle {
        color: #aeb9c8;
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Drill card */
    .drill-card {
        border-left: 4px solid #63d4cf;
        background: #2b3442;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        border-top: 1px solid #3d4859;
        border-right: 1px solid #3d4859;
        border-bottom: 1px solid #3d4859;
    }

    /* Feature guide tag chips */
    .guide-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: rgba(142, 193, 238, 0.16);
        border: 1px solid #4b586c;
        color: #9cc8ee;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        vertical-align: middle;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #232b38;
        border-right: 1px solid #3d4859;
    }

    /* --- Accessibility --- */
    .sr-only {
        position: absolute; width: 1px; height: 1px;
        padding: 0; margin: -1px; overflow: hidden;
        clip: rect(0,0,0,0); white-space: nowrap; border: 0;
    }
    :focus-visible {
        outline: 2px solid #63d4cf;
        outline-offset: 2px;
    }
    /* Ensure focus is visible on Streamlit widgets */
    .stSelectbox:focus-within,
    .stSlider:focus-within,
    .stRadio:focus-within,
    .stButton:focus-within,
    .stTextInput:focus-within,
    .stNumberInput:focus-within {
        box-shadow: 0 0 0 2px rgba(142, 193, 238, 0.35);
        border-radius: 6px;
    }
    /* Status badges include text labels alongside color for non-color-dependent info */
    .status-badge::before {
        content: none; /* text is already inside the badge element */
    }
    /* Improved contrast for secondary text (WCAG AA: >=4.5:1 on #212833) */
    .hero-subtitle, .metric-card span[style*="8b949e"] {
        color: #aeb9c8 !important;
    }

    /* --- Responsive: Tablet (768px) --- */
    @media (max-width: 900px) {
        .hero-title { font-size: 1.5rem; }
        .hero-subtitle { font-size: 0.85rem; }
        .hero-banner { padding: 16px 18px; }
        .metric-card { padding: 14px; }
    }

    /* --- Responsive: Mobile (480px) --- */
    @media (max-width: 600px) {
        .hero-title { font-size: 1.2rem; }
        .hero-subtitle { font-size: 0.8rem; }
        .hero-banner { padding: 12px 14px; border-radius: 10px; }
        .metric-card { padding: 12px; font-size: 0.85rem; }
        .drill-card { padding: 10px 12px; font-size: 0.85rem; }
        .guide-tag { font-size: 0.6rem; padding: 1px 6px; }
        /* Stack status badges vertically on very small screens */
        .hero-banner span.status-badge { display: block; margin: 4px 0; }
    }

    /* --- Premium Analysis Replay --- */
    .analysis-replay-shell {
        background: linear-gradient(145deg, #2e3949 0%, #222b39 100%);
        border: 1px solid #3d4859; border-bottom: 0;
        border-radius: 16px 16px 0 0; padding: 20px 22px 10px;
        margin-top: 20px;
    }
    .analysis-replay-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    .analysis-replay-head h2 { margin:4px 0 3px; font-size:1.65rem; color:#a9cdec; }
    .analysis-replay-head p { margin:0; color:#7e8b9d; font-size:.9rem; }
    .eyebrow,.priority-kicker { color:#63d4cf; font-size:.68rem; letter-spacing:.14em; font-weight:800; }
    .replay-pill { border:1px solid rgba(142,193,238,.35); color:#9cc8ee; background:rgba(142,193,238,.12); border-radius:999px; padding:7px 10px; font-size:.68rem; font-weight:800; white-space:nowrap; }
    .priority-card { margin:18px 0; padding:18px 20px; border:1px solid rgba(67,217,163,.35); background:linear-gradient(145deg, rgba(67,217,163,.08), #2b3442); border-radius:14px; }
    .priority-title { font-size:1.15rem; font-weight:800; margin:4px 0 7px; color:#a9cdec; }
    .priority-copy { color:#aeb9c8; line-height:1.55; }
    .analysis-empty { margin:20px 0; padding:30px; border:1px dashed #4b586c; border-radius:16px; background:#2b3442; }
    .analysis-empty-kicker { color:#7e8b9d; font-size:.7rem; letter-spacing:.14em; font-weight:800; }
    .analysis-empty-title { font-size:1.3rem; font-weight:800; margin:5px 0; color:#a9cdec; }
    .analysis-empty-copy { color:#7e8b9d; }

    /* --- Streamlit column stacking on narrow viewports --- */
    @media (max-width: 768px) {
        /* Force Streamlit columns to stack on tablet/mobile */
        [data-testid="stHorizontalBlock"] > div {
            flex-basis: 100% !important;
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Accessibility: skip-to-content link + lang attribute ---
st.markdown(
    '<a href="#main-content" class="skip-link">Skip to main content</a>'
    '<script>document.documentElement.lang="en";</script>',
    unsafe_allow_html=True,
)


# ---------------- LOADING OVERLAY ----------------
# Pure-CSS PaceAI preloader (no JS needed): orbiting spinner -> success check
# -> self-fading overlay. Used both for the boot splash and inline loading.
_PRELOADER_CSS = """
<style>
    @keyframes paceaiFadeOut { to { opacity:0; visibility:hidden; pointer-events:none; } }
    @keyframes paceaiPulse { 0%,100% { opacity:.75; transform:scale(1); } 50% { opacity:1; transform:scale(1.06); } }
    @keyframes paceaiOrbit { to { transform:rotate(360deg); } }
    @keyframes paceaiHide { to { opacity:0; visibility:hidden; } }
    @keyframes paceaiSuccessIn { 0% { opacity:0; transform:scale(.6); } 40% { opacity:1; transform:scale(1.15); } 100% { opacity:1; transform:scale(1); } }
    @keyframes paceaiSuccessGlow { 0% { opacity:0; transform:scale(.6); } 40% { opacity:1; transform:scale(1.15); } 100% { opacity:0; transform:scale(1.4); } }
    @keyframes paceaiDrawCheck { to { stroke-dashoffset:0; } }

    .pace-preloader { position:fixed; inset:0; z-index:99999; background:#212833;
        display:flex; align-items:center; justify-content:center; overflow:hidden;
        font-family:'Segoe UI',Arial,sans-serif;
        animation:paceaiFadeOut .6s ease 3.3s forwards; opacity:1; }
    .pace-preloader .glow { position:absolute; width:min(420px,80vw); height:min(420px,80vw); border-radius:50%;
        background:radial-gradient(circle, rgba(142,193,238,.16) 0%, rgba(99,212,207,.10) 45%, rgba(0,0,0,0) 72%);
        animation:paceaiPulse 4.5s ease-in-out infinite; }
    .pace-preloader .stage { position:relative; width:180px; height:180px; display:flex; align-items:center; justify-content:center; }
    .pace-preloader .orbit-ring { position:absolute; width:132px; height:132px; border-radius:50%; border:1px solid rgba(142,193,238,.25); }
    .pace-preloader .orbit-spin { position:absolute; width:132px; height:132px; will-change:transform;
        animation:paceaiOrbit 2.6s linear infinite, paceaiHide .35s ease 2.5s forwards; }
    .pace-preloader .orbit-dot { position:absolute; top:-4px; left:50%; margin-left:-4px; width:8px; height:8px; border-radius:50%;
        background:#8ec1ee; box-shadow:0 0 6px 2px rgba(142,193,238,.55), 0 0 16px 6px rgba(99,212,207,.25); }
    .pace-preloader .logo-badge { position:relative; width:64px; height:64px; border-radius:50%;
        background:linear-gradient(145deg,#2e3949 0%,#232b38 100%); border:1px solid #3d4859;
        display:flex; align-items:center; justify-content:center; font-size:26px; box-shadow:0 4px 24px rgba(0,0,0,.35);
        animation:paceaiHide .35s ease 2.5s forwards; }
    .pace-preloader .loading-text { position:absolute; bottom:-64px; left:50%; transform:translateX(-50%);
        color:#7e8b9d; font-size:13px; letter-spacing:1.5px; text-transform:uppercase;
        white-space:nowrap; text-align:center; max-width:80vw; }
    .pace-preloader .success { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        opacity:0; visibility:hidden; animation:paceaiSuccessIn .35s ease 2.5s forwards; }
    .pace-preloader .success .sglow { position:absolute; width:110px; height:110px; border-radius:50%;
        background:radial-gradient(circle, rgba(67,217,163,.30) 0%, rgba(67,217,163,0) 70%);
        animation:paceaiSuccessGlow .6s ease-out 2.5s both; }
    .pace-preloader .check-badge { width:64px; height:64px; border-radius:50%;
        background:linear-gradient(145deg,#2e3949 0%,#232b38 100%); border:1px solid rgba(67,217,163,.6);
        display:flex; align-items:center; justify-content:center; box-shadow:0 4px 24px rgba(0,0,0,.35); }
    .pace-preloader .checkmark-path { stroke-dasharray:28; stroke-dashoffset:28; animation:paceaiDrawCheck .32s ease-out 2.56s forwards; }

    @media (prefers-reduced-motion: reduce) {
        .pace-preloader .glow, .pace-preloader .orbit-spin { animation:none; }
        .pace-preloader .glow { opacity:.85; }
        .pace-preloader .success { animation:paceaiHide 0s linear 2.5s forwards; }
        .pace-preloader .checkmark-path { stroke-dashoffset:0; }
    }
</style>
"""


def _paceai_preloader(message: str) -> str:
    return f"""
<div class="pace-preloader" role="status" aria-live="polite" aria-label="{message}">
    <div class="glow"></div>
    <div class="stage">
        <div class="orbit-ring"></div>
        <div class="orbit-spin"><div class="orbit-dot"></div></div>
        <div class="logo-badge">&#9889;</div>
        <div class="loading-text">{message}</div>
        <div class="success">
            <div class="sglow"></div>
            <div class="check-badge">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                    <path class="checkmark-path" d="M4 12.5 L9.5 18 L20 5.5"
                          stroke="#43d9a3" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
    </div>
</div>
"""


def render_loader(message: str = "Loading PaceAI..."):
    """Inline PaceAI preloader (used while models train on first run)."""
    components.html(_PRELOADER_CSS + _paceai_preloader(message), height=560, scrolling=False)


def render_fullscreen_splash(message: str = "Loading PaceAI..."):
    """Full-viewport PaceAI preloader shown during app/model startup (self-fading)."""
    components.html(_PRELOADER_CSS + _paceai_preloader(message), height=560, scrolling=False)


# ---------------- HELPERS ----------------
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


class _LiveAnalysisScreen:
    """Streamlit wrapper around the premium live-analysis HTML view.

    Re-renders the analysis screen into a single placeholder on every stage
    tick so Streamlit updates live without a full app rerun. Presentation only
    -- all state comes from `analysis_ui.AnalysisState`, which the pipeline's
    progress callback and the returned AnalysisResult populate.
    """

    def __init__(self):
        self._ph = st.empty()

    def render(self, state: analysis_ui.AnalysisState, show_art: bool = True):
        self._ph.markdown(analysis_ui.render_lab_html(state, show_art=show_art),
                          unsafe_allow_html=True)

    def clear(self):
        self._ph.empty()


def _live_state(phase="running", **kw) -> analysis_ui.AnalysisState:
    return analysis_ui.AnalysisState(phase=phase, **kw)


def _clear_video_run():
    """Reset the processed-file token so a (re)analysis of the same upload runs again."""
    st.session_state["video_processed_file_id"] = None


def _run_video_analysis(uploaded, perf_bundle, injury_bundle, bowling_arm,
                        target_fps, resize_choice, denoise, camera_view,
                        slow_factor, zoom_end, debug_overlay, screen) -> dict:
    """Run the full CV pipeline on an uploaded file ONCE.

    Responsibility: write the upload to a temp file, run pipeline.analyze_video
    (with the live-analysis screen driving progress), store all results in
    session_state, and render the completion card. Returns the AnalysisState
    used for the completion screen (or raises).

    Guarded by the caller on ``uploaded.file_id`` so a Streamlit re-run does
    NOT re-process the same file -- results already persist in session_state.
    """
    upload_t0 = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(uploaded.read())
        video_path = tmp.name
    finally:
        tmp.close()
    upload_time = time.perf_counter() - upload_t0

    try:
        if not os.path.exists(config.POSE_MODEL_PATH):
            st.warning("MediaPipe pose task model missing. Run pose downloader or manual entry.")
            return None

        screen.render(_live_state(
            phase="running", fps=target_fps, resize=tuple(resize_choice),
        ), show_art=True)

        def _stage_cb(done, total, label):
            screen.render(_live_state(
                phase="running", done=done, total=total,
                current_label=label, fps=target_fps,
                resize=tuple(resize_choice),
            ), show_art=True)

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
                slow_factor=slow_factor,
                zoom_end=zoom_end,
                run_ml=False,
                progress_cb=_stage_cb,
                debug_overlay=debug_overlay,
            )
        finally:
            try:
                os.remove(video_path)
            except OSError:
                pass

        st.session_state["video_stage_times"] = dict(result.stage_times or {})
        st.session_state["video_feature_vector"] = dict(result.feature_vector or {})
        st.session_state["video_upload_time"] = upload_time
        st.session_state["last_warnings"] = list(result.warnings or [])
        st.session_state["video_subject_verified"] = getattr(result, "subject_verified", True)
        st.session_state["video_scoring_blocked_reason"] = getattr(
            result, "scoring_blocked_reason", None)
        st.session_state["video_delivery_reliable"] = getattr(result, "delivery_reliable", True)
        st.session_state["video_output_path"] = getattr(result, "video_path", None)
        st.session_state["pose_video_path"] = getattr(result, "pose_video_path", None)
        st.session_state["reels_video_path"] = getattr(result, "reels_video_path", None)
        st.session_state["analysis_replay_path"] = getattr(result, "analysis_replay_path", None)
        st.session_state["analysis_key_moments"] = getattr(result, "key_moments", None) or getattr(result, "events", None) or []
        st.session_state["ball_stats"] = getattr(result, "ball_stats", {})
        st.session_state["video_player_roles"] = getattr(result, "player_roles", None)

        balls = result.ball_stats or {}
        screen.render(_live_state(
            phase="complete",
            done=len([1 for g in analysis_ui.GROUPS for _ in g.steps]),
            total=len([1 for g in analysis_ui.GROUPS for _ in g.steps]),
            current_label="Complete",
            fps=target_fps,
            resize=tuple(resize_choice),
            total_frames=balls.get("total_frames"),
            original_dims=tuple(result.original_frame_dims)
            if result.original_frame_dims else None,
            bowler_track_id=result.bowler_track_id,
            bowler_confidence=result.bowler_confidence,
            elapsed_s=(result.stage_times or {}).get("total"),
        ), show_art=False)
        st.button("VIEW ANALYSIS →",
                  on_click=lambda: st.session_state.update(scroll_to_replay=True),
                  type="primary")
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err = f"{type(e).__name__}: {e}"
        screen.render(_live_state(
            phase="error", error=err, fps=target_fps,
            resize=tuple(resize_choice),
            done=0,
            total=len([1 for g in analysis_ui.GROUPS for _ in g.steps]),
        ), show_art=True)
        st.button("TRY AGAIN", on_click=_clear_video_run)
        st.markdown("#### 🔧 TECHNICAL DETAILS")
        st.caption("The analysis was interrupted. The technical reason "
                   "(visible to judges / for debugging) is below.")
        with st.expander("Error details", expanded=False):
            st.code(err + "\n\n" + tb, language="python")
        return None


# ---------------- CACHED MODEL LOADING ----------------
@st.cache_resource
def load_or_train_models(model_name: str = "random_forest"):
    perf_path = os.path.join(config.MODEL_DIR, f"performance_{model_name}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{model_name}.joblib")

    if os.path.exists(perf_path) and os.path.exists(injury_path):
        return ml_models.load_bundle(perf_path), ml_models.load_bundle(injury_path)

    st.info(f"No saved '{model_name}' models found -- training on synthetic demo data now "
            f"(run `python train_demo_model.py` once to cache this).")
    render_loader(message=f"Training {model_name} models on synthetic demo data...")
    df = generate_clinical_synthetic_dataset()
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
    render_fullscreen_splash("Loading...")
    _ = load_or_train_models("random_forest")
    rerun()


FEATURE_LABELS = {
    "shoulder_rotation_deg": ("Shoulder Counter-Rotation", "deg", 0, 90, 18.0),
    "elbow_flexion_deg": ("Elbow Flexion", "deg", 0, 45, 8.0),
    "wrist_angle_deg": ("Wrist Angle", "deg", 90, 180, 165.0),
    "hip_rotation_deg": ("Pelvic Tilt (from horizontal)", "deg", 0, 80, 45.0),
    "knee_flexion_deg": ("Front-Knee Flexion", "deg", 0, 60, 10.0),
    "trunk_lean_deg": ("Trunk Lateral Lean", "deg", 0, 60, 25.0),
    "stride_length_norm": ("Stride Length (norm)", "x H", 0.3, 1.6, 1.05),
    "release_angle_deg": ("Release Angle", "deg", 30, 90, 78.0),
    "angular_velocity_deg_s": ("Shoulder-Rotation Speed", "deg/s", 100, 1500, 1100.0),
    "ground_contact_time_s": ("Front Foot Contact Time", "s", 0.05, 0.35, 0.11),
}

# Elite Fast Bowler Benchmark for comparison (literature-informed profile)
ELITE_BENCHMARK = {
    "shoulder_rotation_deg": 18.0,
    "elbow_flexion_deg": 8.0,
    "wrist_angle_deg": 165.0,
    "hip_rotation_deg": 45.0,
    "knee_flexion_deg": 10.0,
    "trunk_lean_deg": 25.0,
    "stride_length_norm": 1.05,
    "release_angle_deg": 78.0,
    "angular_velocity_deg_s": 1100.0,
    "ground_contact_time_s": 0.11,
}

PRESETS = {
    "Custom / Manual": None,
    "⚡ Elite Fast Bowler (Pro)": {
        "shoulder_rotation_deg": 18.0, "elbow_flexion_deg": 8.0, "wrist_angle_deg": 165.0,
        "hip_rotation_deg": 45.0, "knee_flexion_deg": 10.0, "trunk_lean_deg": 25.0,
        "stride_length_norm": 1.05, "release_angle_deg": 78.0, "angular_velocity_deg_s": 1100.0,
        "ground_contact_time_s": 0.11
    },
    "🚨 High Biomechanical Risk Indicator Action": {
        "shoulder_rotation_deg": 28.0, "elbow_flexion_deg": 22.0, "wrist_angle_deg": 135.0,
        "hip_rotation_deg": 58.0, "knee_flexion_deg": 38.0, "trunk_lean_deg": 45.0,
        "stride_length_norm": 0.72, "release_angle_deg": 60.0, "angular_velocity_deg_s": 460.0,
        "ground_contact_time_s": 0.28
    },
    "🎯 Seam & Swing Specialist": {
        "shoulder_rotation_deg": 25.0, "elbow_flexion_deg": 10.0, "wrist_angle_deg": 170.0,
        "hip_rotation_deg": 38.0, "knee_flexion_deg": 14.0, "trunk_lean_deg": 24.0,
        "stride_length_norm": 0.98, "release_angle_deg": 75.0, "angular_velocity_deg_s": 800.0,
        "ground_contact_time_s": 0.14
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
        bar_color = "#ff7086" if value >= 70 else "#e8b34a" if value >= 40 else "#43d9a3"
    else:
        bar_color = "#43d9a3" if value >= 70 else "#e8b34a" if value >= 50 else "#ff7086"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:#7e8b9d'>{subtitle}</span>"},
        number={"font": {"size": 42, "color": "#8ec1ee"}, "suffix": "%" if is_risk else ""},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#5f6d80", "tickfont": {"color": "#5f6d80"}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#2b3442",
            "borderwidth": 1,
            "bordercolor": "#3d4859",
            "steps": [
                {"range": [0, 40], "color": "rgba(67, 217, 163, 0.14)"},
                {"range": [40, 70], "color": "rgba(232, 179, 74, 0.14)"},
                {"range": [70, max_val], "color": "rgba(67, 217, 163, 0.18)" if not is_risk else "rgba(255, 112, 134, 0.18)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e9eef5"},
        height=240,
        margin=dict(l=25, r=25, t=60, b=20)
    )
    return fig


def render_radar_comparison(current_feats: dict):
    categories = [
        "Shoulder Rot.", "Arm Extension", "Wrist Cocking", "Pelvic Tilt",
        "Knee Brace", "Upright Trunk", "Stride Prowess", "Release Velocity"
    ]

    def normalize(val, lo, hi):
        return max(0, min(100, ((val - lo) / (hi - lo)) * 100))

    user_vals = [
        normalize(60 - current_feats["shoulder_rotation_deg"], 0, 60),  # low counter-rotation = better
        normalize(45 - current_feats["elbow_flexion_deg"], 0, 45),
        normalize(current_feats["wrist_angle_deg"], 90, 180),
        normalize(current_feats["hip_rotation_deg"], 0, 80),
        normalize(60 - current_feats["knee_flexion_deg"], 0, 60),  # Braced = lower flexion
        normalize(60 - current_feats["trunk_lean_deg"], 0, 60),
        normalize(current_feats["stride_length_norm"], 0.3, 1.6),
        normalize(current_feats["angular_velocity_deg_s"], 100, 1500),
    ]

    bench_vals = [
        normalize(60 - ELITE_BENCHMARK["shoulder_rotation_deg"], 0, 60),
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
        line=dict(color='#43d9a3', width=2),
        fillcolor='rgba(67, 217, 163, 0.18)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=bench_vals, theta=categories, fill='toself',
        name='Elite Benchmark (145 km/h)',
        line=dict(color='#8ec1ee', width=2, dash='dot'),
        fillcolor='rgba(142, 193, 238, 0.14)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#5f6d80", gridcolor="#35404f"),
            angularaxis=dict(color="#aeb9c8", gridcolor="#35404f")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e9eef5"),
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
    colors = ["#ff7086" if v > 0 else "#8ec1ee" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color="#8ec1ee", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aeb9c8"),
        xaxis=dict(title="Relative Model Impact (SHAP value)", gridcolor="#35404f", zerolinecolor="#5f6d80"),
        yaxis=dict(gridcolor="#2b3442"),
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
                           marker_color=["#8ec1ee" if k != "total" else "#43d9a3"
                                         for k, _ in ordered]))
    fig.update_layout(title="Stage timing", height=320,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#aeb9c8"),
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Seconds")
    st.plotly_chart(fig, width='stretch')
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
            st.markdown(f"**Biomechanical risk-indicator model** (`{injury_bundle.model_name}`)"
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


# Every feature in the app, explained in plain English (tag, title, description).
FEATURES_GUIDE = [
    ("SIMULATOR", "Bio-Simulator", "Try different bowling actions with sliders -- no video needed."),
    ("VIDEO", "📹 Video Capture", "Upload a clip; the app finds the bowler, reads 33 body landmarks, and measures the delivery automatically."),
    ("BALL", "🎯 Ball Tracking", "The red box follows the cricket ball from release to impact; a dashed box means the app is guessing where it is between detections."),
    ("ARM", "Bowling Arm", "Which arm the bowler bowls with. The app mirrors the joints so left-handers aren't analyzed backwards."),
    ("AI", "🧠 AI Backbone", "The math model that turns measurements into a demonstration performance score and a biomechanical risk-indicator level. Random Forest is the safe default."),
    ("PRESET", "🎥 Processing", "Speed vs accuracy of the video analysis. Fast = rough but quick; Maximum accuracy = precise but slow."),
    ("SCORE", "Performance", "A literature-informed demonstration performance indicator (0--100) for this delivery's mechanics. Higher = closer to published elite pace-bowler ranges."),
    ("RISK", "🚨 Biomechanical Risk Indicator", "How many literature-informed biomechanical trigger thresholds this action crosses. Low = none exceeded, Moderate = a few, High = several."),
    ("LEGALITY", "ICC Screening", "Whether the elbow flexion at release stays within the <=15\u00b0 ICC reference value. Screening only -- an official legality ruling requires lab-grade 3D motion capture per ICC protocol."),
    ("KNEE", "🦵 Knee Brace", "How straight the front knee is at landing. Low degrees = better braking and less knee stress."),
    ("GAUGES", "📊 Gauges & Stress", "Big dials for your score and risk, plus how much load lands on the back, knee and shoulder."),
    ("RADAR", "Kinetic Radar", "Your shape compared with an elite bowler's. A wider, more balanced shape is better."),
    ("XAI", "🧠 Explainable AI", "Which single measurement moved your score or risk up or down the most."),
    ("DRILLS", "Coaching Drills", "Exercises and technique fixes for whatever got flagged in this delivery."),
    ("CLINICAL", "Literature Risk Thresholds", "Checks your delivery against published biomechanical screening benchmarks and workload rules (ACWR, overs, rest days). Screening only -- not a prediction of injury."),
    ("REPORT", "📑 Report", "Downloads all of this run's results as a JSON file you can keep or share."),
    ("HISTORY", "📚 History & Compare", "Every saved delivery, listed and compared side by side over time."),
    ("SAVE", "💾 Save to History", "Stores this run so you can compare it against future sessions."),
    ("TIMING", "Run Timing", "How many seconds each analysis step took -- only useful when tuning for speed."),
]


def render_features_guide():
    """Plain-English, tag-labeled explanation of every feature in the app."""
    st.markdown("#### \U0001f5c2\ufe0f Every feature explained")
    for tag, title, desc in FEATURES_GUIDE:
        st.markdown(
            f"<span class='guide-tag' aria-hidden='true'>{_esc(tag)}</span> &nbsp; <strong>{_esc(title)}</strong> &mdash; {_esc(desc)}",
            unsafe_allow_html=True,
        )


# ---------------- HISTORY & COMPARE PAGE ----------------
def _session_name(row):
    parts = [f"#{row['id']}", row["created_at"]]
    if row.get("athlete"):
        parts.append(f"({row['athlete']})")
    if row.get("label"):
        parts.append(f"— {row['label']}")
    return " ".join(parts)


def _athlete_of(row):
    return row.get("athlete") or "Unnamed bowler"


def _risk_of(row):
    risk = row.get("injury_risk")
    if isinstance(risk, dict):
        return risk.get("risk_level", "—")
    return row.get("risk_level") or "—"


def render_history_page():
    all_records = history_db.load_all()
    st.title("History & Comparison")
    st.caption("Every delivery you saved from the Analyze page lives here -- browse past "
               "results, compare sessions side by side, and track performance over time.")

    if not all_records:
        st.info("No saved results yet. Go to **Analyze**, run a delivery, and press "
                "**Save to History**.")
        return

    athletes = sorted({_athlete_of(r) for r in all_records})
    athlete_choice = st.selectbox("👤 Filter by bowler", ["All bowlers"] + athletes,
                                  key="history_athlete")
    records = [r for r in all_records
               if athlete_choice == "All bowlers" or _athlete_of(r) == athlete_choice]

    st.download_button(
        label="📤 Export all history (JSON)",
        data=json.dumps(all_records, indent=2, default=str),
        file_name="bowling_history_export.json",
        mime="application/json",
        key="history_export",
    )

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
            "Bowler": _athlete_of(r),
            "Label": r.get("label") or "",
            "Tags": r.get("tags") or "",
            "Mode": r.get("input_mode") or "",
            "Arm": r.get("bowling_arm") or "",
            "Model": r.get("model") or "",
            "Performance": round(r["performance_score"], 1) if r.get("performance_score") is not None else None,
            "Risk": _risk_of(r),
        }
        for r in records
    ])
    st.dataframe(table, width='stretch', hide_index=True)

    # --- Performance over time ---
    chrono = sorted([r for r in records if r.get("performance_score") is not None],
                    key=lambda r: r["created_at"])
    if len(chrono) >= 2:
        st.subheader("Performance over time")
        fig = go.Figure()
        if athlete_choice == "All bowlers":
            for athlete in athletes:
                sub = [r for r in chrono if _athlete_of(r) == athlete]
                if len(sub) >= 1:
                    fig.add_trace(go.Scatter(
                        x=[r["created_at"] for r in sub],
                        y=[r["performance_score"] for r in sub],
                        mode="lines+markers",
                        name=athlete,
                    ))
        else:
            fig.add_trace(go.Scatter(
                x=[r["created_at"] for r in chrono],
                y=[r["performance_score"] for r in chrono],
                mode="lines+markers+text",
                text=[f"#{r['id']}" for r in chrono],
                textposition="top center",
                line=dict(color="#43d9a3", width=2),
            ))
        fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#aeb9c8"), margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Date", yaxis_title="Performance score",
                          yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, width='stretch')
    elif chrono:
        st.subheader("Performance over time")
        st.caption("Save more results to see a demo-performance trend chart.")

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
                marker_color="#43d9a3", text=[f"{r.get('performance_score'):.0f}" if r.get('performance_score') is not None else "—" for r in sel],
                textposition="outside"))
            fig.update_layout(title="Performance score", height=320,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#aeb9c8"), margin=dict(l=10, r=10, t=40, b=10),
                              yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = go.Figure(go.Bar(
                x=sel_names, y=[_risk_of(r) for r in sel],
                marker_color=["#ff7086" if _risk_of(r) == "high" else "#e8b34a"
                              if _risk_of(r) == "moderate" else "#43d9a3" for r in sel],
                text=[_risk_of(r).title() for r in sel], textposition="outside"))
            fig.update_layout(title="Biomechanical risk indicator", height=320,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#aeb9c8"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width='stretch')

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
        st.dataframe(pd.DataFrame(feat_rows), width='stretch', hide_index=True)

        if len(selected) >= 2:
            fig = go.Figure()
            for name, r in zip(sel_names, sel):
                x = [FEATURE_LABELS[k][0] for k in config.FEATURE_NAMES]
                y = [r["feature_vector"].get(k) for k in config.FEATURE_NAMES]
                fig.add_trace(go.Bar(x=x, y=y, name=name))
            fig.update_layout(title="Feature values by session", barmode="group",
                              height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#aeb9c8"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width='stretch')

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
                            width='stretch')
        with c2:
            risk = detail.get("injury_risk")
            if isinstance(risk, dict) and risk.get("probabilities"):
                probs = risk["probabilities"]
                risk_num = {"low": 25, "moderate": 60, "high": 90}[risk.get("risk_level", "low")]
                st.plotly_chart(render_modern_gauge(risk_num, f"Biomechanical Risk: {risk['risk_level'].upper()}",
                                                    is_risk=True), width='stretch')
                if len(probs) >= 3:
                    st.caption(f"P(low)={probs[0]:.2f}  P(moderate)={probs[1]:.2f}  P(high)={probs[2]:.2f}")
            else:
                st.info("No biomechanical risk-indicator prediction stored for this session.")
        with st.expander("Features"):
            feat_df = pd.DataFrame([
                {"Feature": FEATURE_LABELS.get(k, (k,))[0],
                 "Value": round(v, 2) if v is not None else None,
                 "Unit": FEATURE_LABELS.get(k, ("", ""))[1]}
                for k, v in detail["feature_vector"].items()
            ])
            st.dataframe(feat_df, width='stretch', hide_index=True)
        with st.expander("Coaching recommendations"):
            notes = detail.get("coaching_notes") or []
            for note in notes:
                st.markdown(f"- {note}")
            if not notes:
                st.caption("No coaching notes stored for this session.")
        with st.expander("Explainable AI — feature contributions"):
            tab1, tab2 = st.tabs(["Performance drivers", "Biomechanical-risk drivers"])
            with tab1:
                shap_perf = detail.get("shap_performance")
                if shap_perf:
                    st.plotly_chart(render_shap_bar(shap_perf,
                                                    "Feature contribution to performance score"),
                                    width='stretch')
                else:
                    st.caption("No performance SHAP data stored for this session.")
            with tab2:
                shap_injury = detail.get("shap_injury")
                if shap_injury:
                    st.plotly_chart(render_shap_bar(shap_injury,
                                                    "Feature contribution to biomechanical risk-indicator score"),
                                    width='stretch')
                else:
                    st.caption("No biomechanical-risk SHAP data stored for this session.")
        with st.expander("Run timing"):
            render_timings(detail.get("stage_times") or {})

    # --- Manage history ---
    st.subheader("Manage history")
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        del_opts = st.multiselect("Select sessions to delete", list(options.keys()),
                                  format_func=lambda i: options[i], key="delete_select")
    with m2:
        if st.button("Delete selected", width='stretch'):
            history_db.delete_analysis(del_opts)
            st.toast(f"Deleted {len(del_opts)} session(s).")
            rerun()
    with m3:
        if st.button("Clear all history", width='stretch'):
            _confirm_clear_history()


@st.dialog("Clear All History", width="medium")
def _confirm_clear_history():
    st.warning("This will **permanently delete** all saved analysis results. This action cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Clear All History", type="primary", use_container_width=True):
            history_db.clear_all()
            st.toast("All history cleared.")
            st.rerun()


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🏏 PaceAI Biomechanics")
    st.caption("AI Motion Capture, Kinetics & Biomechanics Screening")
    st.markdown("---")

    page = st.radio("🧭 Navigation", ["⚡ Analyze", "📚 History & Compare"],
                    help="Analyze: run a new delivery. History: browse saved results and compare.")

    if page == "⚡ Analyze":
        input_mode = st.radio("Analysis Mode", ["Interactive Bio-Simulator", "\U0001f4f9 Video Motion Capture"], index=0,
                              help="Simulator: adjust the biomechanics with sliders — no video needed. "
                                   "Video: upload a clip and the app measures the delivery automatically.")
    else:
        input_mode = "Interactive Bio-Simulator"

    st.markdown("#### Model & Bowling Setup")
    bowling_arm = st.selectbox("Bowling Arm", ["Right-Arm", "Left-Arm"],
                               help="Which arm the bowler bowls with. Joints are mirrored automatically "
                                    "so left-handers aren't analyzed backwards.")
    model_choice = st.selectbox(
        "AI Prediction Backbone",
        ["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"],
        help="Selects the ML architecture for kinetic scoring. XGBoost/CatBoost/CNN-LSTM/Transformer "
             "fall back to a scikit-learn equivalent automatically if their library isn't installed."
    )

    # Video-only processing controls (kept for CV speed/accuracy tuning)
    camera_view = "behind"
    target_fps, resize_choice, denoise = 20, (640, 360), False
    slow_factor, zoom_end = 2.5, 1.8
    with st.expander("Video Processing (CV speed/accuracy)"):
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
            index=[(640, 360), (960, 540), (1280, 720)].index(resize_choice),
            help="Pixel size of the frames that get analyzed. Higher = more detail but slower.")
        denoise = st.checkbox("Denoise frames", value=denoise,
                              help="On = more accurate on noisy footage but much slower.")

    with st.expander("Diagnostics"):
        debug_overlay = st.checkbox(
            "Draw debug overlay on Analysis Replay",
            value=False,
            help="Adds a diagnostic panel (locked bowler track id + cricket-evidence "
                 "confidence + ball state) to the Analysis Replay video. Off by default.")

    with st.expander("🎬 Reels Settings"):
        slow_factor = st.slider(
            "Slow-motion factor", 1.5, 4.0, 2.5, step=0.1,
            help="How much to slow down the replay. 2.5x = 40% playback speed.")
        zoom_end = st.slider(
            "Final zoom", 1.2, 3.0, 1.8, step=0.1,
            help="How much to zoom in at the end of the clip. 1.8x = crop to 56% of frame.")

    # st.markdown("---")
    # if st.button("🚪 Log Out"):
    #     st.session_state["authenticated"] = False
    #     st.rerun()

render_chat_widget()

st.sidebar.markdown("---")
st.sidebar.markdown("#### System Status")
st.sidebar.caption(f"⚡ XGBoost: **{'Active' if ml_models.BACKEND_INFO['xgboost_available'] else 'Scikit Fallback'}**")
st.sidebar.caption(f"CatBoost: **{'Active' if ml_models.BACKEND_INFO['catboost_available'] else 'Scikit Fallback'}**")
st.sidebar.caption(f"🧠 PyTorch: **{'Active' if ml_models.BACKEND_INFO['torch_available'] else 'Disabled'}**")
st.sidebar.caption(f"🔬 SHAP Engine: **{'Active' if explainability.SHAP_AVAILABLE else 'Finite Diff'}**")
st.sidebar.caption(f"📚 History entries: **{history_db.count()}**")

# ---------------- PAGE DISPATCH ----------------
if page == "📚 History & Compare":
    render_history_page()
    st.stop()

perf_bundle, injury_bundle = load_or_train_models(model_choice)

# ---------------- HERO BANNER ----------------
st.markdown("""
<div class="hero-banner" role="banner" id="main-content">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">Fast-Bowling Biomechanics AI</h1>
            <p class="hero-subtitle">Kinematic Chain Profiling - ICC Elbow Screening - Literature-Informed Biomechanical Risk</p>
        </div>
        <div>
            <span class="status-badge badge-legal" style="margin-right: 8px;">AI Engine Ready</span>
            <span class="status-badge badge-low">YOLOv11 + MediaPipe</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


feature_vector = {}
stage_times = {}

# ---------------- FEATURE GUIDE (plain English) ----------------
with st.expander("New here? Every feature explained in plain English", expanded=False):
    render_features_guide()

# ---------------- INPUT SECTION ----------------
if input_mode == "Interactive Bio-Simulator":
    st.markdown("### 1. Delivery Kinematic Parameters")
    
    # Preset Selector
    selected_preset = st.selectbox("⚡ Quick Load Action Preset:", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset]

    with st.expander("Fine-Tune Biomechanical Sliders (Release & Impact Points)", expanded=True):
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
        processed = st.session_state.get("video_processed_file_id")
        if processed != uploaded.file_id:
            screen = _LiveAnalysisScreen()
            # Keep the REAL video AnalysisResult. It carries the identity,
            # provenance and landmark fields that the ML-only result created
            # later cannot supply (it is rebuilt from a bare feature vector).
            st.session_state["video_result"] = _run_video_analysis(
                uploaded, perf_bundle, injury_bundle, bowling_arm,
                target_fps, resize_choice, denoise, camera_view,
                slow_factor, zoom_end, debug_overlay, screen,
            )
            st.session_state["video_processed_file_id"] = uploaded.file_id
        feature_vector = st.session_state.get("video_feature_vector", {})


# ---------------- VIDEO OUTPUT REFERENCES ----------------
# Keep intermediate outputs available for the technical diagnostics section.
# The user-facing experience is rendered later, after ML results are available.
_ball_vid = st.session_state.get("video_output_path") \
    if os.path.exists(st.session_state.get("video_output_path") or "") else None
_pose_vid = st.session_state.get("pose_video_path") \
    if os.path.exists(st.session_state.get("pose_video_path") or "") else None
_reels_vid = st.session_state.get("reels_video_path") \
    if os.path.exists(st.session_state.get("reels_video_path") or "") else None
_analysis_replay = st.session_state.get("analysis_replay_path") \
    if os.path.exists(st.session_state.get("analysis_replay_path") or "") else None

# Fallback: if the primary analysis replay is missing, try other video outputs
# so the user still sees *something* even when the full overlay pipeline failed.
if _analysis_replay is None:
    for _fallback_key in ("video_output_path", "pose_video_path", "reels_video_path"):
        _fallback = st.session_state.get(_fallback_key)
        if _fallback and os.path.exists(_fallback):
            _analysis_replay = _fallback
            break


def _video_key_moments(raw_events):
    """Normalize optional pipeline event metadata without inventing events."""
    if not raw_events:
        return []
    out = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("name") or item.get("event")
        t = item.get("time_s", item.get("timestamp", item.get("time")))
        if label is None or t is None:
            continue
        try:
            t = max(0.0, float(t))
        except (TypeError, ValueError):
            continue
        out.append({"label": str(label), "time_s": t})
    return out


def render_analysis_replay(hero_video, result, feature_vector, ball_stats):
    """Premium single-video analysis experience built entirely from real outputs."""
    if not hero_video:
        # Surface pipeline warnings so the user knows *why* the replay is missing
        _warns = st.session_state.get("last_warnings") or []
        _replay_warns = [w for w in _warns if "Replay" in w or "replay" in w]
        st.markdown("""
        <div class="analysis-empty" role="status">
          <div class="analysis-empty-kicker">ANALYSIS REPLAY</div>
          <div class="analysis-empty-title">Replay unavailable</div>
          <div class="analysis-empty-copy">The analysis completed, but a playable unified replay was not generated.</div>
        </div>
        """, unsafe_allow_html=True)
        if _replay_warns:
            for _w in _replay_warns:
                st.warning(_w)
        return

    bstats = ball_stats or {}
    n_det = int(bstats.get("n_detected", 0) or 0)
    n_pred = int(bstats.get("n_interpolated", 0) or 0)
    total_frames = max(1, int(bstats.get("total_frames", 1) or 1) or
                       int(bstats.get("n_frames", 1) or 1) or 1)
    coverage = float(bstats.get("coverage_pct", 0) or 0)
    det_ratio = n_det / total_frames
    if det_ratio >= 0.7 and coverage >= 70:
        quality = "HIGH"
    elif det_ratio >= 0.4 and coverage >= 40:
        quality = "MODERATE"
    else:
        quality = "LOW"

    fps = float(getattr(config, "TARGET_FPS", 20) or 20)
    duration_s = total_frames / fps
    release_idx = bstats.get("release_idx")
    impact_idx = bstats.get("impact_idx")
    release_t = (int(release_idx) / fps) if release_idx is not None else None
    impact_t = (int(impact_idx) / fps) if impact_idx is not None else None

    moments = _video_key_moments(st.session_state.get("analysis_key_moments"))

    risk = result.injury_risk or {}
    risk_level = str(risk.get("risk_level", "")).lower()
    risk_probs = risk.get("probabilities") or []
    p_high = risk_probs[2] if len(risk_probs) > 2 else None
    # Only a real classifier probability may appear on the replay overlay. When
    # scoring was refused there is no probability to show, and inventing one
    # ("low" -> 22%) would put a fabricated number on top of the video.
    risk_pct = (p_high * 100) if p_high is not None else None
    risk_withheld = result.subject_verified is False or not risk_level

    bowler_id = getattr(result, "bowler_track_id", None)
    bowler_conf = getattr(result, "bowler_confidence", None)

    meta = dict(
        track=bowler_id,
        conf=bowler_conf,
        role=getattr(result, "bowler_role", None) or "bowler",
        duration_s=duration_s,
        frames=total_frames,
        fps=fps,
        quality=quality,
        n_det=n_det,
        n_pred=n_pred,
        coverage_pct=coverage,
        release_idx=release_idx,
        impact_idx=impact_idx,
        outcome=bstats.get("outcome"),
        events_n=len(moments),
        risk_pct=risk_pct if (bowler_id is not None and not risk_withheld) else None,
        risk_level=risk_level if not risk_withheld else "",
        features=feature_vector,
    )

    st.markdown(kinetic_ui.KINETIC_CSS, unsafe_allow_html=True)
    st.markdown(kinetic_ui.header_html(meta), unsafe_allow_html=True)

    wc_l, wc_r = st.columns([3, 1])
    with wc_l:
        st.markdown(kinetic_ui.viewport_html(meta), unsafe_allow_html=True)
        st.video(hero_video)
        if moments:
            labels = [m["label"] for m in moments]
            selected = st.selectbox("Jump to key moment", ["Start"] + labels, key="paceai_key_moment")
            if selected != "Start":
                selected_time = next(m["time_s"] for m in moments if m["label"] == selected)
                st.video(hero_video, start_time=int(selected_time))
                st.caption(f"Key moment: **{selected}** · {selected_time:.2f}s")
        else:
            st.caption("Use the native player controls for slow motion and frame-by-frame review. Key-event navigation appears when the pipeline provides event timestamps.")
        st.markdown(kinetic_ui.timeline_html(moments, duration_s, release_t, impact_t), unsafe_allow_html=True)

    with wc_r:
        st.markdown(kinetic_ui.rail_html(meta), unsafe_allow_html=True)

    st.markdown(kinetic_ui.reel_html(meta), unsafe_allow_html=True)

    if bowler_id is None:
        st.caption("Bowler identity: no bowler-like motion detected in this clip -- no bowler "
                   "box/pose crop was applied (identity is never guessed).")

    # "Players Detected" broadcast roster: the locked bowler + every classified
    # non-bowler player, with cricket-evidence confidence (matches the overlay
    # labels in the replay video). Hidden honestly when there is no data.
    roster_html = analysis_ui.render_role_roster(
        st.session_state.get("video_player_roles") or result.player_roles,
        bowler_track_id=getattr(result, "bowler_track_id", None),
        bowler_confidence=getattr(result, "bowler_confidence", None),
    )
    if roster_html:
        st.markdown(roster_html, unsafe_allow_html=True)

    coaching = result.coaching_notes or []
    if coaching:
        st.markdown("""
        <div class="priority-card" role="region" aria-label="Coaching priority">
          <div class="priority-kicker">COACHING PRIORITY</div>
          <div class="priority-title">What should I work on?</div>
          <div class="priority-copy">""" + _esc(coaching[0]) + """</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Technical replay diagnostics", expanded=False):
        if _ball_vid:
            st.markdown("**Ball tracking render**")
            st.video(_ball_vid)
        if _pose_vid:
            st.markdown("**Pose render**")
            st.video(_pose_vid)
        if _reels_vid:
            st.markdown("**Slow-motion highlight render**")
            st.video(_reels_vid)
        if bstats:
            c1, c2, c3 = st.columns(3)
            c1.metric("Tracked frames", n_det + n_pred)
            c2.metric("Predicted gap frames", n_pred)
            c3.metric("Coverage", f"{coverage:.0f}%")
        if not (_ball_vid or _pose_vid or _reels_vid):
            st.info("No intermediate video artifacts are available.")


# ---------------- ANALYSIS & VISUALIZATION ----------------
_VIDEO_ONLY_FIELDS = (
    "feature_provenance", "landmark_source_summary", "stage_backends",
    "bowler_bboxes", "bowler_track_id", "bowler_confidence",
    "bowler_confirmed", "bowler_confirm_reason", "bowler_candidates",
    "identity_switch_count", "batting_stances", "striker_track_id",
    "non_striker_track_id", "original_frame_dims", "player_roles",
    "ball_stats", "video_path", "pose_video_path", "reels_video_path",
    "analysis_replay_path", "bowling_arm", "camera_view", "warnings",
    "subject_verified", "delivery_reliable", "scoring_blocked_reason",
)


def _merge_video_result(result, video_result):
    """Overlay the video-only fields of the real pipeline result onto the ML result.

    `analyze_feature_vector` is called from a bare feature vector, so it cannot
    know anything about the clip: no provenance, no landmark source, no bowler
    identity, no artefact paths. Without this merge the result screen renders a
    permanently incomplete object. Returns a NEW result; the input is untouched.
    """
    if video_result is None:
        return result
    import dataclasses
    overrides = {}
    for name in _VIDEO_ONLY_FIELDS:
        value = getattr(video_result, name, None)
        # Only override when the video run actually produced something, so the
        # ML path's own gating decisions are never clobbered by a stale value.
        if value is None or value == {} or value == []:
            continue
        overrides[name] = value
    if not overrides:
        return result
    return dataclasses.replace(result, **overrides)


if feature_vector:
    video_subject_verified = (
        st.session_state.get("video_subject_verified")
        if input_mode.startswith("📹") else None)
    video_delivery_reliable = (
        st.session_state.get("video_delivery_reliable")
        if input_mode.startswith("📹") else None)
    result = pipeline.analyze_feature_vector(
        feature_vector, perf_bundle, injury_bundle,
        subject_verified=video_subject_verified,
        reliable=video_delivery_reliable)
    if input_mode.startswith("📹"):
        result = _merge_video_result(result, st.session_state.get("video_result"))

    # Make the latest analysis available to the sidebar chat assistant.
    merged_shap = dict(result.shap_contributions_performance or {})
    for k, v in (result.shap_contributions_injury or {}).items():
        merged_shap[f"{k} [biomech risk]"] = v
    st.session_state["features"] = dict(result.feature_vector)
    st.session_state["performance_score"] = result.performance_score
    st.session_state["injury_risk"] = result.injury_risk
    st.session_state["shap_values"] = merged_shap
    st.session_state["recommendations"] = result.coaching_notes

    # Merge video pipeline timings (if this run came from a video). The ML-only
    # `total` from the second call is ADDED to the CV pipeline's total, not
    # dropped over it -- otherwise "Total pipeline time" shows only ML time.
    if input_mode.startswith("📹") and st.session_state.get("video_stage_times"):
        stage_times = dict(st.session_state["video_stage_times"])
        ml_times = dict(result.stage_times or {})
        for k, v in ml_times.items():
            if k == "total":
                stage_times["total"] = stage_times.get("total", 0.0) + v
            else:
                stage_times[k] = v
        upload = st.session_state.get("video_upload_time")
        if upload:
            stage_times["upload"] = upload
    else:
        stage_times = dict(result.stage_times or {})

    _is_video = input_mode.startswith("📹")
    _ball_stats = dict(st.session_state.get("ball_stats") or {})

    # ---------------- REPORT / EXPORT (built once, reused by §10 and §11) ----
    _report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_mode": input_mode,
        "bowling_arm": result.bowling_arm,
        "model": getattr(perf_bundle, "model_name", model_choice),
        "performance_score": result.performance_score,
        "injury_risk": result.injury_risk,
        "icc_legal": (result.feature_vector.get("elbow_flexion_deg") is not None
                      and result.feature_vector["elbow_flexion_deg"]
                      <= config.ICC_ELBOW_EXTENSION_LIMIT_DEG),
        "coaching_feedback": result.coaching_notes,
        "shap_performance": result.shap_contributions_performance,
        "shap_injury": result.shap_contributions_injury,
        "features": result.feature_vector,
        "feature_provenance": result.feature_provenance,
        "landmark_source_summary": result.landmark_source_summary,
        "subject_verified": result.subject_verified,
        "delivery_reliable": result.delivery_reliable,
        "scoring_blocked_reason": result.scoring_blocked_reason,
        "bowler_track_id": result.bowler_track_id,
        "bowler_confidence": result.bowler_confidence,
        "ball_stats": _ball_stats,
        "stage_backends": result.stage_backends,
        "stage_times": stage_times,
        "warnings": list(result.warnings or []),
        "model_note": ("Demo models trained on synthetic data -- illustrative only, "
                       "not a validated research measurement."),
    }
    _report_json = json.dumps(_report, indent=2, default=str)

    # ---------------- ADVANCED SECTION RENDERERS (§10) --------------------
    def _adv_gauges_and_radar():
        if result.performance_score is not None:
            st.plotly_chart(
                render_modern_gauge(result.performance_score,
                                    "Demonstration Performance Indicator",
                                    "Demo score — not a validated measurement"),
                width="stretch")
        else:
            st.caption("No performance score was produced for this delivery — no gauge "
                       "is shown, because an empty dial would imply a score of zero.")
        if result.injury_risk:
            risk = result.injury_risk
            probs = risk.get("probabilities") or []
            st.markdown("**Model risk output (raw)**")
            st.json({"risk_level": risk.get("risk_level"),
                     "probabilities": probs})
            if probs:
                st.caption("Class probabilities come from the demo classifier. They are "
                           "listed for transparency, not as a risk percentage.")
        if all(k in result.feature_vector for k in
               ("shoulder_rotation_deg", "elbow_flexion_deg", "wrist_angle_deg",
                "hip_rotation_deg", "knee_flexion_deg", "trunk_lean_deg",
                "stride_length_norm", "angular_velocity_deg_s")):
            st.plotly_chart(render_radar_comparison(result.feature_vector),
                            width="stretch")
        else:
            st.caption("Radar comparison needs all eight benchmarked measurements; this "
                       "delivery is missing at least one, so it is not shown.")

    def _adv_shap():
        c1, c2 = st.columns(2)
        with c1:
            if result.shap_contributions_performance:
                st.plotly_chart(
                    render_shap_bar(result.shap_contributions_performance,
                                    "Performance drivers"),
                    width="stretch")
            else:
                st.caption("No performance SHAP data was produced for this delivery.")
        with c2:
            if result.shap_contributions_injury:
                st.plotly_chart(
                    render_shap_bar(result.shap_contributions_injury,
                                    "Biomechanical risk drivers"),
                    width="stretch")
            else:
                st.caption("No biomechanical-risk SHAP data was produced for this delivery.")

    def _adv_literature_table():
        st.markdown("**Full clinical benchmark table**")
        st.dataframe(injury_kb.all_benchmarks(), width="stretch", hide_index=True)
        st.markdown("**Workload & ACWR (optional inputs)**")
        a1, a2, a3 = st.columns(3)
        acwr = a1.number_input("ACWR", 0.0, 3.0, 0.8, 0.05)
        load7 = a2.number_input("7-day ball load", 0, 500, 0)
        rest = a3.number_input("Rest days", 0, 30, 3)
        for check in injury_kb.workload_risk(acwr=acwr, seven_day_load=load7 or None,
                                             rest_days=rest):
            _cls = {"at_risk": "error", "warning": "warning"}.get(check["status"])
            getattr(st, _cls or "caption")(f"**{check['check']}** — {check['detail']}")

    def _adv_model_quality():
        render_model_quality_expander(perf_bundle, injury_bundle)

    def _adv_timing():
        if stage_times:
            render_timings(stage_times)
        else:
            st.caption("No timing data available for this run.")

    def _adv_notes():
        for w in result.warnings or []:
            st.warning(w)
        if not result.warnings:
            st.caption("The pipeline reported no degradation notes for this run.")

    def _session_writer():
        st.markdown("### 💾 Save to History")
        st.caption("Persist this delivery so you can compare it against future sessions. "
                   "Saving the same result twice is idempotent.")
        save_athlete = st.text_input("Bowler name", key="save_athlete")
        save_label = st.text_input("Label (optional)", key="save_label")
        save_tags = st.text_input("Tags (comma-separated, optional)", key="save_tags")
        if st.button("💾 Save this result", type="primary"):
            saved_id, inserted = history_db.save_analysis(
                result, label=save_label, input_mode=input_mode,
                bowling_arm=bowling_arm.lower().split("-")[0], model=model_choice,
                athlete=save_athlete, tags=save_tags)
            if inserted:
                st.success(f"Saved to history (id #{saved_id}). Open **History & Compare** "
                           f"in the sidebar to view and compare your saved results.")
            else:
                st.info(f"This result was already saved (id #{saved_id}) — no duplicate was "
                        f"created. Change the label/tags or bowler name to log it separately.")
        st.download_button(
            label="📥 Download Full Delivery Analysis (JSON)",
            data=_report_json,
            file_name="bowling_biomechanics_report.json",
            mime="application/json",
        )

    # ---------------- ADAPTIVE RESULT PAGE (§01-§11) -----------------------
    result_view.render_result_page(
        result,
        perf_bundle=perf_bundle,
        injury_bundle=injury_bundle,
        is_video=_is_video,
        replay_fn=render_analysis_replay if _is_video else None,
        replay_path=_analysis_replay,
        ball_stats=_ball_stats,
        advanced_renderers={
            "gauges_and_radar": _adv_gauges_and_radar,
            "shap": _adv_shap,
            "literature_table": _adv_literature_table,
            "model_quality": _adv_model_quality,
            "timing": _adv_timing,
            "notes": _adv_notes,
        },
        session_writer=_session_writer,
    )

    if _is_video and st.session_state.pop("scroll_to_replay", False):
        st.components.v1.html(
            '<script>document.getElementById("paceai-analysis-replay")'
            "?.scrollIntoView({behavior:'smooth'});</script>",
            height=1,
        )

else:
    st.info("Enter features manually or upload a video to run the analysis.")

st.markdown("---")
st.caption(
    "⚡ **PaceAI Biomechanics Engine** • Demo models are trained on synthetic data "
    "(src/synthetic_data.py) -- retrain on real labeled data (`python train_demo_model.py "
    "--data your_dataset.csv`) before using for real coaching/medical decisions. "
    "Designed for elite high-performance cricket centers, coaches, and sports physiotherapists."
)
