"""
PaceAI Reusable UI Components
==============================
Streamlit helper functions for consistent, production-grade UI patterns.
All functions render via st.markdown or st.columns -- no ML logic.
"""
from __future__ import annotations
import streamlit as st
import html as _html
from typing import Optional, List, Tuple


def _esc(text) -> str:
    """HTML-escape for safe injection."""
    if text is None:
        return ""
    return _html.escape(str(text))


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------

def render_sidebar_nav():
    """Render the sidebar navigation and system status. Returns (page, input_mode)."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="p-nav-brand">
            <div class="p-nav-brand-name">
                <span class="icon">P</span> PaceAI
            </div>
            <div class="p-nav-brand-sub">Cricket Bowling Biomechanics</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown('<div class="p-nav-section-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "nav", ["Analyze", "History & Compare"],
            label_visibility="collapsed",
            help="Analyze: run a new delivery. History: browse saved results.",
        )

        input_mode = None
        if page == "Analyze":
            st.markdown('<div class="p-nav-section-label">Analysis Mode</div>', unsafe_allow_html=True)
            input_mode = st.radio(
                "mode", ["Video Motion Capture", "Interactive Bio-Simulator"],
                label_visibility="collapsed",
                help="Video: upload a clip for full CV analysis. Simulator: slider-based kinematic entry.",
            )

        st.markdown("---")

        # Primary settings (always accessible)
        st.markdown('<div class="p-nav-section-label">Settings</div>', unsafe_allow_html=True)
        bowling_arm = st.selectbox(
            "Bowling Arm", ["Right-Arm", "Left-Arm"],
            help="Which arm the bowler bowls with. Joints are mirrored for left-handers.",
        )
        model_choice = st.selectbox(
            "AI Model",
            ["random_forest", "xgboost", "catboost"],
            help="ML architecture for kinetic scoring.",
        )

        # Advanced settings (collapsed)
        with st.expander("Advanced Settings", expanded=False):
            camera_view = st.selectbox(
                "Camera view", ["behind", "side"],
                help="Recording orientation.",
            )
            processing_preset = st.selectbox(
                "Quality preset", ["Balanced", "Fast", "Maximum accuracy"],
            )
            target_fps = 20
            resize_choice = (640, 360)
            denoise = False
            if processing_preset == "Fast":
                target_fps, resize_choice, denoise = 15, (640, 360), False
            elif processing_preset == "Maximum accuracy":
                target_fps, resize_choice, denoise = 30, (960, 540), True
            else:
                target_fps, resize_choice, denoise = 20, (640, 360), False
            target_fps = st.slider("Target FPS", 5, 30, target_fps, step=5)
            resize_choice = st.selectbox(
                "Resolution", [(640, 360), (960, 540), (1280, 720)],
                format_func=lambda d: f"{d[0]}x{d[1]}",
                index=[(640, 360), (960, 540), (1280, 720)].index(resize_choice),
            )
            denoise = st.checkbox("Denoise frames", value=denoise)
            debug_overlay = st.checkbox("Debug overlay on replay", value=False)
            slow_factor = st.slider("Slow-motion", 1.5, 4.0, 2.5, step=0.1)
            zoom_end = st.slider("Final zoom", 1.2, 3.0, 1.8, step=0.1)

        st.markdown("---")

        # System status
        from src import ml_models, explainability, history_db
        st.markdown('<div class="p-nav-section-label">System</div>', unsafe_allow_html=True)
        _sys_items = [
            ("CV Engine", "on"),
            ("Pose Engine", "on"),
            ("Tracking", "on"),
        ]
        # Check actual backend status
        xgb = ml_models.BACKEND_INFO.get('xgboost_available', False)
        torch = ml_models.BACKEND_INFO.get('torch_available', False)
        shap = explainability.SHAP_AVAILABLE
        _sys_items = [
            ("XGBoost", "on" if xgb else "off"),
            ("PyTorch", "on" if torch else "off"),
            ("SHAP", "on" if shap else "warn"),
        ]
        for name, status in _sys_items:
            st.markdown(
                f'<div class="p-system-row">'
                f'<span class="p-system-dot {status}"></span>{name}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="p-system-row" style="margin-top:4px; color:var(--text-muted);">'
            f'History: {history_db.count()} entries</div>',
            unsafe_allow_html=True,
        )

    defaults = dict(target_fps=20, resize_choice=(640, 360), denoise=False,
                    camera_view="behind", slow_factor=2.5, zoom_end=1.8, debug_overlay=False)
    if page == "Analyze" and input_mode == "Video Motion Capture":
        defaults.update(target_fps=target_fps, resize_choice=resize_choice,
                        denoise=denoise, camera_view=camera_view,
                        slow_factor=slow_factor, zoom_end=zoom_end,
                        debug_overlay=debug_overlay)

    return page, input_mode, bowling_arm, model_choice, defaults


# ---------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------

def render_hero(title: str, subtitle: str = "", badges: list = None):
    """Render the top hero banner."""
    badge_html = ""
    if badges:
        badge_html = '<div class="p-hero-badges">' + "".join(
            f'<span class="p-badge {b.get("cls", "p-badge-neutral")}">'
            f'{_esc(b.get("text", ""))}</span>'
            for b in badges
        ) + '</div>'

    st.markdown(f"""
    <div class="p-hero" id="main-content">
        <h1 class="p-hero-title">{_esc(title)}</h1>
        <p class="p-hero-subtitle">{_esc(subtitle)}</p>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UPLOAD ZONE
# ---------------------------------------------------------------------------

def render_upload_zone():
    """Render the video upload area. Returns uploaded file or None."""
    st.markdown("""
    <div class="p-upload-zone">
        <div class="p-upload-icon">&#127909;</div>
        <div class="p-upload-title">Upload Bowling Video</div>
        <div class="p-upload-subtitle">Drag and drop or click to select</div>
        <div class="p-upload-formats">MP4 / MOV / AVI</div>
    </div>
    """, unsafe_allow_html=True)
    return st.file_uploader(
        "Upload bowling delivery video clip",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# KPI CARD
# ---------------------------------------------------------------------------

def render_kpi(label: str, value: str, note: str = "", color: str = "primary"):
    """Render a single KPI metric card."""
    color_cls = f" {color}" if color != "primary" else ""
    note_html = f'<div class="p-kpi-note">{_esc(note)}</div>' if note else ""
    st.markdown(f"""
    <div class="p-kpi">
        <div class="p-kpi-label">{_esc(label)}</div>
        <div class="p-kpi-value{color_cls}">{_esc(value)}</div>
        {note_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RESULTS GRID
# ---------------------------------------------------------------------------

def render_results_grid(items: list):
    """
    Render a 4-column grid of KPI cards.
    items: list of dicts with keys: label, value, note, color
    """
    n = len(items)
    cols = st.columns(min(n, 4))
    for i, item in enumerate(items):
        col_idx = i % 4
        with cols[col_idx]:
            render_kpi(
                item.get("label", ""),
                item.get("value", ""),
                item.get("note", ""),
                item.get("color", "primary"),
            )


# ---------------------------------------------------------------------------
# STAGE LIST (for analysis progress)
# ---------------------------------------------------------------------------

def render_stage_list(stages: list, active_idx: int = -1, done_count: int = 0):
    """
    Render a vertical list of pipeline stages.
    stages: list of dicts with keys: label, tick
    active_idx: tick of the currently active stage
    done_count: number of completed stages
    """
    html_parts = ['<ul class="p-stage-list">']
    for stage in stages:
        tick = stage.get("tick", 0)
        label = stage.get("label", "")
        if tick < active_idx or (active_idx < 0 and tick <= done_count):
            cls = "done"
            icon = "&#10003;"  # checkmark
        elif tick == active_idx:
            cls = "active"
            icon = "&#9679;"  # dot
        else:
            cls = "pending"
            icon = "&#9675;"  # circle
        html_parts.append(
            f'<li class="p-stage-item">'
            f'<span class="p-stage-icon {cls}">{icon}</span>'
            f'<span class="p-stage-label {cls}">{_esc(label)}</span>'
            f'</li>'
        )
    html_parts.append('</ul>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FINDING CARD
# ---------------------------------------------------------------------------

def render_finding(kicker: str, title: str, body: str):
    """Render a key-finding card (used for coaching priorities, etc.)."""
    st.markdown(f"""
    <div class="p-finding">
        <div class="p-finding-kicker">{_esc(kicker)}</div>
        <div class="p-finding-title">{_esc(title)}</div>
        <div class="p-finding-body">{_esc(body)}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------------------------

def render_empty(icon: str, title: str, desc: str):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div class="p-empty">
        <div class="p-empty-icon">{icon}</div>
        <div class="p-empty-title">{_esc(title)}</div>
        <div class="p-empty-desc">{_esc(desc)}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

def render_footer():
    """Render the app footer."""
    st.markdown("""
    <div class="p-footer">
        <div class="p-footer-text">
            PaceAI Biomechanics Engine &mdash; Demo models trained on synthetic data.
            Screenings only, not clinical diagnoses or validated measurements.
        </div>
    </div>
    """, unsafe_allow_html=True)
