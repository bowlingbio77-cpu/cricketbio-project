"""
PaceAI Design System
====================
Single source of truth for all visual tokens, component styles, and layout CSS.

Imported by app.py and injected via st.markdown(unsafe_allow_html=True).
No ML/pipeline code lives here -- presentation only.

The active theme is DARK. A reference block of light-theme tokens for the
"CRICKET-CV v4.2" kinetic-biomechanics prototype lives below the dark tokens
(see section "KINETIC BIOMECHANICS LIGHT THEME"); it is NOT applied at runtime.
"""

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------

# Backgrounds
BG_APP = "#080B10"
BG_PRIMARY = "#0F141B"
BG_SECONDARY = "#151B23"
BG_TERTIARY = "#1C2330"
BG_ELEVATED = "#1A2030"

# Borders
BORDER_DEFAULT = "#252D38"
BORDER_SUBTLE = "#1E2530"
BORDER_STRONG = "#333D4A"

# Text
TEXT_PRIMARY = "#E8ECF1"
TEXT_SECONDARY = "#8B95A5"
TEXT_MUTED = "#5A6577"
TEXT_ACCENT = "#3B9EED"

# Semantic
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_ERROR = "#EF4444"
COLOR_INFO = "#3B9EED"

# Accent (PaceAI cyan)
ACCENT = "#3B9EED"
ACCENT_DIM = "rgba(59,158,237,0.12)"
ACCENT_BORDER = "rgba(59,158,237,0.25)"

# Surfaces for cards
CARD_BG = "linear-gradient(180deg, #131920 0%, #0F141B 100%)"
CARD_BORDER = "#252D38"

# Typography
FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', 'Consolas', monospace"

# Spacing scale (px)
SP_XS = "4px"
SP_SM = "8px"
SP_MD = "12px"
SP_LG = "16px"
SP_XL = "24px"
SP_2XL = "32px"

# Radius
RADIUS_SM = "6px"
RADIUS_MD = "10px"
RADIUS_LG = "14px"
RADIUS_XL = "18px"


# ---------------------------------------------------------------------------
# KINETIC BIOMECHANICS LIGHT THEME  (prototype "CRICKET-CV v4.2")
# ---------------------------------------------------------------------------
# Reference block ONLY -- the app's active theme is the dark PaceAI theme above.
# Tokens below mirror the stitched design package:
#   * docs/DESIGN.md                                        (design spec)
#   * assets/paceai_cricket_cv_v4_2_prototype.html          (full-screen mockup)
#   * assets/paceai_cricket_cv_v4_2_screen.png              (screenshot)
# Nothing in this block is applied at runtime; it exists so a future light-mode
# refactor can import a single, spec-correct source of truth.

# --- Light surface palette (from docs/DESIGN.md) ---
LIGHT_SURFACE = "#f4f4f4"
LIGHT_SURFACE_DIM = "#dadada"
LIGHT_SURFACE_BRIGHT = "#f9f9f9"
LIGHT_SURFACE_CONTAINER_LOWEST = "#ffffff"
LIGHT_SURFACE_CONTAINER_LOW = "#edf1f5"
LIGHT_SURFACE_CONTAINER = "#e5ebf2"
LIGHT_SURFACE_CONTAINER_HIGH = "#dbe3ec"
LIGHT_SURFACE_CONTAINER_HIGHEST = "#cfd9e5"
LIGHT_ON_SURFACE = "#0f2238"
LIGHT_ON_SURFACE_VARIANT = "#34495e"
LIGHT_INVERSE_SURFACE = "#2f3131"
LIGHT_INVERSE_ON_SURFACE = "#f1f1f1"

# Outline / stroke tones
LIGHT_OUTLINE = "#8b9eb5"
LIGHT_OUTLINE_VARIANT = "#c4d1df"
LIGHT_SURFACE_TINT = "#506070"

# --- Light brand palette (material tokens) ---
LIGHT_PRIMARY = "#506070"
LIGHT_ON_PRIMARY = "#0c2543"
LIGHT_PRIMARY_CONTAINER = "#d9eafd"
LIGHT_ON_PRIMARY_CONTAINER = "#596a7a"
LIGHT_INVERSE_PRIMARY = "#b7c8db"
LIGHT_SECONDARY = "#406089"
LIGHT_ON_SECONDARY = "#ffffff"
LIGHT_SECONDARY_CONTAINER = "#aecefd"
LIGHT_ON_SECONDARY_CONTAINER = "#375880"
LIGHT_TERTIARY = "#30647d"
LIGHT_ON_TERTIARY = "#ffffff"
LIGHT_TERTIARY_CONTAINER = "#ceecff"
LIGHT_ON_TERTIARY_CONTAINER = "#3b6e87"
LIGHT_BACKGROUND = "#f9f9f9"
LIGHT_ON_BACKGROUND = "#1a1c1c"

# Error state
LIGHT_ERROR = "#ba1a1a"
LIGHT_ON_ERROR = "#ffffff"
LIGHT_ERROR_CONTAINER = "#ffdad6"
LIGHT_ON_ERROR_CONTAINER = "#93000a"

# --- Component accents actually used in the v4.2 prototype (code.html) ---
LIGHT_NAVY = "#1a3d64"        # primary structural typography / solid buttons
LIGHT_NAVY_HOVER = "#142f4c"
LIGHT_ICE_COBALT = "#d9eafd"  # high-visibility interactive surface / chip selected
LIGHT_TEAL_SLATE = "#1d546c"  # sports-tech accent: telemetry curves, vectors
LIGHT_MONO_FAMILY = "'JetBrains Mono', 'SF Mono', Consolas, monospace"
LIGHT_SANS_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

# --- Typography scale (from docs/DESIGN.md) ---
LIGHT_DISPLAY_LG = "48px/56px/700/-0.025em"
LIGHT_DISPLAY_LG_MOBILE = "36px/44px/700/-0.02em"
LIGHT_HEADLINE_LG = "32px/40px/600/-0.02em"
LIGHT_HEADLINE_MD = "24px/32px/600/-0.015em"
LIGHT_HEADLINE_SM = "20px/28px/600/-0.01em"
LIGHT_TITLE_MD = "16px/24px/600/-0.005em"
LIGHT_BODY_LG = "16px/24px/400/0em"
LIGHT_BODY_MD = "14px/20px/400/0em"
LIGHT_BODY_SM = "12px/16px/400/0.01em"
LIGHT_LABEL_LG = "14px/20px/500/0.01em"
LIGHT_LABEL_MD = "12px/16px/500/0.02em"
LIGHT_LABEL_XS = "10px/14px/600/0.06em"
LIGHT_METRIC_DISPLAY = "32px/36px/700/-0.03em"   # telemetry readouts; use tabular figures

# --- Layout tokens ---
LIGHT_SPACING = {
    "space-xs": "0.25rem",
    "space-sm": "0.5rem",
    "space-md": "1rem",
    "space-lg": "1.5rem",
    "space-xl": "2rem",
    "gutter": "1.5rem",
    "gutter-mobile": "0.75rem",
    "margin": "2rem",
    "margin-mobile": "1rem",
}
LIGHT_RADIUS = {
    "sm": "0.125rem",       # standard controls  (~4px)
    "DEFAULT": "0.25rem",
    "md": "0.375rem",
    "lg": "0.5rem",         # cards / telemetry containers (8px)
    "xl": "0.75rem",        # floating dialogs / modals (12px)
    "full": "9999px",       # continuous sensor nodes / joint markers
}
LIGHT_GRID = {
    "desktop": (12, 24, 32),   # columns, gutter px, margin px
    "tablet": (8, 16, 24),
    "mobile": (4, 12, 16),
}
LIGHT_ELEVATION = {
    "card": "0 1px 3px rgba(26, 61, 100, 0.06), 0 1px 2px rgba(26, 61, 100, 0.04)",
    "raised": "0 4px 12px rgba(26, 61, 100, 0.10)",
    "overlay": "0 12px 28px -4px rgba(26, 61, 100, 0.14)",
    "recessed_inset": "inset 0 1px 2px rgba(26, 61, 100, 0.08)",
}


# ---------------------------------------------------------------------------
# MASTER CSS
# ---------------------------------------------------------------------------

DESIGN_SYSTEM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ============================================================
   RESET & BASE
   ============================================================ */
html, body, [class*="css"] {{
    font-family: {FONT_FAMILY} !important;
    color: {TEXT_PRIMARY};
}}

/* Streamlit overrides */
.stApp {{
    background-color: {BG_APP} !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {BG_PRIMARY} !important;
    border-right: 1px solid {BORDER_DEFAULT} !important;
}}

/* Remove default Streamlit padding on main */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}}

/* ============================================================
   SKIP LINK (a11y)
   ============================================================ */
.skip-link {{
    position: absolute; left: -9999px; top: auto;
    width: 1px; height: 1px; overflow: hidden;
    z-index: 999999; padding: 12px 20px; margin: 8px;
    background: {ACCENT}; color: {BG_APP}; font-weight: 700;
    border-radius: {RADIUS_SM}; text-decoration: none; font-size: 0.85rem;
}}
.skip-link:focus {{
    position: fixed; left: 12px; top: 12px;
    width: auto; height: auto; overflow: visible;
}}

/* ============================================================
   FOCUS STATES (a11y)
   ============================================================ */
:focus-visible {{
    outline: 2px solid {ACCENT};
    outline-offset: 2px;
}}
.stSelectbox:focus-within,
.stSlider:focus-within,
.stRadio:focus-within,
.stButton:focus-within,
.stTextInput:focus-within,
.stNumberInput:focus-within {{
    box-shadow: 0 0 0 2px rgba(59,158,237,0.3);
}}
.sr-only {{
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}}

/* ============================================================
   TYPOGRAPHY
   ============================================================ */
.p-page-title {{
    font-size: 1.5rem; font-weight: 800; color: {TEXT_PRIMARY};
    letter-spacing: -0.02em; margin: 0 0 4px 0;
}}
.p-page-subtitle {{
    font-size: 0.85rem; color: {TEXT_SECONDARY}; margin: 0 0 20px 0;
    line-height: 1.5;
}}
.p-section-title {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin: 0 0 12px 0;
}}
.p-label {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: {TEXT_SECONDARY};
}}

/* ============================================================
   NAVIGATION SIDEBAR
   ============================================================ */
.p-nav-brand {{
    padding: 20px 0 16px;
    border-bottom: 1px solid {BORDER_DEFAULT};
    margin-bottom: 16px;
}}
.p-nav-brand-name {{
    font-size: 1.1rem; font-weight: 800; color: {TEXT_PRIMARY};
    letter-spacing: -0.01em; display: flex; align-items: center; gap: 8px;
}}
.p-nav-brand-name .icon {{
    width: 28px; height: 28px; border-radius: {RADIUS_SM};
    background: linear-gradient(135deg, {ACCENT}, #2563EB);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 14px; color: white; font-weight: 900;
}}
.p-nav-brand-sub {{
    font-size: 0.7rem; color: {TEXT_MUTED}; margin-top: 2px;
    letter-spacing: 0.04em;
}}
.p-nav-section-label {{
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin: 16px 0 8px;
}}
.p-nav-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: {RADIUS_SM};
    font-size: 0.85rem; font-weight: 500; color: {TEXT_SECONDARY};
    cursor: pointer; transition: all 0.15s ease;
    border: 1px solid transparent;
}}
.p-nav-item:hover {{
    background: {BG_SECONDARY}; color: {TEXT_PRIMARY};
}}
.p-nav-item.active {{
    background: {ACCENT_DIM}; color: {ACCENT};
    border-color: {ACCENT_BORDER};
    font-weight: 600;
}}
.p-nav-item .nav-icon {{
    width: 18px; text-align: center; font-size: 0.9rem;
}}

/* System status area */
.p-system-row {{
    display: flex; align-items: center; gap: 6px;
    padding: 3px 0; font-size: 0.72rem; color: {TEXT_MUTED};
}}
.p-system-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}}
.p-system-dot.on {{ background: {COLOR_SUCCESS}; }}
.p-system-dot.off {{ background: {COLOR_ERROR}; }}
.p-system-dot.warn {{ background: {COLOR_WARNING}; }}

/* ============================================================
   CARDS
   ============================================================ */
.p-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: {RADIUS_LG};
    padding: 20px;
    transition: border-color 0.2s ease;
}}
.p-card:hover {{
    border-color: {BORDER_STRONG};
}}
.p-card-header {{
    margin-bottom: 16px;
}}
.p-card-title {{
    font-size: 0.95rem; font-weight: 700; color: {TEXT_PRIMARY};
    margin: 0;
}}
.p-card-subtitle {{
    font-size: 0.75rem; color: {TEXT_SECONDARY}; margin: 2px 0 0;
}}

/* Metric card (KPI) */
.p-kpi {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: {RADIUS_LG};
    padding: 18px 20px;
}}
.p-kpi-label {{
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 8px;
}}
.p-kpi-value {{
    font-size: 1.8rem; font-weight: 800; color: {TEXT_PRIMARY};
    line-height: 1.1; margin-bottom: 4px;
}}
.p-kpi-value.success {{ color: {COLOR_SUCCESS}; }}
.p-kpi-value.warning {{ color: {COLOR_WARNING}; }}
.p-kpi-value.error {{ color: {COLOR_ERROR}; }}
.p-kpi-value.accent {{ color: {ACCENT}; }}
.p-kpi-note {{
    font-size: 0.7rem; color: {TEXT_MUTED}; line-height: 1.4;
}}

/* ============================================================
   BADGES
   ============================================================ */
.p-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 100px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; white-space: nowrap;
}}
.p-badge-success {{
    background: rgba(34,197,94,0.12); color: {COLOR_SUCCESS};
    border: 1px solid rgba(34,197,94,0.25);
}}
.p-badge-warning {{
    background: rgba(245,158,11,0.12); color: {COLOR_WARNING};
    border: 1px solid rgba(245,158,11,0.25);
}}
.p-badge-error {{
    background: rgba(239,68,68,0.12); color: {COLOR_ERROR};
    border: 1px solid rgba(239,68,68,0.25);
}}
.p-badge-info {{
    background: {ACCENT_DIM}; color: {ACCENT};
    border: 1px solid {ACCENT_BORDER};
}}
.p-badge-neutral {{
    background: rgba(90,101,119,0.15); color: {TEXT_SECONDARY};
    border: 1px solid rgba(90,101,119,0.25);
}}
.p-badge-dot::before {{
    content: '';
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; flex-shrink: 0;
}}

/* ============================================================
   BUTTONS
   ============================================================ */
.p-btn-primary {{
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    padding: 12px 28px; border-radius: {RADIUS_MD};
    background: linear-gradient(180deg, {ACCENT}, #2563EB);
    color: white; font-weight: 700; font-size: 0.85rem;
    letter-spacing: 0.02em; border: none; cursor: pointer;
    transition: all 0.2s ease; width: 100%;
    box-shadow: 0 2px 8px rgba(59,158,237,0.25);
}}
.p-btn-primary:hover {{
    box-shadow: 0 4px 16px rgba(59,158,237,0.35);
    transform: translateY(-1px);
}}
.p-btn-secondary {{
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    padding: 10px 20px; border-radius: {RADIUS_MD};
    background: transparent; color: {TEXT_SECONDARY};
    font-weight: 600; font-size: 0.8rem;
    border: 1px solid {BORDER_DEFAULT}; cursor: pointer;
    transition: all 0.15s ease;
}}
.p-btn-secondary:hover {{
    background: {BG_SECONDARY}; color: {TEXT_PRIMARY};
    border-color: {BORDER_STRONG};
}}

/* ============================================================
   UPLOAD ZONE
   ============================================================ */
.p-upload-zone {{
    border: 2px dashed {BORDER_STRONG};
    border-radius: {RADIUS_XL};
    padding: 48px 32px;
    text-align: center;
    background: {BG_PRIMARY};
    transition: all 0.2s ease;
    cursor: pointer;
}}
.p-upload-zone:hover {{
    border-color: {ACCENT};
    background: {ACCENT_DIM};
}}
.p-upload-icon {{
    font-size: 2rem; margin-bottom: 12px; opacity: 0.6;
}}
.p-upload-title {{
    font-size: 1rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 4px;
}}
.p-upload-subtitle {{
    font-size: 0.8rem; color: {TEXT_SECONDARY}; margin-bottom: 12px;
}}
.p-upload-formats {{
    font-size: 0.65rem; color: {TEXT_MUTED}; letter-spacing: 0.08em;
    text-transform: uppercase;
}}

/* ============================================================
   PROGRESS / STAGES
   ============================================================ */
.p-progress-track {{
    height: 3px; border-radius: 3px;
    background: {BG_SECONDARY}; overflow: hidden;
}}
.p-progress-fill {{
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, {ACCENT}, {COLOR_SUCCESS});
    transition: width 0.4s ease;
}}
.p-stage-list {{
    list-style: none; padding: 0; margin: 0;
}}
.p-stage-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; font-size: 0.82rem;
}}
.p-stage-icon {{
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700; flex-shrink: 0;
}}
.p-stage-icon.done {{
    background: rgba(34,197,94,0.15); color: {COLOR_SUCCESS};
}}
.p-stage-icon.active {{
    background: {ACCENT_DIM}; color: {ACCENT};
    animation: pPulse 1.5s ease-in-out infinite;
}}
.p-stage-icon.pending {{
    background: rgba(90,101,119,0.1); color: {TEXT_MUTED};
}}
.p-stage-label.done {{ color: {TEXT_SECONDARY}; }}
.p-stage-label.active {{ color: {TEXT_PRIMARY}; font-weight: 600; }}
.p-stage-label.pending {{ color: {TEXT_MUTED}; }}

@keyframes pPulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

/* ============================================================
   TABS (Streamlit override)
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {BG_SECONDARY};
    border-radius: {RADIUS_MD};
    padding: 3px;
    border: 1px solid {BORDER_DEFAULT};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: {RADIUS_SM} !important;
    color: {TEXT_SECONDARY} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {BG_TERTIARY} !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    border: 1px solid {BORDER_DEFAULT} !important;
}}

/* ============================================================
   EXPANDER
   ============================================================ */
.stExpander {{
    background: {BG_PRIMARY} !important;
    border: 1px solid {BORDER_DEFAULT} !important;
    border-radius: {RADIUS_MD} !important;
}}

/* ============================================================
   DATAFRAME
   ============================================================ */
.stDataFrame {{
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {RADIUS_MD};
    overflow: hidden;
}}

/* ============================================================
   HERO BANNER (Analyze page)
   ============================================================ */
.p-hero {{
    background: linear-gradient(135deg, #0D1117 0%, #111827 50%, #0F172A 100%);
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {RADIUS_XL};
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
.p-hero::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, {ACCENT}, transparent);
    opacity: 0.4;
}}
.p-hero-title {{
    font-size: 1.6rem; font-weight: 800; color: {TEXT_PRIMARY};
    letter-spacing: -0.02em; margin: 0 0 6px 0;
}}
.p-hero-subtitle {{
    font-size: 0.85rem; color: {TEXT_SECONDARY}; margin: 0 0 16px 0;
    line-height: 1.5; max-width: 560px;
}}
.p-hero-badges {{
    display: flex; gap: 8px; flex-wrap: wrap;
}}

/* ============================================================
   EMPTY STATE
   ============================================================ */
.p-empty {{
    text-align: center; padding: 48px 24px;
    border: 1px dashed {BORDER_STRONG};
    border-radius: {RADIUS_XL};
    background: {BG_PRIMARY};
}}
.p-empty-icon {{
    font-size: 2.5rem; margin-bottom: 12px; opacity: 0.4;
}}
.p-empty-title {{
    font-size: 1.1rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 4px;
}}
.p-empty-desc {{
    font-size: 0.82rem; color: {TEXT_SECONDARY}; max-width: 400px; margin: 0 auto;
}}

/* ============================================================
   ANALYSIS REPLAY
   ============================================================ */
.p-replay-shell {{
    background: {BG_PRIMARY};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: {RADIUS_XL};
    overflow: hidden;
    margin: 24px 0;
}}
.p-replay-header {{
    padding: 20px 24px 16px;
    border-bottom: 1px solid {BORDER_DEFAULT};
    display: flex; justify-content: space-between; align-items: flex-start;
}}
.p-replay-header h3 {{
    font-size: 1.1rem; font-weight: 700; color: {TEXT_PRIMARY};
    margin: 0 0 2px 0;
}}
.p-replay-header p {{
    font-size: 0.78rem; color: {TEXT_SECONDARY}; margin: 0;
}}
.p-replay-body {{
    padding: 0;
}}
.p-replay-status {{
    padding: 12px 24px;
    display: flex; gap: 12px; flex-wrap: wrap;
    border-top: 1px solid {BORDER_DEFAULT};
    background: {BG_SECONDARY};
}}

/* ============================================================
   DRILL / COACHING CARD
   ============================================================ */
.p-drill {{
    border-left: 3px solid {COLOR_SUCCESS};
    background: {BG_SECONDARY};
    padding: 14px 18px;
    border-radius: 0 {RADIUS_SM} {RADIUS_SM} 0;
    margin-bottom: 10px;
    border-top: 1px solid {BORDER_DEFAULT};
    border-right: 1px solid {BORDER_DEFAULT};
    border-bottom: 1px solid {BORDER_DEFAULT};
}}

/* ============================================================
   PRIORITY / FINDINGS CARD
   ============================================================ */
.p-finding {{
    border: 1px solid rgba(59,158,237,0.2);
    background: linear-gradient(135deg, rgba(59,158,237,0.06), {BG_PRIMARY});
    border-radius: {RADIUS_LG};
    padding: 20px;
    margin: 16px 0;
}}
.p-finding-kicker {{
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: {ACCENT}; margin-bottom: 6px;
}}
.p-finding-title {{
    font-size: 1rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-bottom: 8px;
}}
.p-finding-body {{
    font-size: 0.85rem; color: {TEXT_SECONDARY}; line-height: 1.6;
}}

/* ============================================================
   RESULTS GRID
   ============================================================ */
.p-results-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}}
@media (max-width: 900px) {{
    .p-results-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 600px) {{
    .p-results-grid {{ grid-template-columns: 1fr; }}
}}

/* ============================================================
   HISTORY CARDS
   ============================================================ */
.p-history-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: {RADIUS_MD};
    padding: 16px;
    transition: border-color 0.2s ease;
    cursor: pointer;
}}
.p-history-card:hover {{
    border-color: {ACCENT_BORDER};
}}
.p-history-meta {{
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    margin-bottom: 8px;
}}
.p-history-title {{
    font-size: 0.85rem; font-weight: 600; color: {TEXT_PRIMARY};
}}
.p-history-date {{
    font-size: 0.7rem; color: {TEXT_MUTED};
}}

/* ============================================================
   FOOTER
   ============================================================ */
.p-footer {{
    border-top: 1px solid {BORDER_DEFAULT};
    padding: 16px 0;
    margin-top: 24px;
    text-align: center;
}}
.p-footer-text {{
    font-size: 0.7rem; color: {TEXT_MUTED}; line-height: 1.6;
}}

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 900px) {{
    .p-hero {{ padding: 24px 20px; }}
    .p-hero-title {{ font-size: 1.3rem; }}
    .p-kpi-value {{ font-size: 1.4rem; }}
}}
@media (max-width: 600px) {{
    .p-hero {{ padding: 18px 16px; border-radius: {RADIUS_LG}; }}
    .p-hero-title {{ font-size: 1.1rem; }}
    .p-kpi {{ padding: 14px 16px; }}
    .p-kpi-value {{ font-size: 1.2rem; }}
}}

/* Streamlit column stacking on narrow */
@media (max-width: 768px) {{
    [data-testid="stHorizontalBlock"] > div {{
        flex-basis: 100% !important;
        max-width: 100% !important;
    }}
}}

/* ============================================================
   REDUCED MOTION
   ============================================================ */
@media (prefers-reduced-motion: reduce) {{
    .p-stage-icon.active {{ animation: none; }}
    *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# HELPER: inject design system
# ---------------------------------------------------------------------------

def inject():
    """Call once at the top of app.py to inject the design system CSS."""
    import streamlit as st
    st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
    # Accessibility: skip link + lang
    st.markdown(
        '<a href="#main-content" class="skip-link">Skip to main content</a>'
        '<script>document.documentElement.lang="en";</script>',
        unsafe_allow_html=True,
    )
