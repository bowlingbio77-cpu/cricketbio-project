"""
analysis_ui.py

Premium live-analysis screen for the PaceAI video pipeline.

This module is purely a PRESENTATION layer. It renders the state that the
backend pipeline already produces (via ``pipeline.analyze_video`` progress
callbacks and the returned ``AnalysisResult``) into an HTML/CSS "sports
performance lab" view. It does NOT touch detection, tracking, ball tracking,
pose, biomechanics, or any ML stage.

Honest-progress contract (important):
  * The backend only exposes STAGE-level progress (``progress_cb(done, total,
    label)`` fired once per completed stage). There is no per-frame percentage.
  * This module therefore shows a thin progress line derived from the REAL
    ``done / total`` fraction, and never fabricates a frame counter while a
    stage is still running.
  * Frame-level metadata (resolution / fps / frame count) is only displayed
    when a real value exists. Unknown values are hidden, not faked.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Stage grouping -- logical display order that also matches the real pipeline
# firing order (see pipeline.py `_progress` calls). Each step has a key, a
# short display label, a human "current operation" description, and the
# `done_at` tick count at which the backend marks it complete.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    desc: str
    done_at: int


@dataclass(frozen=True)
class StageGroup:
    num: str
    title: str
    steps: Tuple[Stage, ...]


GROUPS = (
    StageGroup("01", "Understand", (
        Stage("video", "Video", "Reading and validating the uploaded footage.", 1),
        Stage("cricket", "Cricket Scene", "Checking the footage for a valid cricket scene.", 2),
    )),
    StageGroup("02", "Track", (
        Stage("ball_detect", "Ball", "Searching for the cricket ball across frames.", 3),
        Stage("bowler", "Bowler", "Following the selected bowler through the delivery.", 4),
        Stage("trajectory", "Trajectory", "Building the ball's frame-by-frame trajectory.", 7),
    )),
    StageGroup("03", "Analyze", (
        Stage("pose", "Pose", "Extracting body landmarks from the bowler.", 5),
        Stage("biomech", "Biomechanics", "Calculating bowling movement metrics.", 6),
    )),
    StageGroup("04", "Assess", (
        Stage("perf", "Performance", "Generating performance and risk insights.", 8),
        Stage("coaching", "Risk & Coaching", "Scoring performance, risk and coaching.", 9),
    )),
)

# Pipeline label (as fired by pipeline.py) -> tick index.
_PIPELINE_TICK_BY_LABEL = {
    "Video loaded": 1,
    "Cricket pre-check": 2,
    "Ball detection": 3,
    "Bowler detection": 4,
    "Pose extraction": 5,
    "Biomechanics": 6,
    "Ball tracking": 7,
    "ML analysis": 8,
    "Complete": 9,
}

# Human-readable "current operation" hero line for each backend stage label.
_CURRENT_OP = {
    "Video loaded": "VIDEO INGESTION",
    "Cricket pre-check": "CRICKET VALIDATION",
    "Ball detection": "BALL DETECTION",
    "Bowler detection": "BOWLER TRACKING",
    "Pose extraction": "POSE ESTIMATION",
    "Biomechanics": "BIOMECHANICS",
    "Ball tracking": "BALL TRAJECTORY",
    "ML analysis": "PERFORMANCE",
}


# ---------------------------------------------------------------------------
# Analysis UI state
# ---------------------------------------------------------------------------

@dataclass
class AnalysisState:
    phase: str = "idle"          # "idle" | "running" | "complete" | "error"
    done: int = 0
    total: int = len([1 for g in GROUPS for _ in g.steps])
    current_label: str = ""      # active / just-finished backend label
    fps: Optional[int] = None                 # real target fps (from UI)
    resize: Optional[Tuple[int, int]] = None  # real processing resolution (w,h)
    total_frames: Optional[int] = None        # real clip frame count (after ML/backing pass)
    original_dims: Optional[Tuple[int, int]] = None  # real (h, w) after processing
    bowler_track_id: Optional[int] = None
    bowler_confidence: Optional[float] = None
    error: Optional[str] = None
    elapsed_s: Optional[float] = None
    meta: dict = field(default_factory=dict)


def resolve_active_tick(label: str) -> Optional[int]:
    """Map a backend progress label to its canonical tick (1..9)."""
    return _PIPELINE_TICK_BY_LABEL.get(label)


# ---------------------------------------------------------------------------
# HTML/CSS rendering
# ---------------------------------------------------------------------------

_SHELL_CSS = """
<style>
    /* scoped under .paceai-lab to avoid collisions with the rest of the app */
    .paceai-lab {
        --lab-bg: #ffffff;
        --lab-panel: #edf3f9;
        --lab-line: #d0dce7;
        --lab-muted: #6b7a8f;
        --lab-body: #1a1c1c;
        --lab-teal: #1d546c;
        --lab-ok: #15803d;
        --lab-warn: #9a6700;
        --lab-danger: #cf222e;
        --lab-font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        --lab-mono: 'JetBrains Mono', 'Consolas', monospace;
        font-family: var(--lab-font);
        color: var(--lab-body);
        background: var(--lab-bg);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--lab-line);
        box-shadow: 0 12px 40px rgba(26,61,100,.12);
    }

    .paceai-lab * { box-sizing: border-box; }

    .lab-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 340px;
        gap: 0;
    }

    /* ---------- LEFT: video hero + current op ---------- */
    .lab-stage { position: relative; min-height: 380px; background: #edf3f9; }

    .lab-stage-bg {
        position: absolute; inset: 0;
        background:
            radial-gradient(120% 120% at 20% 0%, rgba(26,61,100,.10), transparent 55%),
            radial-gradient(100% 100% at 100% 100%, rgba(29,84,108,.08), transparent 55%),
            linear-gradient(160deg, #f4f8fc 0%, #e9f1f7 100%);
    }

    /* subtle motion-analysis stage illustration (only while no real frame) */
    .lab-stage-art { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .lab-art-field {
        position: relative; width: 60%; max-width: 520px; height: 200px;
        border: 1px solid rgba(26,61,100,.25); border-radius: 12px;
        background: repeating-linear-gradient(90deg, rgba(26,61,100,.05) 0 2px, transparent 2px calc(100%/10)),
                    linear-gradient(rgba(21,128,61,.04), transparent);
    }
    .lab-art-field::after { /* horizon / pitch line */
        content: ""; position: absolute; left: 8%; right: 8%; top: 78%;
        height: 1px; background: linear-gradient(90deg, transparent, rgba(29,84,108,.35), transparent);
    }
    .lab-art-bowler {
        position: absolute; left: 16%; bottom: 20%; width: 46px; height: 96px;
        border-radius: 46px 46px 0 0;
        background: linear-gradient(90deg, rgba(26,61,100,.22), rgba(26,61,100,.08));
        border: 1px solid rgba(26,61,100,.4);
    }
    .lab-art-ball {
        position: absolute; right: 18%; bottom: 34%; width: 16px; height: 16px;
        border-radius: 50%; background: #cf222e; border: 2px solid #e5534f;
        box-shadow: 0 0 0 3px rgba(207,34,46,.15);
    }
    .lab-scan {
        position: absolute; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--lab-teal), transparent);
        animation: labScan 2.8s ease-in-out infinite;
    }
    @keyframes labScan { 0% { top: 6%; } 50% { top: 92%; } 100% { top: 6%; } }

    /* HUD overlays */
    .lab-hud { position: absolute; left: 0; right: 0; bottom: 0; top: 0; pointer-events: none; z-index: 3; }
    .hud-corner { position: absolute; display: flex; align-items: center; gap: 8px; }
    .hud-tl { top: 12px; left: 14px; }
    .hud-tr { top: 12px; right: 14px; }
    .hud-bl { bottom: 12px; left: 14px; }
    .hud-br { bottom: 12px; right: 14px; }
    .hud-brand { display: flex; flex-direction: column; line-height: 1.1; }
    .hud-brand b { font-size: 12px; letter-spacing: .28em; color: #1a3d64; }
    .hud-brand span { font-size: 9px; letter-spacing: .22em; color: var(--lab-muted); }
    .hud-pill {
        font-family: var(--lab-mono); font-size: 10px; letter-spacing: .12em;
        padding: 4px 9px; border-radius: 5px; border: 1px solid var(--lab-line);
        background: rgba(255,255,255,.85); color: var(--lab-body); text-transform: uppercase;
        backdrop-filter: blur(2px);
    }
    .hud-pill .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
    .hud-live .dot { background: var(--lab-danger); box-shadow: 0 0 6px rgba(207,34,46,.55); animation: labPulse 1.2s infinite; }
    .hud-lock .dot { background: var(--lab-ok); box-shadow: 0 0 6px rgba(21,128,61,.5); }
    .hud-search .dot { background: var(--lab-warn); box-shadow: 0 0 6px rgba(154,103,0,.5); animation: labPulse 1.2s infinite; }
    @keyframes labPulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

    .lab-frame { font-family: var(--lab-mono); font-size: 11px; color: var(--lab-body); letter-spacing: .06em; }
    .lab-frame b { color: var(--lab-teal); }

    /* ---------- Current operation (hero line) ---------- */
    .lab-current {
        padding: 14px 20px 18px; border-top: 1px solid var(--lab-line);
        background: linear-gradient(180deg, rgba(237,243,249,.7), rgba(255,255,255,0));
    }
    .lab-current-kicker { font-size: 10px; letter-spacing: .24em; color: var(--lab-muted); font-weight: 700; }
    .lab-current-title { font-size: 22px; font-weight: 800; margin: 4px 0 2px; color: #1a3d64; letter-spacing: .01em; }
    .lab-current-desc { color: var(--lab-muted); font-size: 13px; line-height: 1.5; }

    /* thin premium progress line */
    .lab-progress { padding: 0 20px 16px; }
    .lab-progress-track {
        position: relative; height: 3px; border-radius: 3px;
        background: rgba(26,61,100,.15); overflow: hidden;
    }
    .lab-progress-fill { position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg, #1a3d64, #1d546c); width: 0%; transition: width .4s ease; }
    .lab-progress-legend { display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--lab-mono); font-size: 10px; color: var(--lab-muted); letter-spacing: .1em; }

    /* ---------- RIGHT: live analysis panel ---------- */
    .lab-panel {
        border-left: 1px solid var(--lab-line); padding: 18px 18px 14px;
        background: var(--lab-panel);
    }
    .lab-panel-title { font-size: 11px; letter-spacing: .24em; font-weight: 800; color: var(--lab-muted); display: flex; align-items: center; justify-content: space-between; }
    .lab-panel-title .st { display: inline-flex; align-items: center; gap: 7px; }
    .lab-group { margin: 16px 0 0; }
    .lab-group-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .lab-group-num { font-family: var(--lab-mono); font-size: 10px; color: var(--lab-muted); letter-spacing: .08em; }
    .lab-group-name { font-size: 11px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--lab-body); }
    .lab-group-line { flex: 1; height: 1px; background: var(--lab-line); }

    .lab-step { display: flex; align-items: center; gap: 10px; padding: 6px 0 6px 2px; }
    .lab-step .ic { display: inline-flex; width: 16px; height: 16px; flex: 0 0 16px; align-items: center; justify-content: center; font-family: var(--lab-mono); font-size: 11px; }
    .lab-step .tx { font-size: 13px; }
    .lab-step.done .ic { color: var(--lab-ok); }
    .lab-step.done .tx { color: var(--lab-muted); }
    .lab-step.active .ic { color: var(--lab-teal); animation: labFade 1.4s infinite; }
    .lab-step.active .tx { color: #1a3d64; font-weight: 600; }
    .lab-step.up .ic { color: rgba(107,122,143,.5); }
    .lab-step.up .tx { color: rgba(107,122,143,.6); }
    @keyframes labFade { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

    /* ---------- metadata strip (below stage art / inside video area) ---------- */
    .lab-meta { position: absolute; left: 14px; top: 40px; z-index: 2; display: flex; flex-direction: column; gap: 4px; }
    .lab-meta .mi { font-family: var(--lab-mono); font-size: 10px; color: var(--lab-muted); letter-spacing: .06em; }
    .lab-meta .mi b { color: var(--lab-body); font-weight: 600; }

    /* ---------- completion / error ---------- */
    .lab-done, .lab-err { padding: 26px 24px; text-align: left; }
    .lab-done .ok-ring { width: 34px; height: 34px; border-radius: 50%; border: 2px solid var(--lab-ok); display: inline-flex; align-items: center; justify-content: center; color: var(--lab-ok); margin-bottom: 10px; }
    .lab-err .ok-ring { border-color: var(--lab-danger); color: var(--lab-danger); }
    .lab-done .tt, .lab-err .tt { font-size: 20px; font-weight: 800; color: #1a3d64; }
    .lab-done .dd, .lab-err .dd { color: var(--lab-muted); font-size: 13px; margin-top: 4px; line-height: 1.5; }
    .lab-done .btn, .lab-err .btn {
        margin-top: 16px; display: inline-flex; align-items: center; gap: 8px;
        padding: 10px 18px; border-radius: 8px; border: 1px solid rgba(21,128,61,.4);
        background: linear-gradient(180deg, rgba(21,128,61,.14), rgba(21,128,61,.05));
        color: var(--lab-ok); font-weight: 700; cursor: pointer; font-family: var(--lab-font); font-size: 14px;
    }
    .lab-err .btn { border-color: rgba(207,34,46,.4); color: var(--lab-danger); background: linear-gradient(180deg, rgba(207,34,46,.10), transparent); }

    /* ---------- reduce motion ---------- */
    @media (prefers-reduced-motion: reduce) {
        .lab-scan, .lab-step.active .ic, .hud-live .dot, .hud-search .dot,
        .lab-progress-fill { animation: none !important; transition: none !important; }
    }

    /* ---------- responsive ---------- */
    @media (max-width: 900px) {
        .lab-grid { grid-template-columns: 1fr; }
        .lab-panel { border-left: 0; border-top: 1px solid var(--lab-line); }
    }
    @media (max-width: 640px) {
        .lab-grid { grid-template-columns: 1fr; }
        .lab-stage { min-height: 320px; }
        .lab-art-field { width: 84%; }
        .hud-brand b { font-size: 10px; }
        .hud-pill { font-size: 8px; padding: 3px 6px; }
    }
</style>
"""


def _esc(v) -> str:
    return "" if v is None else _html.escape(str(v))


def _active_label_idx(label: str) -> Optional[int]:
    """Tick index of the given backend label, or None if unknown."""
    return _PIPELINE_TICK_BY_LABEL.get(label)


def _stage_badges(state: AnalysisState) -> Tuple[str, str]:
    """Return (top-right pill html, bottom-left pill html)."""
    active_idx = _active_label_idx(state.current_label)
    locking = state.bowler_track_id is not None
    if state.phase == "running":
        top = (
            f'<span class="hud-pill hud-live"><span class="dot"></span>ANALYSIS ACTIVE</span>'
            if active_idx is not None else
            f'<span class="hud-pill hud-live"><span class="dot"></span>ANALYSIS ACTIVE</span>'
        )
        if state.bowler_track_id is not None:
            bot = f'<span class="hud-pill hud-lock"><span class="dot"></span>BOWLER TRACK · LOCKED</span>'
        else:
            bot = f'<span class="hud-pill hud-search"><span class="dot"></span>BOWLER TRACK · SEARCHING</span>'
        return top, bot
    if state.phase == "complete":
        return (
            f'<span class="hud-pill hud-lock"><span class="dot"></span>ANALYSIS COMPLETE</span>',
            f'<span class="hud-pill hud-lock"><span class="dot"></span>BOWLER TRACK · LOCKED</span>'
            if state.bowler_track_id is not None else
            f'<span class="hud-pill"><span class="dot" style="background:var(--lab-warn)"></span>BOWLER TRACK · UNLOCKED</span>',
        )
    return (
        f'<span class="hud-pill hud-warn"><span class="dot" style="background:var(--lab-danger)"></span>ANALYSIS INTERRUPTED</span>',
        f'<span class="hud-pill">STATUS UNKNOWN</span>',
    )


def _render_stage_art() -> str:
    return (
        '<div class="lab-stage-art" aria-hidden="true">'
        '  <div class="lab-art-field">'
        '    <div class="lab-scan"></div>'
        '    <div class="lab-art-bowler"></div>'
        '    <div class="lab-art-ball"></div>'
        '  </div>'
        '</div>'
    )


def _meta_strip(state: AnalysisState) -> str:
    items = []
    if state.fps:
        items.append(f'<span class="mi">FPS <b>{_esc(state.fps)}</b></span>')
    if state.resize:
        items.append(f'<span class="mi">RES <b>{_esc(state.resize[0])}×{_esc(state.resize[1])}</b></span>')
    if state.total_frames:
        items.append(f'<span class="mi">FRAMES <b>{_esc(state.total_frames)}</b></span>')
    if state.elapsed_s is not None:
        try:
            _el = f"{float(state.elapsed_s):.1f}s"
        except (TypeError, ValueError):
            _el = str(state.elapsed_s)
        items.append(f'<span class="mi">ELAPSED <b>{_esc(_el)}</b></span>')
    if not items:
        return ""
    return '<div class="lab-meta">' + "".join(items) + "</div>"


def _render_group_html(group: StageGroup, done: int, active_idx: Optional[int]) -> str:
    rows = []
    for step in group.steps:
        if active_idx is None:
            cls = "done" if step.done_at <= done else "up"
        else:
            cls = "done" if step.done_at < active_idx else ("active" if step.done_at == active_idx else "up")
        icon = "✓" if cls == "done" else ("●" if cls == "active" else "○")
        rows.append(
            f'<div class="lab-step {cls}">'
            f'<span class="ic" aria-hidden="true">{icon}</span>'
            f'<span class="tx">{_esc(step.label)}</span>'
            f"</div>"
        )
    head = (
        f'<div class="lab-group-head">'
        f'<span class="lab-group-num">{_esc(group.num)}</span>'
        f'<span class="lab-group-name">{_esc(group.title)}</span>'
        f'<span class="lab-group-line"></span>'
        f"</div>"
    )
    return f'<div class="lab-group">{head}{"".join(rows)}</div>'


def _render_panel(state: AnalysisState) -> str:
    active_idx = _active_label_idx(state.current_label)
    groups = "".join(_render_group_html(g, state.done, active_idx) for g in GROUPS)
    status_icon = "●" if state.phase == "running" else "✓"
    status_text = "RUNNING" if state.phase == "running" else "STOPPED"
    return (
        '<div class="lab-panel" role="region" aria-label="Live analysis stages">'
        f'<div class="lab-panel-title"><span class="st"><span style="color:var(--lab-teal)">{status_icon}</span>'
        f'LIVE ANALYSIS</span><span style="font-family:var(--lab-mono);font-size:10px">{status_text}</span></div>'
        f"{groups}"
        "</div>"
    )


def _current_op_html(state: AnalysisState, desc: str) -> str:
    op = _CURRENT_OP.get(state.current_label, "ANALYSIS")
    return (
        '<div class="lab-current">'
        f'<div class="lab-current-kicker">CURRENT OPERATION</div>'
        f'<div class="lab-current-title">{_esc(op)}</div>'
        f'<div class="lab-current-desc">{_esc(desc)}</div>'
        "</div>"
    )


def _progress_html(state: AnalysisState) -> str:
    pct = int(round(state.done / max(1, state.total) * 100))
    shown = f"{pct}% · STAGE {state.done}/{state.total}" if state.phase == "running" else "ANALYSIS ACTIVE"
    return (
        '<div class="lab-progress" role="progressbar" aria-valuenow="'
        f'{state.done}" aria-valuemin="0" aria-valuemax="{state.total}" aria-label="Analysis progress">'
        '<div class="lab-progress-track"><div class="lab-progress-fill" style="width:'
        f'{pct}%"></div></div>'
        f'<div class="lab-progress-legend"><span>{_esc(shown)}</span><span>REAL FRAME DATA ONLY</span></div>'
        "</div>"
    )


def _find_active_desc(state: AnalysisState) -> str:
    if state.phase != "running" or state.current_label == "Complete":
        # By the time we show a completion/error state, describe the final result.
        return "Finalizing the delivery analysis."
    for group in GROUPS:
        for step in group.steps:
            if step.done_at == (_active_label_idx(state.current_label) or 0):
                return step.desc
    return "Processing the delivery."


# ---------------------------------------------------------------------------
# "Players Detected" broadcast roster (presentation results panel)
# ---------------------------------------------------------------------------

# Broadcast role palette -- kept in sync with analysis_replay._ROLE_COLORS
# (BGR) so the roster matches the replay overlays exactly.
_ROSTER_COLORS = {
    "bowler": "#ffc861",
    "batsman": "#ff4d4d",
    "wicketkeeper": "#ffc800",
    "umpire": "#00d7d7",
    "fielder": "#b4b4b4",
    "unknown": "#8b949e",
}

_ROSTER_CSS = """
<style>
    .paceai-roster {
        background: var(--lab-panel, #edf3f9);
        border: 1px solid var(--lab-line, #d0dce7);
        border-radius: 14px;
        padding: 18px 20px 14px;
        margin: 18px 0 6px;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    .paceai-roster-head {
        display: flex; align-items: baseline; justify-content: space-between;
        gap: 12px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .paceai-roster-kicker {
        font-size: 11px; font-weight: 800; letter-spacing: .2em;
        color: #1d546c; text-transform: uppercase;
    }
    .paceai-roster-note {
        font-size: 11px; color: #6b7a8f;
    }
    .paceai-roster-row {
        display: grid; grid-template-columns: 12px 150px 60px minmax(0,1fr) 56px;
        gap: 10px; align-items: center;
        padding: 7px 4px; border-radius: 8px;
        font-size: 13px;
    }
    .paceai-roster-row + .paceai-roster-row { border-top: 1px solid rgba(208,220,231,.7); }
    .paceai-roster-row.bowler-row { background: rgba(255,200,97,.14); }
    .paceai-roster-dot {
        width: 10px; height: 10px; border-radius: 50%; display: inline-block;
        box-shadow: 0 0 6px currentColor; justify-self: center;
    }
    .paceai-roster-role { font-weight: 800; color: #1a1c1c; letter-spacing: .04em; }
    .paceai-roster-track {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 12px; color: #6b7a8f;
    }
    .paceai-roster-bar {
        height: 5px; border-radius: 3px; background: rgba(26,61,100,.12);
        overflow: hidden;
    }
    .paceai-roster-bar i {
        display: block; height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #1a3d64, currentColor);
    }
    .paceai-roster-conf {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 12px; font-weight: 700; color: #1a3d64; text-align: right;
    }
    .paceai-roster-empty {
        padding: 14px 6px 10px; color: #6b7a8f; font-size: 13px; line-height: 1.5;
    }
    @media (max-width: 640px) {
        .paceai-roster-row { grid-template-columns: 12px 1fr 56px; }
        .paceai-roster-track, .paceai-roster-bar { display: none; }
    }
</style>
"""


def render_role_roster(player_roles: Optional[dict] = None,
                       bowler_track_id: Optional[int] = None,
                       bowler_confidence: Optional[float] = None) -> str:
    """HTML for the "Players Detected" broadcast roster panel.

    Shows the locked bowler first (the analysis subject), then every
    classified non-bowler player (batsman / wicketkeeper / umpire / fielder),
    each with its cricket-evidence confidence. Colors match the Analysis
    Replay overlays (see analysis_replay._ROLE_COLORS).

    Returns an empty string when there is no role data at all (no bowler lock
    and no player roles) so callers can hide the panel honestly.
    """
    if not player_roles and bowler_track_id is None:
        return ""

    rows = []
    # --- Bowler (the analysis subject) is always first and highlighted. ---
    if bowler_track_id is not None:
        conf = bowler_confidence if bowler_confidence is not None else None
        rows.append(_roster_row("BOWLER", f"#{bowler_track_id}", conf,
                                _ROSTER_COLORS["bowler"], prominent=True))

    # --- Non-bowler roles, sorted by confidence (highest first). ---
    if player_roles:
        ordered = sorted(
            player_roles.items(),
            key=lambda kv: (kv[1].get("confidence") if isinstance(kv[1], dict)
                            else 0.0) or 0.0,
            reverse=True,
        )
        for tid, info in ordered:
            role = info.get("role", "unknown") if isinstance(info, dict) else "unknown"
            conf = info.get("confidence") if isinstance(info, dict) else None
            rows.append(_roster_row(role.upper(), f"#{tid}", conf,
                                    _ROSTER_COLORS.get(role, _ROSTER_COLORS["unknown"])))

    if not rows:
        return ""

    note = "Cricket-evidence classification, matching the overlay labels"
    body = "".join(rows)
    return (
        _ROSTER_CSS
        + '<div class="paceai-roster" data-testid="paceai-roster">'
        + '<div class="paceai-roster-head">'
        + '<span class="paceai-roster-kicker">Players Detected</span>'
        + f'<span class="paceai-roster-note">{_esc(note)}</span>'
        + "</div>"
        + body
        + "</div>"
    )


def _roster_row(label: str, track_tag: str, confidence: Optional[float],
                color: str, prominent: bool = False) -> str:
    pct = None
    if confidence is not None:
        try:
            pct = max(0.0, min(1.0, float(confidence))) * 100.0
        except (TypeError, ValueError):
            pct = None
    if pct is None:
        conf_text = "n/a"
        width = 0
    else:
        conf_text = f"{pct:.0f}%"
        width = pct
    row_cls = "paceai-roster-row" + (" bowler-row" if prominent else "")
    return (
        f'<div class="{row_cls}">'
        f'<span class="paceai-roster-dot" style="background:{color};color:{color}"></span>'
        f'<span class="paceai-roster-role">{_esc(label)}</span>'
        f'<span class="paceai-roster-track">{_esc(track_tag)}</span>'
        f'<span class="paceai-roster-bar"><i style="width:{width:.0f}%;color:{color}"></i></span>'
        f'<span class="paceai-roster-conf">{_esc(conf_text)}</span>'
        "</div>"
    )


def render_lab_html(state: AnalysisState, show_art: bool = True) -> str:
    """Return the full HTML markup for the current analysis screen state."""
    top_pill, bot_pill = _stage_badges(state)
    art = _render_stage_art() if show_art else ""

    if state.phase == "complete":
        left = (
            f'<div class="lab-stage"><div class="lab-stage-bg"></div>{art}'
            f'<div class="lab-hud">'
            f'<div class="hud-corner hud-tl"><div class="hud-brand"><b>PACEAI</b><span>COMPUTER VISION</span></div></div>'
            f'<div class="hud-corner hud-tr">{top_pill}</div>'
            f'{_meta_strip(state)}'
            f'</div></div>'
            '<div class="lab-done">'
            '<div class="ok-ring">✓</div>'
            '<div class="tt">ANALYSIS COMPLETE</div>'
            '<div class="dd">Your bowling delivery was analyzed.</div>'
            "</div>"
        )
        right = _render_panel(state)
    elif state.phase == "error":
        left = (
            f'<div class="lab-stage"><div class="lab-stage-bg"></div>{art}'
            f'<div class="lab-hud">'
            f'<div class="hud-corner hud-tl"><div class="hud-brand"><b>PACEAI</b><span>COMPUTER VISION</span></div></div>'
            f'<div class="hud-corner hud-tr">{top_pill}</div>'
            f'{_meta_strip(state)}'
            f"</div></div>"
            '<div class="lab-err">'
            '<div class="ok-ring">✕</div>'
            '<div class="tt">ANALYSIS INTERRUPTED</div>'
            '<div class="dd">We couldn\'t complete the delivery analysis. View technical details below for the reason.</div>'
            "</div>"
        )
        right = _render_panel(state)
    else:
        left = (
            f'<div class="lab-stage"><div class="lab-stage-bg"></div>{art}'
            f'<div class="lab-hud">'
            f'<div class="hud-corner hud-tl"><div class="hud-brand"><b>PACEAI</b><span>COMPUTER VISION</span></div></div>'
            f'<div class="hud-corner hud-tr">{top_pill}</div>'
            f'<div class="hud-corner hud-bl">{bot_pill}</div>'
            f'<div class="hud-corner hud-br"><span class="lab-frame"><b>ANALYSIS</b> ACTIVE</span></div>'
            f'{_meta_strip(state)}'
            f"</div></div>"
            f"{_current_op_html(state, _find_active_desc(state))}"
            f"{_progress_html(state)}"
        )
        right = _render_panel(state)

    return (
        _SHELL_CSS +
        f'<div class="paceai-lab" data-testid="paceai-lab">'
        f"<div class=\"lab-grid\">{left}{right}</div>"
        f"</div>"
    )
