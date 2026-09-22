"""
PaceAI Kinetic Biomechanics Workspace UI
========================================
Presentation-only HTML builders for the post-analysis screen, mirroring the
"CRICKET-CV v4.2" prototype layout (docs/DESIGN.md):

  * breadcrumb workspace header with run chips
  * video viewport chrome (real player rendered by caller)
  * 75/25 columns: viewport + five-phase timeline (left), telemetry rail (right)
  * bottom delivery reel of the current run

Each builder returns a fully self-contained HTML block so it can be injected
through separate st.markdown / st.columns calls. No pipeline, ML, or storage
logic lives here -- callers supply already-computed values.
"""

import html as _html


SURFACE = "#f4f4f4"
PANEL = "#edf3f9"
LINE = "#d0dce7"
BORDER_STRONG = "#b7c8db"
NAVY = "#1a3d64"
TEAL = "#1d546c"
BODY = "#1a1c1c"
SECONDARY = "#495867"
MUTED = "#6b7a8f"
OUTLINE = "#788ca0"
ICE = "#d9eafd"
OK = "#15803d"
WARN = "#9a6700"
DANGER = "#cf222e"


def esc(value) -> str:
    return "" if value is None else _html.escape(str(value))


def _fmt(value, suffix="", digits=1, dash="\u2014"):
    if value is None:
        return dash
    try:
        value = float(value)
    except (TypeError, ValueError):
        return esc(value)
    return f"{value:.{digits}f}{suffix}"


KINETIC_CSS = """
<style>
    :root {
        --k-surface: %(SURFACE)s; --k-panel: %(PANEL)s; --k-line: %(LINE)s;
        --k-body: %(BODY)s; --k-secondary: %(SECONDARY)s; --k-muted: %(MUTED)s;
        --k-nav: %(NAVY)s; --k-teal: %(TEAL)s; --k-ice: %(ICE)s;
    }
    .kin-ws {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        margin: 20px 0 6px;
    }
    .kin-ws * { box-sizing: border-box; }

    /* ---------- breadcrumb / workspace header ---------- */
    .kin-breadcrumb {
        display: flex; align-items: flex-end; justify-content: space-between;
        gap: 16px; flex-wrap: wrap;
        padding: 16px 20px; border-radius: 14px 14px 0 0;
        background: linear-gradient(135deg, #ffffff 0%%, #eef3f9 100%%);
        border: 1px solid var(--k-line); border-bottom: 0;
    }
    .kin-eyebrow { color: var(--k-teal); font-size: .66rem; letter-spacing: .18em;
        font-weight: 800; text-transform: uppercase; }
    .kin-title { color: var(--k-nav); font-size: 1.45rem; font-weight: 800;
        margin: 3px 0 2px; letter-spacing: -.01em; }
    .kin-breadcrumb p { margin: 0; color: var(--k-muted); font-size: .82rem; }
    .kin-chips { display: flex; gap: 7px; flex-wrap: wrap; }
    .kin-chip {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: .64rem; font-weight: 700; letter-spacing: .08em;
        color: var(--k-nav); background: rgba(217, 234, 253, .55);
        border: 1px solid var(--k-line); border-radius: 999px;
        padding: 4px 10px; text-transform: uppercase; white-space: nowrap;
    }
    .kin-chip.kin-ok { color: var(--k-ok, %(OK)s); background: rgba(21,128,61,.08); border-color: rgba(21,128,61,.35); }
    .kin-chip.kin-amber { color: %(WARN)s; background: rgba(154,103,0,.08); border-color: rgba(154,103,0,.35); }
    .kin-chip.kin-red { color: %(DANGER)s; background: rgba(207,34,46,.08); border-color: rgba(207,34,46,.35); }

    /* ---------- video viewport chrome ---------- */
    .kin-viewport {
        background: #ffffff; border: 1px solid var(--k-line);
        border-radius: 10px; overflow: hidden; margin-bottom: 10px;
    }
    .kin-vp-hud { display: flex; align-items: center; justify-content: space-between;
        gap: 10px; padding: 7px 10px; border-bottom: 1px solid var(--k-line);
        font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: .62rem;
        color: var(--k-muted); letter-spacing: .1em; flex-wrap: wrap; }
    .kin-vp-hud b { color: var(--k-nav); }
    .kin-vp-hud .kin-vp-live { color: %(DANGER)s; }

    /* ---------- telemetry rail ---------- */
    .kin-rail { display: flex; flex-direction: column; gap: 10px; }
    .kin-card { background: #ffffff; border: 1px solid var(--k-line);
        border-radius: 10px; padding: 12px 13px; }
    .kin-card-kicker { font-size: .6rem; letter-spacing: .18em; font-weight: 800;
        text-transform: uppercase; color: var(--k-muted); margin-bottom: 6px; }
    .kin-identity { display: flex; align-items: center; gap: 10px; }
    .kin-avatar { width: 36px; height: 36px; border-radius: 50%%;
        background: var(--k-ice); color: var(--k-nav); display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: .85rem;
        border: 1px solid var(--k-line); }
    .kin-identity-name { font-weight: 800; color: var(--k-body); font-size: .85rem; }
    .kin-identity-sub { color: var(--k-muted); font-size: .68rem; }
    .kin-identity .kin-chip { margin-left: auto; }

    .kin-kpis { display: flex; flex-direction: column; gap: 7px; }
    .kin-kpi { display: flex; align-items: baseline; justify-content: space-between;
        gap: 10px; padding: 9px 11px; border-radius: 8px;
        background: var(--k-panel); border: 1px solid var(--k-line); }
    .kin-kpi-meta { min-width: 0; }
    .kin-kpi-label { color: var(--k-secondary); font-size: .62rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase; }
    .kin-kpi-sub { color: var(--k-muted); font-size: .6rem; margin-top: 1px; }
    .kin-kpi-val { font-family: 'JetBrains Mono', 'Consolas', monospace;
        color: var(--k-nav); font-size: 1rem; font-weight: 800; white-space: nowrap; }
    .kin-kpi-val small { color: var(--k-muted); font-weight: 600; font-size: .6rem; }

    /* ---------- load donut ---------- */
    .kin-load { display: flex; align-items: center; gap: 14px; }
    .kin-donut { --p: 0; --ring: %(OK)s;
        width: 68px; height: 68px; border-radius: 50%%; flex: 0 0 68px;
        background: conic-gradient(var(--ring) calc(var(--p) * 1%%), rgba(26,61,100,.12) 0);
        display: flex; align-items: center; justify-content: center; position: relative; }
    .kin-donut::after { content: ""; position: absolute; inset: 8px; border-radius: 50%%; background: #ffffff; }
    .kin-donut span { position: relative; z-index: 1; font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: .78rem; font-weight: 800; color: var(--k-nav); }
    .kin-load-meta { min-width: 0; }
    .kin-load-title { font-size: .72rem; font-weight: 800; color: var(--k-body); }
    .kin-load-note { color: var(--k-muted); font-size: .62rem; line-height: 1.45; }

    /* ---------- timeline ---------- */
    .kin-timeline { padding: 14px 20px 16px; border: 1px solid var(--k-line);
        border-radius: 10px; background: #ffffff; }
    .kin-tl-track { position: relative; height: 6px; border-radius: 3px;
        background: rgba(26,61,100,.12); margin: 0 10px; }
    .kin-tl-marker { position: absolute; top: -4px; width: 14px; height: 14px;
        border-radius: 50%%; background: #ffffff; border: 3px solid var(--k-nav);
        box-shadow: 0 0 0 3px rgba(26,61,100,.15); }
    .kin-tl-marker.release { border-color: %(DANGER)s; box-shadow: 0 0 0 3px rgba(207,34,46,.15); }
    .kin-tl-marker.impact { border-color: %(OK)s; box-shadow: 0 0 0 3px rgba(21,128,61,.15); }
    .kin-phases { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 12px; }
    .kin-phase { text-align: center; padding: 8px 4px; border-radius: 8px;
        border: 1px solid var(--k-line); background: #ffffff; }
    .kin-phase.active { border-color: var(--k-nav); background: var(--k-ice); }
    .kin-phase-name { font-size: .58rem; font-weight: 800; letter-spacing: .1em;
        color: var(--k-secondary); text-transform: uppercase; }
    .kin-phase.active .kin-phase-name { color: var(--k-nav); }
    .kin-phase-time { font-family: 'JetBrains Mono', 'Consolas', monospace;
        color: var(--k-nav); font-size: .8rem; font-weight: 800; margin-top: 3px; }
    .kin-phase-sub { color: var(--k-muted); font-size: .55rem; margin-top: 1px; }

    /* ---------- delivery reel ---------- */
    .kin-reel { display: flex; align-items: center; gap: 8px; padding: 10px 20px 14px;
        border: 1px solid var(--k-line); border-radius: 10px;
        flex-wrap: wrap; }
    .kin-reel-label { font-size: .6rem; font-weight: 800; letter-spacing: .14em;
        text-transform: uppercase; color: var(--k-muted); margin-right: 4px; }
    .kin-thumb { width: 74px; height: 42px; border-radius: 6px;
        background: linear-gradient(135deg, var(--k-ice), #ffffff);
        border: 1px solid var(--k-line); position: relative; overflow: hidden; }
    .kin-thumb i { position: absolute; left: 6px; right: 6px; height: 2px;
        background: var(--k-nav); top: 50%%; transform: rotate(-12deg); }
    .kin-thumb b { position: absolute; bottom: 3px; left: 5px; font-size: .52rem;
        color: var(--k-nav); font-family: 'JetBrains Mono', 'Consolas', monospace; }
    .kin-reel .kin-chip { background: #ffffff; }

    @media (max-width: 900px) {
        .kin-phases { grid-template-columns: 1fr; }
    }
</style>
""" % dict(
    SURFACE=SURFACE, PANEL=PANEL, LINE=LINE, BORDER_STRONG=BORDER_STRONG,
    NAVY=NAVY, TEAL=TEAL, BODY=BODY, SECONDARY=SECONDARY, MUTED=MUTED,
    OUTLINE=OUTLINE, ICE=ICE, OK=OK, WARN=WARN, DANGER=DANGER,
)


def header_html(meta: dict) -> str:
    chips = []
    if meta.get("track") is not None:
        chips.append(f'<span class="kin-chip">TRACK&nbsp;#{esc(meta["track"])}</span>')
    if meta.get("duration_s") is not None:
        chips.append(f'<span class="kin-chip">CLIP&nbsp;{_fmt(meta["duration_s"])}s</span>')
    if meta.get("frames") is not None:
        chips.append(f'<span class="kin-chip">FRAMES&nbsp;{esc(meta["frames"])}</span>')
    if meta.get("fps") is not None:
        chips.append(f'<span class="kin-chip">{esc(meta["fps"])}&nbsp;FPS</span>')
    quality = meta.get("quality")
    if quality:
        q_cls = "kin-ok" if quality == "HIGH" else "kin-amber" if quality == "MODERATE" else "kin-red"
        chips.append(f'<span class="kin-chip {q_cls}">TRACK&nbsp;Q&nbsp;{esc(quality)}</span>')
    return f"""
    <div class="kin-ws" id="paceai-analysis-replay" aria-label="PaceAI kinetic analysis workspace">
      <div class="kin-breadcrumb">
        <div>
          <div class="kin-eyebrow">PACEAI / DELIVERY ANALYSIS</div>
          <div class="kin-title">Kinetic Analysis Workspace</div>
          <p>One synchronized view of the bowling action, ball path and pose evidence.</p>
        </div>
        <div class="kin-chips">{''.join(chips)}</div>
      </div>
    </div>
    """


def viewport_html(meta: dict) -> str:
    right = []
    if meta.get("quality"):
        right.append(f'<b>TRACKING {esc(meta["quality"])}</b>')
    if meta.get("events_n") is not None:
        right.append(f"{esc(meta['events_n'])} KEY EVENTS")
    if meta.get("n_det") is not None:
        right.append(f"BALL&nbsp;{esc(meta['n_det'])}&nbsp;DET")
    left = '<span class="kin-vp-live">\u25cf REC</span>' if meta.get("armed") is not None and meta.get("armed") else '<b>CAM-01</b>'
    return f"""
    <div class="kin-viewport">
      <div class="kin-vp-hud">
        <span>{left}&nbsp;\u00b7&nbsp;DATA&nbsp;LOCKED</span>
        <span>{' &nbsp;\u00b7&nbsp; '.join(right)}</span>
      </div>
    </div>
    """


def timeline_html(moments, duration_s=None, release_t=None, impact_t=None) -> str:
    def _match(*keys):
        for m in moments or []:
            label = str(m.get("label") or "").lower()
            if any(k in label for k in keys):
                return m.get("time_s")
        return None

    approach = _match("approach", "back foot", "backfoot", "punt", "bound", "start")
    bfc = _match("bfc", "back foot contact", "collection", "count-load")
    ffc = _match("ffc", "front foot", "pre-delivery", "stride")
    release = release_t if release_t is not None else _match("release", "ball release")
    impact = impact_t if impact_t is not None else _match("impact")

    def _pos(t):
        if t is None or not duration_s:
            return None
        return max(0.0, min(100.0, float(t) / float(duration_s) * 100.0))

    def _cell(name, t, sub, active=False):
        t_txt = _fmt(t, "s") if t is not None else "\u2014"
        active_cls = " active" if active else ""
        return (f'<div class="kin-phase{active_cls}">'
                f'<div class="kin-phase-name">{esc(name)}</div>'
                f'<div class="kin-phase-time">{t_txt}</div>'
                f'<div class="kin-phase-sub">{esc(sub)}</div></div>')

    markers = []
    for tag, t in (("kin-tl-marker release", release),
                   ("kin-tl-marker impact", impact)):
        if t is not None and duration_s:
            markers.append(f'<div class="{tag}" style="left:{_pos(t):.2f}%"></div>')
    marker_html = "".join(markers)

    return f"""
    <div class="kin-timeline" aria-label="Key event timeline">
      <div class="kin-tl-track">
        {marker_html}
      </div>
      <div class="kin-phases">
        {_cell("APPROACH", approach, "run-up", active=approach is not None)}
        {_cell("BFC", bfc, "pivot")}
        {_cell("FFC", ffc, "stride")}
        {_cell("RELEASE", release, "ball", active=release is not None)}
        {_cell("FOLLOW\u2013THROUGH", impact, "impact", active=impact is not None)}
      </div>
    </div>
    """


def rail_html(meta: dict) -> str:
    fv = meta.get("features") or {}
    risk_pct = meta.get("risk_pct")
    risk_level = str(meta.get("risk_level") or "low").lower()

    ring = OK
    if risk_pct is not None:
        ring = DANGER if risk_pct >= 70 else WARN if risk_pct >= 40 else OK

    def _kpi(label, value, unit, sub="estimated from pose"):
        v = _fmt(value, "")
        unit_h = f"<small>{esc(unit)}</small>" if unit else ""
        return (f'<div class="kin-kpi">'
                f'<div class="kin-kpi-meta"><div class="kin-kpi-label">{esc(label)}</div>'
                f'<div class="kin-kpi-sub">{esc(sub)}</div></div>'
                f'<div class="kin-kpi-val">{v}{unit_h}</div></div>')

    identity = ""
    if meta.get("track") is not None or meta.get("role"):
        sub = f"confidence {_fmt(meta.get('conf'), '', 2)}" if meta.get("conf") is not None else "identity locked"
        role = str(meta.get("role") or "bowler").upper()
        avatar = esc(role[:1])
        identity = f"""
        <div class="kin-card">
          <div class="kin-card-kicker">Bowler Identity</div>
          <div class="kin-identity">
            <div class="kin-avatar">{avatar}</div>
            <div>
              <div class="kin-identity-name">Track #{esc(meta.get('track') or '\u2014')}</div>
              <div class="kin-identity-sub">{esc(sub)}</div>
            </div>
            <span class="kin-chip">{esc(role)}</span>
          </div>
        </div>"""

    kpis = [
        ("Release Angular Velocity", fv.get("angular_velocity_deg_s"), "\u00b0/s",
         "shoulder-rotation rate (release-speed proxy)"),
        ("Front-Leg Bracing Angle", (180 - fv["knee_flexion_deg"]) if fv.get("knee_flexion_deg") is not None else None, "\u00b0",
         "\u2248180\u00b0 \u2212 front-knee flexion"),
        ("SH\u2013Pelvis Separation", fv.get("hip_rotation_deg"), "\u00b0",
         "pelvic counter-rotation surrogate"),
        ("Trunk Lateral Flexion", fv.get("trunk_lean_deg"), "\u00b0",
         "at release"),
        ("Release Inclination", fv.get("release_angle_deg"), "\u00b0",
         "release angle vs vertical"),
    ]
    kpi_html = "".join(_kpi(*k) for k in kpis)

    donut = ""
    if risk_pct is not None:
        donut = f"""
        <div class="kin-card">
          <div class="kin-card-kicker">Block Load Tolerance</div>
          <div class="kin-load">
            <div class="kin-donut" style="--p:{_fmt(risk_pct, '', 0)}; --ring:{ring}">
              <span>{_fmt(risk_pct, '', 0)}%</span>
            </div>
            <div class="kin-load-meta">
              <div class="kin-load-title">{esc(risk_level.upper())} risk indicator</div>
              <div class="kin-load-note">Demo-model probability of high biomechanical load for this delivery.</div>
            </div>
          </div>
        </div>"""

    return f"""
    <div class="kin-rail">
      {identity}
      <div class="kin-card">
        <div class="kin-card-kicker">Keyframe Telemetry</div>
        <div class="kin-kpis">{kpi_html}</div>
      </div>
      {donut}
      <div class="kin-card">
        <div class="kin-card-kicker">Screening Note</div>
        <div class="kin-load-note">
          Values are pose-derived estimates &mdash; not instrumented sensors.
          Reference only; see Deep-Dive Diagnostics below for the full report export.
        </div>
      </div>
    </div>
    """


def reel_html(meta: dict) -> str:
    chips = []
    if meta.get("n_det") is not None:
        chips.append(f'<span class="kin-chip">BALL&nbsp;{esc(meta["n_det"])}&nbsp;DET</span>')
    if meta.get("n_pred") is not None:
        chips.append(f'<span class="kin-chip">INT&nbsp;{esc(meta["n_pred"])}</span>')
    if meta.get("coverage_pct") is not None:
        cov = meta["coverage_pct"]
        cls = "kin-ok" if cov >= 70 else "kin-amber" if cov >= 40 else "kin-red"
        chips.append(f'<span class="kin-chip {cls}">COVERAGE&nbsp;{_fmt(cov, "%", 0)}</span>')
    if meta.get("release_idx") is not None:
        chips.append(f'<span class="kin-chip">RELEASE&nbsp;F&nbsp;{esc(meta["release_idx"])}</span>')
    if meta.get("impact_idx") is not None:
        chips.append(f'<span class="kin-chip">IMPACT&nbsp;F&nbsp;{esc(meta["impact_idx"])}</span>')
    if meta.get("outcome"):
        chips.append(f'<span class="kin-chip">{esc(meta["outcome"])}</span>')

    return f"""
    <div class="kin-reel" aria-label="Delivery reel">
      <span class="kin-reel-label">Delivery Reel</span>
      <div class="kin-thumb"><i></i><b>THIS&nbsp;DEL</b></div>
      {''.join(chips)}
    </div>
    """