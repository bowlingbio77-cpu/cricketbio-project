"""
PaceAI result interpretation & presentation layer.

Reads an existing ``pipeline.AnalysisResult`` (plus the two model bundles and the
optional video artefacts) and renders it as an adaptive result page:

    01  Delivery Status        can I trust this analysis?
    02  What this tells you    one main finding / why it matters / what to work on
    03  Top Findings           max 3, each title / observed / why / action
    04  Analysis Replay        delegated to the existing replay renderer
    05  Key Measurements       grouped, each with value / unit / context / source type
    06  Why this was flagged   finding -> measurement -> reference -> interpretation -> action
    07  Coaching               priority / what / why / how to practise
    08  Performance Indicator  value + band + model context + interval
    09  Risk Indicators        screening context, never a diagnosis
    10  Advanced Analysis      collapsed technical layer
    11  History / Export       utilities only

DESIGN CONTRACT (do not violate):

* This module NEVER computes a new biomechanical quantity, never re-derives a
  threshold, and never mutates the result. Every number shown originates in
  ``pipeline``/``ml_models``/``injury_knowledge_base``/``coaching``.
* An absent measurement is rendered as "not measured". It is NEVER rendered as
  ``0``, ``0.0``, a default, a pass, or an empty gauge. ``fabricate()`` does not
  exist here on purpose.
* "DEMO" is never upgraded to "validated", "screening" never to "diagnosis",
  "reference" never to "official decision". Validity gates are shown BEFORE any
  derived interpretation, and an unrepresentable output is withheld explicitly
  rather than approximated.
* Everything demoted out of the primary path keeps its data in the export JSON.

The three functions ``assess_delivery`` / ``build_findings`` / ``build_measurements``
are pure (no ``st`` calls) so the importance logic is unit-testable in isolation.
"""
from __future__ import annotations

import html as _html_std
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import streamlit as st

from . import config
from . import injury_knowledge_base as injury_kb

# ---------------------------------------------------------------------------
# Theme tokens (Phase 10). Scoped under .pai-* so they cannot collide with the
# existing .metric-card / .kin-* / .lab-* stylesheets in this app.
# ---------------------------------------------------------------------------
PAI_BG = "#070A0F"
PAI_SURFACE = "#0D121A"
PAI_RAISED = "#141B25"
PAI_ACCENT = "#20D9FF"
PAI_TEXT = "#F4F7FA"
PAI_MUTED = "#7F8B99"
PAI_OK = "#43D9a3"
PAI_WARN = "#E8B34A"
PAI_DANGER = "#FF7086"
PAI_LINE = "#1E2733"

RESULT_CSS = f"""
<style>
  .pai-root {{
      --pai-bg:{PAI_BG}; --pai-surface:{PAI_SURFACE}; --pai-raised:{PAI_RAISED};
      --pai-accent:{PAI_ACCENT}; --pai-text:{PAI_TEXT}; --pai-muted:{PAI_MUTED};
      --pai-ok:{PAI_OK}; --pai-warn:{PAI_WARN}; --pai-danger:{PAI_DANGER};
      --pai-line:{PAI_LINE};
  }}

  /* ---- type scale: one display tier, then section, body, metadata.
         Weight is spent on the display line and section titles only; every
         other level separates with size, spacing and colour. ---- */
  .pai-display {{
      font-size:clamp(1.95rem,4.2vw,3.15rem); font-weight:700; line-height:1.06;
      letter-spacing:-.028em; color:var(--pai-text); margin:0 0 16px; max-width:24ch;
  }}
  .pai-display.tight {{ font-size:clamp(1.45rem,2.8vw,2rem); max-width:34ch; }}
  .pai-eyebrow {{
      font-size:.63rem; letter-spacing:.19em; font-weight:600; text-transform:uppercase;
      color:var(--pai-muted); margin:0 0 9px;
  }}
  .pai-eyebrow.accent {{ color:var(--pai-accent); }}
  .pai-lede {{
      font-size:1.04rem; font-weight:400; line-height:1.58; color:var(--pai-text);
      margin:0 0 10px; max-width:60ch;
  }}
  .pai-body {{ font-size:.9rem; font-weight:400; line-height:1.62; color:var(--pai-muted); margin:0 0 9px; max-width:64ch; }}

  /* ---- 01 validity: a quiet line above a rule, never a box ---- */
  .pai-state {{
      display:flex; gap:12px 20px; align-items:baseline; flex-wrap:wrap;
      padding:0 0 16px; border-bottom:1px solid var(--pai-line);
  }}
  .pai-state-word {{
      flex:0 0 auto; font-size:.72rem; letter-spacing:.17em; font-weight:700;
      text-transform:uppercase; white-space:nowrap;
  }}
  .pai-state-word.ok {{ color:var(--pai-ok); }}
  .pai-state-word.warn {{ color:var(--pai-warn); }}
  .pai-state-word.danger {{ color:var(--pai-danger); }}
  .pai-state-copy {{ flex:1 1 340px; font-size:.86rem; line-height:1.56; color:var(--pai-muted); margin:0; }}
  .pai-withheld {{ margin:16px 0 0; padding:0 0 0 15px; border-left:2px solid var(--pai-danger); }}
  .pai-withheld-lab {{ font-size:.6rem; letter-spacing:.16em; font-weight:600; text-transform:uppercase; color:var(--pai-danger); margin:0 0 4px; }}
  .pai-withheld-val {{ font-size:.86rem; line-height:1.56; color:var(--pai-text); margin:0; }}

  /* status marks: a dot plus words, so colour is never the only signal */
  .pai-marks {{ display:flex; gap:18px; flex-wrap:wrap; margin:12px 0 0; padding:0; list-style:none; }}
  .pai-mark {{
      font-size:.63rem; letter-spacing:.11em; font-weight:600; text-transform:uppercase;
      color:var(--pai-muted); display:flex; align-items:center; gap:7px;
  }}
  .pai-mark::before {{ content:""; flex:0 0 5px; width:5px; height:5px; border-radius:50%; background:var(--pai-muted); }}
  .pai-mark.ok::before     {{ background:var(--pai-ok); }}
  .pai-mark.warn::before   {{ background:var(--pai-warn); }}
  .pai-mark.danger::before {{ background:var(--pai-danger); }}
  .pai-mark.accent::before {{ background:var(--pai-accent); }}

  /* ---- the stage: video dominant, insight rail subordinate ---- */
  .pai-rail {{ padding:0 0 0 28px; border-left:1px solid var(--pai-line); }}
  .pai-rail-lab {{
      font-size:.6rem; letter-spacing:.17em; font-weight:600; text-transform:uppercase;
      color:var(--pai-accent); margin:0 0 5px;
  }}
  .pai-rail-val {{ font-size:.9rem; line-height:1.58; color:var(--pai-text); margin:0 0 22px; }}

  /* measurement strip under the video: number first, label quiet */
  .pai-strip {{ display:flex; flex-wrap:wrap; margin:20px 0 0; border-top:1px solid var(--pai-line); }}
  .pai-strip-cell {{ flex:1 1 0; min-width:112px; padding:15px 20px 0 0; }}
  .pai-strip-val {{ font-size:1.6rem; font-weight:700; line-height:1.08; letter-spacing:-.022em; color:var(--pai-text); }}
  .pai-strip-val.absent {{ font-size:.92rem; font-weight:600; color:var(--pai-warn); letter-spacing:0; }}
  .pai-strip-lab {{ font-size:.61rem; letter-spacing:.13em; font-weight:600; text-transform:uppercase; color:var(--pai-muted); margin:7px 0 0; }}
  .pai-strip-src {{ font-size:.59rem; letter-spacing:.09em; font-weight:600; text-transform:uppercase; margin:4px 0 0; }}

  /* source-type vocabulary, straight from feature_provenance */
  .pai-src-measured {{ color:var(--pai-ok); }}
  .pai-src-degraded {{ color:var(--pai-warn); }}
  .pai-src-default  {{ color:var(--pai-danger); }}
  .pai-src-model    {{ color:var(--pai-accent); }}
  .pai-src-sim      {{ color:var(--pai-muted); }}

  /* ---- sections: a rule and a number, nothing more ---- */
  .pai-sec {{ margin:66px 0 0; padding-top:15px; border-top:1px solid var(--pai-line); }}
  .pai-sec-first {{ margin-top:38px; }}
  .pai-sec-title {{
      font-size:.65rem; letter-spacing:.2em; font-weight:600; text-transform:uppercase;
      color:var(--pai-muted); margin:0 0 5px;
  }}
  .pai-sec.accent .pai-sec-title {{ color:var(--pai-accent); }}
  .pai-sec-sub {{ font-size:.8rem; line-height:1.56; color:var(--pai-muted); margin:0 0 22px; max-width:62ch; }}

  /* rows: hairline separated, edge aligned, never equal-weight cards */
  .pai-row {{ padding:22px 0; border-top:1px solid var(--pai-line); }}
  .pai-row-first {{ border-top:0; padding-top:0; }}
  .pai-row-top {{ display:flex; gap:18px; align-items:baseline; }}
  .pai-row-title {{ flex:1 1 auto; font-size:1.14rem; font-weight:600; line-height:1.3; color:var(--pai-text); margin:0; }}
  .pai-row-sev {{ flex:0 0 auto; font-size:.6rem; letter-spacing:.14em; font-weight:600; text-transform:uppercase; color:var(--pai-muted); white-space:nowrap; }}
  .pai-row-sev.high {{ color:var(--pai-danger); }}
  .pai-row-sev.moderate {{ color:var(--pai-warn); }}
  .pai-row-sev.positive {{ color:var(--pai-ok); }}
  .pai-row-block {{ margin:10px 0 0; }}
  .pai-row-lab {{ font-size:.6rem; letter-spacing:.15em; font-weight:600; text-transform:uppercase; color:var(--pai-muted); margin:0 0 3px; }}
  .pai-row-val {{ font-size:.88rem; line-height:1.58; color:var(--pai-text); margin:0 0 14px; max-width:62ch; }}

  /* ---- 07 coaching reads as an instruction, not a report ---- */
  .pai-drill {{ padding:24px 0; border-top:1px solid var(--pai-line); }}
  .pai-drill-first {{ border-top:0; padding-top:0; }}
  .pai-drill-head {{ display:flex; gap:16px; align-items:baseline; margin:0 0 12px; }}
  .pai-drill-n {{ flex:0 0 auto; font-size:.66rem; letter-spacing:.14em; font-weight:700; color:var(--pai-accent); }}
  .pai-drill-title {{ font-size:1.24rem; font-weight:600; line-height:1.28; color:var(--pai-text); margin:0; }}

  /* ---- 06 evidence: numbered steps on a single measure ---- */
  .pai-chain {{ display:flex; gap:15px; padding:9px 0; border-top:1px solid rgba(30,39,51,.5); }}
  .pai-chain-first {{ border-top:0; }}
  .pai-chain-n {{ flex:0 0 auto; min-width:15px; font-size:.63rem; font-weight:700; color:var(--pai-muted); padding-top:2px; }}
  .pai-chain-body {{ flex:1 1 auto; min-width:0; }}
  .pai-chain-lab {{ font-size:.6rem; letter-spacing:.15em; font-weight:600; text-transform:uppercase; color:var(--pai-muted); }}
  .pai-chain-val {{ font-size:.87rem; line-height:1.56; color:var(--pai-text); margin:2px 0 0; }}

  /* ---- 08 / 09 readouts ---- */
  .pai-readout {{ padding:20px 0; border-top:1px solid var(--pai-line); }}
  .pai-readout-first {{ border-top:0; padding-top:0; }}
  .pai-readout-head {{ display:flex; gap:16px; align-items:baseline; margin:0 0 8px; }}
  .pai-readout-val {{ font-size:1.45rem; font-weight:700; line-height:1.1; letter-spacing:-.02em; }}
  .pai-readout-val.ok {{ color:var(--pai-ok); }}
  .pai-readout-val.warn {{ color:var(--pai-warn); }}
  .pai-readout-val.danger {{ color:var(--pai-danger); }}
  .pai-readout-val.absent {{ font-size:1rem; font-weight:600; color:var(--pai-warn); letter-spacing:0; }}

  /* ---- the full measurement table (advanced) ---- */
  .pai-mgroup-name {{
      font-size:.61rem; letter-spacing:.15em; font-weight:600; text-transform:uppercase;
      color:var(--pai-muted); margin:22px 0 6px;
  }}
  .pai-metric {{ display:flex; gap:18px; align-items:baseline; justify-content:space-between; padding:8px 0; border-top:1px solid rgba(30,39,51,.5); }}
  .pai-metric-name {{ font-size:.85rem; color:var(--pai-text); }}
  .pai-metric-val {{ font-size:.85rem; font-weight:700; color:var(--pai-text); white-space:nowrap; }}
  .pai-metric-val.absent {{ color:var(--pai-muted); font-weight:600; }}
  .pai-metric-srcline {{ font-size:.62rem; color:var(--pai-muted); margin:0; line-height:1.5; }}

  .pai-empty {{ padding:26px 0; border-top:1px solid var(--pai-line); }}
  .pai-empty-title {{ font-size:1.05rem; font-weight:600; color:var(--pai-text); margin:0 0 6px; }}
  .pai-empty-copy {{ font-size:.87rem; line-height:1.6; color:var(--pai-muted); margin:0; max-width:60ch; }}

  /* ---- tablet: video first, insight beneath it ---- */
  @media (max-width:1100px) {{
    .pai-rail {{ padding:24px 0 0 0; border-left:0; border-top:1px solid var(--pai-line); }}
  }}
  /* ---- mobile: recompose the order, do not just shrink ---- */
  @media (max-width:640px) {{
    .pai-strip-cell {{ flex:1 1 46%; padding-right:14px; }}
    .pai-row-top {{ flex-direction:column; gap:7px; }}
    .pai-sec {{ margin-top:52px; }}
  }}
  @media (prefers-reduced-motion:reduce) {{
    .pai-root * {{ animation:none !important; transition:none !important; }}
  }}
</style>
"""


def esc(value) -> str:
    """HTML-escape. Every dynamic value in this module goes through here."""
    if value is None:
        return ""
    return _html_std.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Measurement metadata: plain-language label + what it measures + grouping.
# Units are duplicated from app.FEATURE_LABELS on purpose (app.py imports this
# module, so importing back would be circular). tests/test_result_view.py
# asserts the two tables agree, so they cannot silently diverge.
# ---------------------------------------------------------------------------
GROUPS = ("Release", "Lower body", "Trunk & pelvis", "Upper body")

MEASUREMENTS = (
    # (feature, plain label, unit, group, "what this measures")
    ("release_angle_deg", "Release angle", "deg", "Release",
     "How far the bowling arm has come through the body at the moment the ball leaves the hand."),
    ("elbow_flexion_deg", "Elbow bend at release", "deg", "Release",
     "How bent the bowling elbow is as the ball is released. Lower means a straighter arm."),
    ("wrist_angle_deg", "Wrist angle", "deg", "Release",
     "How much the hand is laid over / bent back relative to the forearm at release."),
    ("angular_velocity_deg_s", "Shoulder turn speed", "deg/s", "Release",
     "How fast the shoulders are rotating through the delivery, in degrees per second."),
    ("knee_flexion_deg", "Front-knee bend", "deg", "Lower body",
     "How bent the front knee is when the front foot lands. Lower means a more braced leg."),
    ("ground_contact_time_s", "Front-foot contact time", "s", "Lower body",
     "How long the front foot stays planted before the body goes over it."),
    ("stride_length_norm", "Stride length", "x height", "Lower body",
     "The last stride before delivery, as a multiple of the bowler's own height."),
    ("trunk_lean_deg", "Trunk lean", "deg", "Trunk & pelvis",
     "How far the body is bent sideways at the top of the delivery."),
    ("hip_rotation_deg", "Pelvis tilt", "deg", "Trunk & pelvis",
     "How much the hip line is tilted away from horizontal at release."),
    ("shoulder_rotation_deg", "Shoulder-hip separation", "deg", "Upper body",
     "How far the shoulders are turned away from the hips at release."),
)

_META = {m[0]: {"label": m[1], "unit": m[2], "group": m[3], "plain": m[4]}
         for m in MEASUREMENTS}

# Source-type labels, derived from the pipeline's own feature_provenance.
_SRC_MEASURED = ("MEASURED", "pai-src-measured",
                 "Metric 3D body landmarks from the uploaded clip.")
_SRC_DEGRADED = ("MEASURED (2D FALLBACK)", "pai-src-degraded",
                 "Derived from 2D image landmarks, so these degrees are "
                 "approximate — camera angle distorts them.")
_SRC_DEFAULT = ("NOT MEASURED — DEFAULT USED", "pai-src-default",
                "The body landmarks needed for this were not found. This is a "
                "placeholder value, not a measurement of this delivery.")
_SRC_SIM = ("ENTERED MANUALLY", "pai-src-sim",
            "Typed into the simulator, not measured from video.")
_SRC_NONE = ("NOT AVAILABLE", "pai-src-default",
             "No value was produced for this measurement.")


# ---------------------------------------------------------------------------
# Pure layer
# ---------------------------------------------------------------------------
@dataclass
class DeliveryStatus:
    """Everything needed to answer "can I trust this analysis?"."""
    state: str = "NORMAL"                 # primary state, most severe wins
    states: tuple = ()                    # every state that applies
    subject_verified: Optional[bool] = None
    delivery_reliable: Optional[bool] = None
    blocked_reason: Optional[str] = None
    completeness: str = "complete"        # complete | partial | limited
    measured: int = 0
    degraded: int = 0
    missing: int = 0
    low_confidence: int = 0
    ood_features: tuple = ()
    bowler_track_id: Optional[int] = None
    bowler_confidence: Optional[float] = None
    landmark_summary: dict = field(default_factory=dict)
    warnings: tuple = ()
    headline: str = ""
    headline_copy: str = ""
    tone: str = "ok"                      # ok | warn | danger

    @property
    def trusted(self) -> bool:
        return self.state not in ("UNRELIABLE", "SCORING_WITHHELD")

    @property
    def scoring_available(self) -> bool:
        return self.blocked_reason is None


@dataclass
class Finding:
    key: str
    title: str
    observed: str = ""
    why: str = ""
    action: str = ""
    severity: str = "moderate"            # high | moderate | low | positive
    feature: Optional[str] = None
    value: Optional[float] = None
    reference: str = ""
    source_type: str = "reference"


@dataclass
class Measurement:
    feature: str
    label: str
    value: Optional[float]
    unit: str
    group: str
    plain: str
    source_label: str
    source_class: str
    source_note: str
    context: str = ""


def _num(d: Optional[dict], key):
    """Return a float, or None. Never substitutes a default."""
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def fmt(value, unit: str = "", digits: int = 1, absent: str = "not measured") -> str:
    """Format a measurement. Absent stays absent — it never becomes 0."""
    if value is None:
        return absent
    if unit == "s":
        return f"{value:.{digits}f} s"
    if unit in ("deg", "deg/s"):
        return f"{value:.{digits}f}°"
    if unit:
        return f"{value:.{digits}f} {unit}"
    return f"{value:.{digits}f}"


def _prov(result, feature: str) -> dict:
    prov = getattr(result, "feature_provenance", None) or {}
    entry = prov.get(feature)
    return entry if isinstance(entry, dict) else {}


def _source_for(result, feature: str, is_video: bool) -> tuple:
    """Map feature_provenance onto a human source label. No new metadata."""
    if not is_video:
        return _SRC_SIM
    entry = _prov(result, feature)
    src = entry.get("source")
    if src == config.PROV_WORLD_3D:
        label, cls, note = _SRC_MEASURED
    elif src == config.PROV_NORMALIZED_2D:
        label, cls, note = _SRC_DEGRADED
    elif src == config.PROV_MISSING:
        label, cls, note = _SRC_DEFAULT
    else:
        label, cls, note = _SRC_NONE
    band = entry.get("confidence")
    vis = entry.get("mean_landmark_visibility")
    if band and src != config.PROV_MISSING:
        note = f"{note} Landmark confidence: {band}"
        if isinstance(vis, (int, float)):
            note += f" (mean visibility {vis:.2f})."
    return label, cls, note


def _ood(result, bundle) -> tuple:
    """Out-of-distribution feature names, from ml_models, without importing torch.

    Returns an empty tuple whenever the check cannot be run (no bundle, or
    ml_models unavailable) so callers can treat "no bundle" and "in range"
    identically -- neither may be presented as a clean bill of health.
    """
    if bundle is None:
        return ()
    try:
        from . import ml_models
        hits = ml_models.out_of_distribution_warnings(
            getattr(result, "feature_vector", None) or {}, bundle)
    except Exception:
        return ()
    return tuple(f for f, _v, _lo, _hi in hits)


def _risk_level(result) -> str:
    """The demo model's own risk level, or '' when it produced none.

    Deliberately no default. The previous result screen fell back to "low" and
    then rendered a hard-coded 22% for it, which made a withheld assessment
    look like a clean one.
    """
    risk = getattr(result, "injury_risk", None) or {}
    if not isinstance(risk, dict):
        return ""
    return str(risk.get("risk_level") or "").strip().lower()


def _risk_score(result):
    """The classifier's own high-class probability as a percentage, or None.

    Never derived from the level label: a level is a category, not a number,
    and converting one into the other is how the old UI ended up with '22%'.
    """
    risk = getattr(result, "injury_risk", None) or {}
    if not isinstance(risk, dict):
        return None
    probs = risk.get("probabilities") or []
    if isinstance(probs, (list, tuple)) and len(probs) > 2:
        value = probs[2]
        if isinstance(value, (int, float)):
            return float(value) * 100.0
    return None


def assess_delivery(result, *, perf_bundle=None, is_video: bool = False) -> DeliveryStatus:
    """Decide how much the user should trust this run.

    Pure function of the AnalysisResult. The ordering of the checks IS the
    importance policy and is deliberately explicit rather than scored.
    """
    subject = getattr(result, "subject_verified", None)
    reliable = getattr(result, "delivery_reliable", None)
    blocked = getattr(result, "scoring_blocked_reason", None)
    prov = getattr(result, "feature_provenance", None) or {}
    # Identity evidence, which the pipeline already produces. docs/RESULT_REDESIGN.md
    # requires both of these on the top trust rung, but the gate previously read
    # `subject_verified` alone -- so a run whose bowler the pipeline had explicitly
    # *rejected* still rendered as trusted. No new API is needed to close this.
    bowler_confirmed = getattr(result, "bowler_confirmed", None)
    identity_switch_count = getattr(result, "identity_switch_count", 0) or 0
    role_engine = getattr(result, "role_engine", None) or {}
    engine_withheld = bool(role_engine.get("analysis_withheld")) if role_engine else False

    identity_unconfirmed = bowler_confirmed is False or engine_withheld
    identity_unstable = identity_switch_count > 0

    measured = degraded = missing = 0
    for fv_key, entry in prov.items():
        if not isinstance(entry, dict):
            continue
        src = entry.get("source")
        if src == config.PROV_WORLD_3D:
            measured += 1
        elif src == config.PROV_NORMALIZED_2D:
            degraded += 1
        elif src == config.PROV_MISSING:
            missing += 1
    low_conf = sum(1 for e in prov.values()
                   if isinstance(e, dict) and e.get("confidence") == "low")

    states = []
    if subject is False or reliable is False:
        states.append("UNRELIABLE")
    if identity_unconfirmed:
        # An unconfirmed bowler is treated exactly like an unverified subject: the
        # measurements may belong to another player, so nothing may be presented
        # as a technique judgement.
        states.append("UNRELIABLE")
        if subject is not False:
            subject = False
    if identity_unstable:
        states.append("UNRELIABLE")
    if blocked:
        states.append("SCORING_WITHHELD")

    ood_perf = _ood(result, perf_bundle)
    if ood_perf:
        states.append("OOD")

    if missing or degraded or low_conf:
        states.append("PARTIAL")
    if not states:
        states.append("NORMAL")

    if missing or subject is False or reliable is False:
        completeness = "limited"
    elif degraded or low_conf:
        completeness = "partial"
    else:
        completeness = "complete"

    state = states[0]
    for candidate in ("UNRELIABLE", "SCORING_WITHHELD", "OOD", "PARTIAL"):
        if candidate in states:
            state = candidate
            break

    if state == "UNRELIABLE":
        tone = "danger"
        if identity_unstable and subject is not False:
            headline = "Analysis quality limited — the bowler identity was not stable"
            headline_copy = (
                f"The selected track changed {identity_switch_count} time(s) during the clip, "
                "so the body measurements below cannot be attributed to one bowler. No "
                "technique judgement has been made from them.")
        elif identity_unconfirmed and subject is not False:
            headline = "Analysis quality limited — the bowler was not confirmed in this clip"
            headline_copy = (
                "The cricket-evidence check did not confirm which person is bowling, so the "
                "body measurements below may belong to another player. No technique judgement "
                "has been made from them. Use the replay to check who is in frame, or confirm "
                "the bowler manually.")
        elif subject is False:
            headline = "Analysis quality limited — the bowler was not confirmed in this clip"
            headline_copy = (
                "More than one person was detected and the movement analysis could not "
                "lock onto a bowler, so the body measurements below may belong to another "
                "player. No technique judgement has been made from them. Use the replay to "
                "check who is in frame, or record a clip with only the bowler in view.")
        else:
            headline = "Analysis quality limited — this clip did not contain a clear delivery"
            headline_copy = (
                "The run-up-to-release window was not found, so the measurements below come "
                "from an arbitrary section of the clip rather than from a bowling action. "
                "No technique judgement has been made from them. Re-record with the whole "
                "delivery in frame.")
    elif state == "SCORING_WITHHELD":
        tone = "danger"
        headline = "Scores withheld for this delivery"
        headline_copy = blocked or "The pipeline declined to produce model scores for this run."
    elif state == "OOD":
        tone = "warn"
        headline = "This delivery sits outside the range the model was trained on"
        headline_copy = (
            "One or more measurements fall outside the range the demo models were fitted on, "
            "so the scores below are extrapolations and should be treated as unreliable. "
            "The raw measurements themselves are still shown.")
    elif state == "PARTIAL":
        tone = "warn"
        headline = "Partially measured delivery"
        headline_copy = (
            "Some measurements were taken from 2D image landmarks or could not be measured at "
            "all. Affected values are labelled individually below. Treat them as indicative "
            "and re-record with the bowler fully in frame and side-on to the camera for a "
            "cleaner read.")
    else:
        tone = "ok"
        if is_video:
            headline = "Analysis quality good — measurements taken from 3D body landmarks"
            headline_copy = ("All measurements for this delivery were taken from metric 3D "
                             "body landmarks. The findings below are directly traceable to "
                             "the replay.")
        else:
            headline = "Simulator entry — values were typed in, not measured"
            headline_copy = ("These values came from the simulator sliders, so they describe "
                             "the action you described rather than a recorded delivery.")

    return DeliveryStatus(
        state=state, states=tuple(states), subject_verified=subject,
        delivery_reliable=reliable, blocked_reason=blocked,
        completeness=completeness, measured=measured, degraded=degraded,
        missing=missing, low_confidence=low_conf, ood_features=ood_perf,
        bowler_track_id=getattr(result, "bowler_track_id", None),
        bowler_confidence=getattr(result, "bowler_confidence", None),
        landmark_summary=dict(getattr(result, "landmark_source_summary", None) or {}),
        warnings=tuple(getattr(result, "warnings", None) or ()),
        headline=headline, headline_copy=headline_copy, tone=tone,
    )


def build_measurements(result, *, is_video: bool = False) -> list:
    """Ordered, grouped measurements with value / unit / context / source type."""
    fv = getattr(result, "feature_vector", None) or {}
    out = []
    for feature, label, unit, group, plain in MEASUREMENTS:
        value = _num(fv, feature)
        src_label, src_cls, src_note = _source_for(result, feature, is_video)
        entry = _prov(result, feature)
        context = ""
        band = entry.get("confidence")
        if band and src_label != _SRC_NONE[0]:
            context = f"Landmark confidence {band}"
        out.append(Measurement(
            feature=feature, label=label, value=value, unit=unit, group=group,
            plain=plain, source_label=src_label, source_class=src_cls,
            source_note=src_note, context=context,
        ))
    return out


def build_supporting_measurements(measurements: list, findings: list,
                                  limit: int = 4) -> list:
    """The few measurements that support the primary finding.

    Section 05 is specified as "only the measurements relevant to the finding",
    so this cannot simply be the first N rows. Selection is a fixed, inspectable
    rule over data already present in `measurements` -- no importance score is
    invented and nothing is re-weighted:

      1. the primary finding's own feature, when it names one
      2. any other feature a finding cites
      3. remaining measured values, in their existing table order

    A measurement with no value is never promoted: an absent reading cannot
    support a claim about the delivery, it can only qualify one.
    """
    by_feature = {m.feature: m for m in measurements}
    picked: list = []
    seen: set = set()

    def take(feature) -> None:
        m = by_feature.get(feature)
        if m is not None and feature not in seen:
            seen.add(feature)
            picked.append(m)

    for f in (findings or [])[:1]:
        if f.feature:
            take(f.feature)
    for f in (findings or []):
        if f.feature:
            take(f.feature)
    for m in measurements:
        if len(picked) >= limit:
            break
        if m.value is not None:
            take(m.feature)
    return picked[:limit]


def summarise_delivery(status: "DeliveryStatus", findings: list,
                       measurements: list) -> str:
    """One plain-language sentence about the delivery as a whole.

    Section 02 is the delivery summary and section 03 is the finding, so this
    must not restate the finding. It counts only what the run actually
    produced, and returns "" when the trust gate has already said everything
    there is to say (see the withheld branch in render_delivery_status).
    """
    if status.state in ("UNRELIABLE", "SCORING_WITHHELD"):
        return ""
    measured = sum(1 for m in measurements if m.value is not None)
    if not measured:
        return "No measurement survived the trust checks, so there is nothing to summarise."
    unit = "value" if measured == 1 else "values"
    # build_findings never returns an empty list: a delivery that crossed nothing
    # still gets a positive confirmation finding (severity "positive"). Counting
    # that as a crossed band would state the opposite of what was measured.
    crossed = [f for f in (findings or []) if f.severity != "positive"]
    if not crossed:
        return (f"All {measured} measured {unit} sat inside the screening bands "
                f"PaceAI checks against.")
    n = len(crossed)
    pattern = "pattern" if n == 1 else "patterns"
    return f"{measured} {unit} measured. {n} {pattern} fell outside those bands."


def _actionable_notes(result) -> list:
    """Coaching notes that describe a technique change.

    Filters out the two boilerplate shapes the pipeline emits for its own
    bookkeeping ('No significant technical flags...', 'Demonstration
    performance score: ...'), which are surfaced in sections 08 and 02 instead.
    """
    notes = list(getattr(result, "coaching_notes", None) or [])
    keep = []
    for n in notes:
        s = str(n)
        if s.startswith("No significant technical flags"):
            continue
        if s.startswith("Demonstration performance score:"):
            continue
        keep.append(s)
    return keep


def build_findings(result, status: DeliveryStatus) -> list:
    """Pick at most three findings, most defensible evidence first.

    Order of evidence strength:
      1. literature threshold triggers (deterministic, on measured features)
      2. the strongest single biomechanical driver of the risk indicator
      3. a positive confirmation when nothing was crossed

    No new importance score is invented: the ordering is a fixed, inspectable
    rule over evidence the pipeline already produced.
    """
    if status.state in ("UNRELIABLE", "SCORING_WITHHELD"):
        return []

    fv = getattr(result, "feature_vector", None) or {}
    clinical = injury_kb.map_from_pipeline_features(fv)
    cards = injury_kb.assess_biomechanical_risks(clinical)
    findings: list = []

    for card in cards[:3]:
        sev = str(card.get("severity", "")).lower()
        findings.append(Finding(
            key=f"kb:{card.get('key') or card.get('injury')}",
            title=str(card.get("injury", "Flagged pattern")),
            observed="; ".join(card.get("trigger_detected") or []) or "—",
            why=(f"Reference screening band crossed"
                 + (f" — reported incidence {card.get('clinical_incidence')}"
                    if card.get("clinical_incidence") else "")
                 + (f"; typical return {card.get('est_recovery_timeline')}"
                    if card.get("est_recovery_timeline") else "") + "."),
            action=("Review this movement pattern with your coach or S&C staff before "
                    "increasing bowling volume."),
            severity="high" if sev.startswith("high") else "moderate",
            reference="Published screening benchmark (see Advanced → literature table)",
            source_type="reference",
        ))

    if len(findings) < 3:
        risk = getattr(result, "injury_risk", None) or {}
        shap = getattr(result, "shap_contributions_injury", None) or {}
        level = str(risk.get("risk_level", "")).lower()
        if level in ("moderate", "high") and shap:
            top = max(shap, key=lambda k: abs(shap[k]))
            meta = _META.get(top, {})
            findings.append(Finding(
                key=f"shap:{top}",
                title=f"{meta.get('label', top.replace('_', ' ').title())} is the main "
                      f"driver of the risk indicator",
                observed=fmt(_num(fv, top), _META.get(top, {}).get("unit", "")),
                why=("The model attributes more of the biomechanical risk-indicator score to "
                     "this measurement than to any other in this delivery."),
                action="Work on this measurement first — it is where a change will move the "
                       "indicator most.",
                severity="moderate", feature=top, value=_num(fv, top),
                source_type="model-derived",
            ))

    if not findings:
        elbow = _num(fv, "elbow_flexion_deg")
        if elbow is not None and elbow > config.ICC_ELBOW_EXTENSION_LIMIT_DEG:
            findings.append(Finding(
                key="icc:elbow",
                title="Bowling elbow is more bent at release than the screening reference",
                observed=f"Elbow flexion {fmt(elbow, 'deg')}",
                why=(f"The screening reference for elbow extension at release is "
                     f"{config.ICC_ELBOW_EXTENSION_LIMIT_DEG}°. This is a screening "
                     f"indicator only, not an official ICC on-field measurement."),
                action="Aim for a straighter arm path through release.",
                severity="moderate", feature="elbow_flexion_deg", value=elbow,
                reference=f"{config.ICC_ELBOW_EXTENSION_LIMIT_DEG}° reference",
                source_type="screening",
            ))

    if not findings:
        findings.append(Finding(
            key="none",
            title="No published screening threshold was crossed on this delivery",
            observed="",
            why=("Each measured value stayed inside the reference bands the app screens "
                 "against. That is a statement about these specific thresholds, not a "
                 "clearance from injury."),
            action="Repeat and compare over several deliveries — single deliveries vary.",
            severity="positive", source_type="reference",
        ))
    return findings[:3]


def build_evidence(result, findings: list, status: DeliveryStatus) -> list:
    """finding -> measurement -> reference -> interpretation -> action chains."""
    fv = getattr(result, "feature_vector", None) or {}
    chains = []
    for f in findings:
        if f.key == "none":
            continue
        measurement = f.observed
        if f.feature and f.value is not None:
            meta = _META.get(f.feature, {})
            measurement = f"{meta.get('label', f.feature)} — {fmt(f.value, meta.get('unit', ''))}"
        if not measurement:
            continue
        chains.append([
            ("Finding", f.title),
            ("Measurement", measurement),
            ("Reference", f.reference or "Published screening benchmark"),
            ("Interpretation", f.why),
            ("Action", f.action),
        ])
    return chains


# ---------------------------------------------------------------------------
# Render layer
# ---------------------------------------------------------------------------
def _html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def _section(number: str, title: str, sub: str = "", *, accent: bool = False,
             first: bool = False) -> None:
    cls = "pai-sec accent" if accent else "pai-sec"
    if first:
        cls += " pai-sec-first"
    _html(
        f'<div class="pai-root"><div class="{cls}">'
        f'<p class="pai-sec-title">{esc(number)} · {esc(title)}</p>'
        + (f'<p class="pai-sec-sub">{esc(sub)}</p>' if sub else "")
        + "</div></div>")


def _marker(number: str, title: str) -> None:
    """Section label for use inside a column, where a full section head with
    its top rule and 66px lead-in would break the composition."""
    _html(f'<div class="pai-root"><p class="pai-eyebrow accent">'
          f'{esc(number)} · {esc(title)}</p></div>')


def render_delivery_status(status: DeliveryStatus, result=None) -> None:
    """01 — Can this be trusted? A quiet line, never an alarm panel."""
    _section("01", "Delivery status",
             "What the quality of this analysis is, before anything it says is read.",
             first=True)

    cls = {"ok": "ok", "warn": "warn", "danger": "danger"}[status.tone]
    marks = []

    if status.subject_verified is True:
        marks.append(("Bowler confirmed", "ok"))
    elif status.subject_verified is False:
        marks.append(("Bowler not confirmed", "danger"))
    if status.delivery_reliable is True:
        marks.append(("Delivery found", "ok"))
    elif status.delivery_reliable is False:
        marks.append(("Delivery not found", "danger"))
    if status.completeness == "complete":
        marks.append(("Fully measured", "ok"))
    elif status.completeness == "partial":
        marks.append((f"Partial — {status.degraded} degraded", "warn"))
    else:
        marks.append((f"Limited — {status.missing} unmeasured", "danger"))
    if status.low_confidence:
        marks.append((f"{status.low_confidence} low-confidence", "warn"))
    if status.ood_features:
        marks.append(("Outside model range", "warn"))
    if status.bowler_track_id is not None:
        conf = status.bowler_confidence
        marks.append((f"Bowler track #{status.bowler_track_id}"
                      + (f" · {conf:.0%}" if isinstance(conf, (int, float)) else ""),
                      "accent"))

    mark_html = "".join(
        f'<li class="pai-mark {c}">{esc(t)}</li>' for t, c in marks)

    _html(
        f'<div class="pai-root"><div class="pai-state" role="status">'
        f'<span class="pai-state-word {cls}">{esc(status.headline)}</span>'
        f'<p class="pai-state-copy">{esc(status.headline_copy)}</p></div>'
        f'<ul class="pai-marks">{mark_html}</ul></div>')

    if status.blocked_reason:
        _html(
            '<div class="pai-root"><div class="pai-withheld">'
            '<p class="pai-withheld-lab">What was withheld</p>'
            f'<p class="pai-withheld-val">{esc(status.blocked_reason)}</p>'
            "</div></div>")

    for w in status.warnings[:6]:
        st.warning(w)
    if len(status.warnings) > 6:
        st.caption(f"{len(status.warnings) - 6} further pipeline notes are listed in "
                   f"Advanced → pipeline notes.")


def render_summary(result, status: DeliveryStatus, findings: list,
                   measurements: list) -> None:
    """02 — One sentence about the delivery as a whole.

    Deliberately not a restatement of the finding: 02 is the delivery, 03 is
    the pattern inside it. See `summarise_delivery`.
    """
    _section("02", "What this delivery tells you")
    line = summarise_delivery(status, findings, measurements)
    if line:
        _html(f'<div class="pai-root"><p class="pai-lede">{esc(line)}</p></div>')
        return
    # Withheld or unreliable: 01 has already said why, so do not repeat it.
    _html('<div class="pai-root"><p class="pai-body">The trust gate above stopped this '
          'delivery before a conclusion could be drawn, so no summary is offered. '
          'What was measured is still listed below, labelled by source.</p></div>')


def render_primary_finding(result, status: DeliveryStatus,
                           finding: Optional[Finding], *,
                           replay_fn=None, replay_path=None,
                           feature_vector: Optional[dict] = None,
                           ball_stats: Optional[dict] = None,
                           strip: Optional[list] = None) -> None:
    """03 + 04 + 05 — the one hero statement, the video, and its evidence.

    Composition is 68/32: the replay is the dominant object and the reading of
    it sits in a subordinate rail. On tablet the rail drops beneath the video
    (see the 1100px breakpoint); on mobile the whole stage recomposes.
    """
    _section("03", "Top finding", accent=True)

    if finding is None:
        _html('<div class="pai-root">'
              '<p class="pai-display tight">Nothing to act on yet.</p>'
              '<p class="pai-body">This delivery did not produce enough trustworthy '
              'evidence to raise a finding. Section 01 says why. The measurements '
              'that were captured are still shown below, each labelled by where it '
              'came from.</p></div>')
    else:
        _html(f'<div class="pai-root">'
              f'<p class="pai-display">{esc(finding.title)}</p></div>')

    video_col, rail_col = st.columns([68, 32], gap="large")

    with video_col:
        # 04 and 05 are labelled unconditionally. The hierarchy is fixed at 11
        # sections: dropping the label when there is no clip would leave the
        # reader wondering whether evidence was withheld, rather than told.
        _marker("04", "Analysis replay")
        if replay_fn is not None and replay_path:
            replay_fn(replay_path, result, feature_vector or {}, ball_stats or {})
        else:
            render_no_replay()
        _marker("05", "Key measurements")
        if strip:
            render_measurement_strip(strip)
        else:
            _html('<div class="pai-root"><div class="pai-empty">'
                  '<p class="pai-empty-title">No measurement to show</p>'
                  '<p class="pai-empty-copy">Nothing this run produced both carried a '
                  'value and was relevant to the finding above.</p></div></div>')

    with rail_col:
        _html('<div class="pai-root"><div class="pai-rail">')
        if finding is not None:
            _rail_block("Why it matters", finding.why)
            if finding.observed:
                _rail_block("What was measured", finding.observed)
        notes = _actionable_notes(result)
        action = (finding.action if finding is not None and finding.action
                  else (notes[0] if notes else ""))
        if action:
            _rail_block("What to work on", action)
        elif status.state in ("UNRELIABLE", "SCORING_WITHHELD"):
            _rail_block("What to work on",
                       "Nothing yet — this delivery has to be re-recorded or the "
                       "bowler confirmed before a change is worth chasing.")
        _html("</div></div>")


def _rail_block(label: str, value: str) -> None:
    _html(f'<p class="pai-rail-lab">{esc(label)}</p>'
          f'<p class="pai-rail-val">{esc(value)}</p>')


def render_measurement_strip(strip: list) -> None:
    """05 — the few measurements that support the finding, number first."""
    cells = []
    for m in strip:
        if m.value is None:
            val = (f'<p class="pai-strip-val absent">'
                   f'{esc(m.source_label.split(" —")[0])}</p>')
        else:
            val = f'<p class="pai-strip-val">{esc(fmt(m.value, m.unit))}</p>'
        cells.append(
            f'<div class="pai-strip-cell">{val}'
            f'<p class="pai-strip-lab">{esc(m.label)}</p>'
            f'<p class="pai-strip-src {esc(m.source_class)}">'
            f'{esc(m.source_label)}</p></div>')
    _html('<div class="pai-root"><div class="pai-strip">'
          + "".join(cells) + "</div></div>")


def render_remaining_findings(findings: list) -> None:
    """03 continued — only the findings the hero has not already stated.

    The primary finding is rendered as the display statement in
    render_primary_finding. Repeating it here as the first row of a list is
    the duplication the "remove 20%" test is designed to catch.
    """
    rest = list(findings or [])[1:]
    if not rest:
        return

    _section("03", "Other findings",
             "The primary finding is stated above. These ranked behind it.")

    for f in rest:
        rows = ""
        if f.observed:
            rows += (f'<div class="pai-row-block">'
                     f'<p class="pai-row-lab">Observed</p>'
                     f'<p class="pai-row-val">{esc(f.observed)}</p></div>')
        rows += (f'<div class="pai-row-block">'
                 f'<p class="pai-row-lab">Why it matters</p>'
                 f'<p class="pai-row-val">{esc(f.why)}</p></div>')
        if f.action:
            rows += (f'<div class="pai-row-block">'
                     f'<p class="pai-row-lab">Action</p>'
                     f'<p class="pai-row-val">{esc(f.action)}</p></div>')
        _html(
            f'<div class="pai-root"><div class="pai-row">'
            f'<div class="pai-row-top">'
            f'<p class="pai-row-title">{esc(f.title)}</p>'
            f'<span class="pai-row-sev {esc(f.severity)}">{esc(f.severity)}</span>'
            f"</div>{rows}</div></div>")


def render_no_replay() -> None:
    """04 for entries with no clip behind them. States the absence, never fakes it."""
    _html('<div class="pai-root"><div class="pai-empty">'
          '<p class="pai-empty-title">No replay for this entry</p>'
          '<p class="pai-empty-copy">These values were typed in rather than measured '
          'from a clip, so there is no video to trace the findings back to. Upload a '
          'bowling video to get the pose overlay, ball track and release marker.</p>'
          "</div></div>")


def render_measurement_table(measurements: list) -> None:
    """Every measurement, grouped, each with its source type.

    The complete table lives in section 10 rather than beside the finding:
    section 05 shows only what supports the finding, and hiding the rest is not
    an option (principle 10), so it is disclosed one level down instead.
    """
    if not measurements:
        return
    blocks = []
    for g in GROUPS:
        items = [m for m in measurements if m.group == g]
        if not items:
            continue
        rows = []
        for m in items:
            if m.value is None:
                val = (f'<span class="pai-metric-val absent">'
                       f'{esc(m.source_label.split(" —")[0])}</span>')
            else:
                val = f'<span class="pai-metric-val">{esc(fmt(m.value, m.unit))}</span>'
            ctx = f' · {esc(m.context)}' if m.context else ""
            rows.append(
                f'<div class="pai-metric">'
                f'<span class="pai-metric-name">{esc(m.label)}</span>{val}'
                f'<span class="pai-metric-srcline">'
                f'<span class="{esc(m.source_class)}">{esc(m.source_label)}</span>'
                f' · {esc(m.plain)}{esc(m.source_note)}{ctx}</span></div>')
        blocks.append(f'<p class="pai-mgroup-name">{esc(g)}</p>' + "".join(rows))

    _html('<div class="pai-root">' + "".join(blocks) + "</div>")


def render_evidence_panel(chains: list, status: Optional[DeliveryStatus] = None) -> None:
    """06 — The reasoning, exposed."""
    _section("06", "Why this was flagged",
             "Each finding traced from the measurement to the action.")
    if not chains:
        if status is not None and status.state in ("UNRELIABLE",
                                                    "SCORING_WITHHELD"):
            reason = ("no reasoning is shown because the pipeline withheld its "
                      "assessment of this delivery")
        else:
            reason = ("no threshold was crossed in this delivery, so there is no "
                      "flag to trace")
        _html('<div class="pai-root"><div class="pai-empty">'
              '<p class="pai-empty-title">No evidence chain</p>'
              f'<p class="pai-empty-copy">For this delivery {esc(reason)}.</p>'
              "</div></div>")
        return
    for chain in chains:
        steps = "".join(
            f'<div class="pai-chain"><span class="pai-chain-n">{i + 1}</span>'
            f'<div class="pai-chain-body"><p class="pai-chain-lab">{esc(lab)}</p>'
            f'<p class="pai-chain-val">{esc(val)}</p></div></div>'
            for i, (lab, val) in enumerate(chain))
        _html(f'<div class="pai-root">{steps}</div>')


def render_coaching_plan(result, status: DeliveryStatus) -> None:
    """07 — Written as a training instruction, not as another analytics panel.

    Each drill is: the change, WHY it was raised (the measured value and the
    band it crossed), and what to do next session. The WHY is only ever filled
    from a finding that actually cites the feature, so a drill can never carry
    a justification the pipeline did not produce.
    """
    _section("07", "What to work on",
             "Each item below was raised by this delivery's own threshold checks. "
             "Nothing here is generic advice.")
    notes = _actionable_notes(result)

    if not notes:
        if status.state in ("UNRELIABLE", "SCORING_WITHHELD"):
            _html('<div class="pai-root"><div class="pai-empty">'
                  '<p class="pai-empty-title">No drill yet</p>'
                  '<p class="pai-empty-copy">No coaching recommendation was produced for '
                  'this delivery'
                  + (" because the bowler could not be confirmed."
                     if status.state == "UNRELIABLE" else ".")
                  + "</p></div></div>")
        else:
            _html('<div class="pai-root"><div class="pai-empty">'
                  '<p class="pai-empty-title">No technique changes were flagged</p>'
                  '<p class="pai-empty-copy">Every measured value stayed inside the '
                  'reference bands this app screens against. Keep repeating the delivery '
                  'and compare across sessions.</p></div></div>')
        return

    findings = build_findings(result, status)
    fv = dict(getattr(result, "feature_vector", None) or {})

    for i, note in enumerate(notes[:4]):
        why = _why_for_note(note, findings, fv)
        why_block = ""
        if why:
            why_block = (f'<div class="pai-row-block">'
                         f'<p class="pai-row-lab">Why</p>'
                         f'<p class="pai-row-val">{esc(why)}</p></div>')
        first = " pai-drill-first" if i == 0 else ""
        _html(
            f'<div class="pai-root"><div class="pai-drill{first}">'
            f'<div class="pai-drill-head">'
            f'<span class="pai-drill-n">{i + 1:02d}</span>'
            f'<p class="pai-drill-title">{esc(note)}</p></div>'
            f"{why_block}"
            f'<div class="pai-row-block">'
            f'<p class="pai-row-lab">Next session</p>'
            f'<p class="pai-row-val">{esc(_next_session(i))}</p></div>'
            f"</div></div>")
    if len(notes) > 4:
        st.caption(f"{len(notes) - 4} further notes are in Advanced → coaching output.")


def _why_for_note(note: str, findings: list, fv: dict) -> str:
    """The measured justification for one coaching note, or "" if there is none.

    A note is only ever explained by a finding that actually fired, so this
    cannot attach a number that did not contribute to the flag.
    """
    for f in findings:
        if f.feature and _num(fv, f.feature) is not None:
            label = _META.get(f.feature, {}).get("label", f.feature)
            unit = _META.get(f.feature, {}).get("unit", "")
            return (f"Your measured {label.lower()} was "
                    f"{fmt(_num(fv, f.feature), unit)} on this delivery.")
    return ""


def _next_session(index: int) -> str:
    """A concrete, repeatable next step. Index only rotates phrasing."""
    return (
        "Record 3–5 deliveries from the same camera angle and compare the movement "
        "against this one.",
        "Film the same angle again next session, then check whether this value moved.",
        "Repeat this delivery five times and keep the clip that matches this one most "
        "closely for side-by-side review.",
        "Bring one clip of this pattern to your next session and review it against this "
        "measurement.",
    )[index % 4]


def render_performance_indicator(result, bundle, ml_models=None) -> None:
    """08 — Honest terminology, visually small, model context attached."""
    _section("08", "Performance indicator",
             "A demonstration score from models trained on synthetic data. It is a "
             "conversation starter, not a measurement of quality.")
    score = getattr(result, "performance_score", None)

    if score is None:
        _html('<div class="pai-root"><div class="pai-empty">'
              '<p class="pai-empty-title">Not scored</p>'
              '<p class="pai-empty-copy">No performance score was produced for this '
              'delivery. The raw measurements above are unaffected.</p></div></div>')
        return

    data_source = getattr(bundle, "data_source", "unknown")
    if data_source == "synthetic":
        ctx = ("Demo model trained on synthetic data whose labels were generated from the "
               "features themselves. Illustrative only — not a validated measurement.")
    elif data_source == "real":
        ctx = "Trained on a real labelled dataset. Not a clinically validated measurement."
    else:
        ctx = "Model provenance unknown (bundle saved by an older version). Treat as demo."

    interval = None
    if ml_models is not None and bundle is not None:
        try:
            interval = ml_models.prediction_interval_performance(
                bundle, getattr(result, "feature_vector", None) or {})
        except Exception:
            interval = None

    if score >= 80:
        band = "strong, technically sound action — focus on consistency and repeatability"
    elif score >= 60:
        band = "solid foundation with a few refinements to make"
    else:
        band = "below benchmark — several technical elements need focused work"

    interval_html = ""
    if interval:
        interval_html = (f'<p class="pai-body">Model uncertainty: 68% of predictions '
                         f'for inputs like this fall between {interval[0]:.0f} and '
                         f'{interval[1]:.0f}. This is a band on the model, not a confidence '
                         f'interval on a physical measurement.</p>')

    tone = "ok" if score >= 80 else ("warn" if score >= 60 else "danger")
    _html(
        f'<div class="pai-root"><div class="pai-readout">'
        f'<div class="pai-readout-head">'
        f'<p class="pai-eyebrow">Performance indicator</p>'
        f'<p class="pai-readout-val {tone}">{score:.0f}'
        f'<span style="font-size:.85rem;color:{PAI_MUTED}"> / 100</span></p></div>'
        f'<p class="pai-body">{esc(band)}.</p>'
        f'<p class="pai-body">{esc(ctx)}</p>{interval_html}'
        f"</div></div>")


def render_risk_indicators(result, status: DeliveryStatus) -> None:
    """09 — Screening context. Never a diagnosis, never a score when withheld."""
    _section("09", "Risk indicators",
             "Published screening bands checked against this delivery's measurements. "
             "These are screening indicators — not a medical assessment, and not an "
             "injury prediction.")
    clinical = injury_kb.map_from_pipeline_features(
        getattr(result, "feature_vector", None) or {})
    cards = injury_kb.assess_biomechanical_risks(clinical)

    if not cards:
        _html('<div class="pai-root"><div class="pai-empty">'
              '<p class="pai-empty-title">No screening band was crossed</p>'
              '<p class="pai-empty-copy">None of the measurements available for this '
              'delivery fell outside the published screening ranges. This is not a '
              'clearance from injury.</p></div></div>')
    else:
        for c in cards:
            sev = str(c.get("severity", "")).lower()
            tone = "danger" if sev.startswith("high") else "warn"
            rows = ""
            for t in (c.get("trigger_detected") or []):
                rows += (f'<div class="pai-row-block">'
                         f'<p class="pai-row-lab">Triggered factor</p>'
                         f'<p class="pai-row-val">{esc(t)}</p></div>')
            if c.get("clinical_incidence"):
                rows += (f'<div class="pai-row-block">'
                         f'<p class="pai-row-lab">Reported incidence</p>'
                         f'<p class="pai-row-val">{esc(c["clinical_incidence"])}</p></div>')
            if c.get("est_recovery_timeline"):
                rows += (f'<div class="pai-row-block">'
                         f'<p class="pai-row-lab">Recovery context</p>'
                         f'<p class="pai-row-val">'
                         f'{esc(c["est_recovery_timeline"])}</p></div>')
            rows += ('<div class="pai-row-block">'
                     '<p class="pai-row-lab">Limitation</p>'
                     '<p class="pai-row-val">A screening band, not a diagnosis. '
                     'Reference values in this build are pending source verification '
                     '(see CITATION.md).</p></div>')
            _html(
                f'<div class="pai-root"><div class="pai-readout">'
                f'<div class="pai-readout-head">'
                f'<p class="pai-eyebrow">{esc(c.get("anatomical_site") or "Screening band")}</p>'
                f'<p class="pai-readout-val {tone}">{esc(str(c.get("injury") or ""))}</p>'
                f"</div>{rows}</div></div>")

    risk = getattr(result, "injury_risk", None)
    fv = getattr(result, "feature_vector", None) or {}

    model_block = ""
    if risk is None:
        model_block = (
            '<p class="pai-eyebrow" style="margin-top:26px">Model risk indicator</p>'
            '<p class="pai-body">No model risk-indicator output was produced for this '
            'delivery, so no risk level or probability is shown. Nothing is inferred in its '
            'place.</p>')
    else:
        level = _risk_level(result) or ""
        tone = "ok" if level == "low" else ("warn" if level == "moderate" else "danger")
        model_block = (
            '<p class="pai-eyebrow" style="margin-top:26px">Model risk indicator</p>'
            f'<p class="pai-readout-val {tone}">{esc(level.upper() or "UNAVAILABLE")}</p>'
            '<p class="pai-body">Demo model output. Its training labels were derived '
            'from the benchmark thresholds, so it largely re-states the screening result '
            'above rather than adding independent evidence. It is a screening context '
            'indicator, not a diagnosis, and not an injury prediction.</p>')

    _html(f'<div class="pai-root">{model_block}</div>')


def render_advanced(result, *, perf_bundle=None, injury_bundle=None,
                    status: Optional[DeliveryStatus] = None,
                    measurements: Optional[list] = None,
                    extra_renderers: Optional[dict] = None) -> None:
    """10 — Everything technical, collapsed."""
    with st.expander("Advanced analysis — full technical detail, diagnostics and export",
                     expanded=False):
        _section("10", "Advanced analysis",
                 "Model internals, SHAP contributions, the complete measurement table, "
                 "the reference table and run diagnostics. Collapsed by default — none of "
                 "this is needed to act on the findings above.")
        extra = extra_renderers or {}
        if extra.get("gauges_and_radar"):
            extra["gauges_and_radar"]()
        if extra.get("shap"):
            extra["shap"]()
        if extra.get("literature_table"):
            extra["literature_table"]()
        if extra.get("model_quality"):
            extra["model_quality"]()
        if extra.get("timing"):
            extra["timing"]()
        if extra.get("notes"):
            extra["notes"]()

        if measurements:
            st.markdown("**All measurements** — every value this run captured, with the "
                        "plain-language meaning of each")
            render_measurement_table(measurements)

        prov = getattr(result, "feature_provenance", None) or {}
        if prov:
            st.markdown("**Feature provenance** — where each measurement came from")
            rows = []
            for feature, meta in _META.items():
                entry = prov.get(feature) or {}
                rows.append({
                    "Feature": meta["label"],
                    "Source": entry.get("source", "n/a"),
                    "Confidence": entry.get("confidence", "n/a"),
                    "Mean visibility": (f"{entry['mean_landmark_visibility']:.2f}"
                                        if isinstance(entry.get("mean_landmark_visibility"),
                                                      (int, float)) else "n/a"),
                })
            st.dataframe(rows, width="stretch", hide_index=True)

        summary = getattr(result, "landmark_source_summary", None) or {}
        if summary:
            st.markdown("**Landmark source**")
            st.json(summary)

        backends = getattr(result, "stage_backends", None) or {}
        if backends:
            st.markdown("**Stage backends**")
            st.json(backends)

        st.markdown("**Raw feature vector**")
        st.json(dict(getattr(result, "feature_vector", None) or {}))


def render_history_export(result, session_writer: Optional[Callable] = None) -> None:
    """11 — Utilities only, deliberately quiet."""
    _section("11", "History & export",
             "Save this delivery, or export the full result as JSON.")
    if session_writer is not None:
        session_writer()


def render_result_page(result, *, perf_bundle=None, injury_bundle=None,
                       is_video: bool = False, replay_fn=None,
                       replay_path=None, ball_stats=None,
                       advanced_renderers=None, session_writer=None) -> DeliveryStatus:
    """Orchestrator for sections 01-11. Returns the computed status.

    Render order is the information priority, not the section numbering:

      first viewport   01 trust gate, 02 delivery summary, 03 the finding as
                       display type, 04 the replay at 68% with 05's supporting
                       measurements beneath it, and one action in the rail
      below that      06 evidence, 07 what to work on, 08 performance,
                       09 risk indicators
      last            10 advanced (collapsed), 11 history/export

    The video sits in the first viewport on purpose. It is the only object that
    lets a bowler verify a claim about their own delivery, and burying it under
    a stack of panels was the single biggest thing separating this from the
    composition it was meant to be.
    """
    _html(RESULT_CSS)
    fv = dict(getattr(result, "feature_vector", None) or {})
    bs = dict(ball_stats or {})

    status = assess_delivery(result, perf_bundle=perf_bundle, is_video=is_video)
    findings = build_findings(result, status)
    measurements = build_measurements(result, is_video=is_video)
    chains = build_evidence(result, findings, status)
    strip = build_supporting_measurements(measurements, findings)

    render_delivery_status(status, result)
    render_summary(result, status, findings, measurements)
    render_primary_finding(
        result, status, findings[0] if findings else None,
        replay_fn=replay_fn, replay_path=replay_path,
        feature_vector=fv, ball_stats=bs, strip=strip)
    render_remaining_findings(findings)
    render_evidence_panel(chains, status)
    render_coaching_plan(result, status)
    render_performance_indicator(result, perf_bundle, _lazy_ml_models())
    render_risk_indicators(result, status)
    render_advanced(result, perf_bundle=perf_bundle, injury_bundle=injury_bundle,
                    status=status, measurements=measurements,
                    extra_renderers=advanced_renderers)
    render_history_export(result, session_writer)
    return status


def _lazy_ml_models():
    try:
        from . import ml_models
        return ml_models
    except Exception:
        return None
