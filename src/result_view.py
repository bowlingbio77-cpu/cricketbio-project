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
  .pai-kicker {{
      font-size:.66rem; letter-spacing:.15em; font-weight:800; text-transform:uppercase;
      color:var(--pai-muted); margin:0 0 6px;
  }}
  .pai-kicker.accent {{ color:var(--pai-accent); }}
  .pai-status {{
      border:1px solid var(--pai-line); border-left:3px solid var(--pai-muted);
      background:var(--pai-surface); border-radius:10px; padding:16px 18px; margin:0 0 6px;
  }}
  .pai-status.ok    {{ border-left-color:var(--pai-ok); }}
  .pai-status.warn  {{ border-left-color:var(--pai-warn); }}
  .pai-status.danger{{ border-left-color:var(--pai-danger); }}
  .pai-status-title {{ font-size:1.02rem; font-weight:800; color:var(--pai-text); margin:0 0 4px; }}
  .pai-status-copy  {{ color:var(--pai-muted); font-size:.87rem; line-height:1.55; margin:0; }}
  .pai-chips {{ display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 0; }}
  .pai-chip {{
      font-size:.66rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase;
      padding:4px 9px; border-radius:999px; border:1px solid var(--pai-line);
      color:var(--pai-muted); background:var(--pai-raised); white-space:nowrap;
  }}
  .pai-chip.ok     {{ color:var(--pai-ok);     border-color:rgba(67,217,163,.45); }}
  .pai-chip.warn   {{ color:var(--pai-warn);   border-color:rgba(232,179,74,.45); }}
  .pai-chip.danger {{ color:var(--pai-danger); border-color:rgba(255,112,134,.45); }}
  .pai-chip.accent {{ color:var(--pai-accent); border-color:rgba(32,217,255,.45); }}

  .pai-hero {{
      border:1px solid var(--pai-line); border-radius:14px; padding:24px 26px;
      background:linear-gradient(140deg, var(--pai-raised) 0%, var(--pai-surface) 70%);
      margin:0 0 8px;
  }}
  .pai-hero.danger {{ border-left:3px solid var(--pai-danger); }}
  .pai-hero.warn   {{ border-left:3px solid var(--pai-warn); }}
  .pai-hero.ok     {{ border-left:3px solid var(--pai-ok); }}
  .pai-hero.accent {{ border-left:3px solid var(--pai-accent); }}
  .pai-hero-main {{
      font-size:1.5rem; font-weight:800; line-height:1.28; color:var(--pai-text);
      margin:6px 0 14px; letter-spacing:-.01em;
  }}
  .pai-hero-row {{ display:flex; gap:26px; flex-wrap:wrap; }}
  .pai-hero-cell {{ flex:1 1 240px; min-width:0; }}
  .pai-hero-lab {{
      font-size:.63rem; letter-spacing:.14em; font-weight:800; text-transform:uppercase;
      color:var(--pai-accent); margin:0 0 3px;
  }}
  .pai-hero-val {{ font-size:.95rem; line-height:1.5; color:var(--pai-text); margin:0; }}

  .pai-finding {{
      border:1px solid var(--pai-line); border-radius:12px; padding:16px 18px;
      background:var(--pai-surface); margin:0 0 10px;
  }}
  .pai-finding-top {{ display:flex; justify-content:space-between; gap:14px; align-items:baseline; }}
  .pai-finding-title {{ font-size:1.02rem; font-weight:800; color:var(--pai-text); margin:0; }}
  .pai-finding-sev {{
      font-size:.63rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase;
      white-space:nowrap; color:var(--pai-muted);
  }}
  .pai-finding-sev.high {{ color:var(--pai-danger); }}
  .pai-finding-sev.moderate {{ color:var(--pai-warn); }}
  .pai-finding-sev.positive {{ color:var(--pai-ok); }}
  .pai-ev {{ display:flex; gap:10px; padding:7px 0; border-top:1px solid var(--pai-line); }}
  .pai-ev:first-of-type {{ border-top:0; }}
  .pai-ev-lab {{
      flex:0 0 108px; font-size:.63rem; letter-spacing:.1em; font-weight:800;
      text-transform:uppercase; color:var(--pai-muted); padding-top:2px;
  }}
  .pai-ev-val {{ flex:1 1 auto; font-size:.88rem; line-height:1.5; color:var(--pai-text); }}

  .pai-metrics {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .pai-mgroup {{ flex:1 1 210px; min-width:0; }}
  .pai-mgroup-name {{
      font-size:.63rem; letter-spacing:.13em; font-weight:800; text-transform:uppercase;
      color:var(--pai-muted); margin:0 0 8px; padding-bottom:6px; border-bottom:1px solid var(--pai-line);
  }}
  .pai-metric {{ padding:7px 0; border-bottom:1px solid rgba(30,39,51,.6); }}
  .pai-metric:last-child {{ border-bottom:0; }}
  .pai-metric-top {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
  .pai-metric-name {{ font-size:.83rem; color:var(--pai-text); }}
  .pai-metric-val {{ font-size:.95rem; font-weight:800; color:var(--pai-accent); white-space:nowrap; }}
  .pai-metric-val.absent {{ color:var(--pai-muted); font-weight:600; font-size:.82rem; }}
  .pai-metric-srcline {{ font-size:.68rem; color:var(--pai-muted); margin:2px 0 0; line-height:1.45; }}
  .pai-src-measured {{ color:var(--pai-ok); font-weight:700; }}
  .pai-src-degraded {{ color:var(--pai-warn); font-weight:700; }}
  .pai-src-default  {{ color:var(--pai-danger); font-weight:700; }}
  .pai-src-model    {{ color:var(--pai-accent); font-weight:700; }}
  .pai-src-sim      {{ color:var(--pai-muted); font-weight:700; }}

  .pai-plan {{ border:1px solid var(--pai-line); border-radius:12px; overflow:hidden; margin:0 0 10px; }}
  .pai-plan-head {{
      display:flex; gap:12px; align-items:center; padding:13px 16px; background:var(--pai-raised);
  }}
  .pai-plan-pri {{
      flex:0 0 auto; font-size:.63rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase;
      padding:3px 9px; border-radius:999px; border:1px solid var(--pai-line); color:var(--pai-muted);
  }}
  .pai-plan-pri.p1 {{ color:var(--pai-danger); border-color:rgba(255,112,134,.5); }}
  .pai-plan-pri.p2 {{ color:var(--pai-warn);   border-color:rgba(232,179,74,.5); }}
  .pai-plan-title {{ flex:1 1 auto; font-size:.93rem; font-weight:700; color:var(--pai-text); }}
  .pai-plan-body {{ padding:12px 16px; background:var(--pai-surface); }}

  .pai-small {{ border:1px solid var(--pai-line); border-radius:10px; padding:14px 16px;
                background:var(--pai-surface); margin:0 0 10px; }}
  .pai-small-grid {{ display:flex; gap:22px; flex-wrap:wrap; }}
  .pai-small-cell {{ flex:1 1 150px; min-width:0; }}
  .pai-small-lab {{ font-size:.62rem; letter-spacing:.12em; font-weight:800; text-transform:uppercase;
                    color:var(--pai-muted); margin:0 0 3px; }}
  .pai-small-val {{ font-size:1.5rem; font-weight:800; color:var(--pai-text); line-height:1.15; }}
  .pai-small-val.absent {{ font-size:1rem; font-weight:700; color:var(--pai-warn); }}
  .pai-small-note {{ font-size:.76rem; color:var(--pai-muted); margin:5px 0 0; line-height:1.5; }}

  .pai-evidence {{
      border:1px solid var(--pai-line); border-radius:10px; padding:14px 16px;
      background:var(--pai-surface); margin:0 0 10px;
  }}
  .pai-chain {{ display:flex; gap:8px; align-items:flex-start; padding:6px 0; }}
  .pai-chain-step {{
      flex:0 0 20px; height:20px; border-radius:50%; border:1px solid var(--pai-line);
      background:var(--pai-raised); color:var(--pai-muted); font-size:.63rem; font-weight:800;
      display:flex; align-items:center; justify-content:center; margin-top:1px;
  }}
  .pai-chain-body {{ flex:1 1 auto; min-width:0; }}
  .pai-chain-lab {{ font-size:.62rem; letter-spacing:.11em; font-weight:800; text-transform:uppercase;
                     color:var(--pai-accent); }}
  .pai-chain-val {{ font-size:.87rem; color:var(--pai-text); line-height:1.5; }}

  .pai-empty {{
      border:1px dashed var(--pai-line); border-radius:12px; padding:22px 20px;
      background:var(--pai-surface); margin:0 0 10px;
  }}
  .pai-empty-title {{ font-size:1rem; font-weight:800; color:var(--pai-text); margin:0 0 4px; }}
  .pai-empty-copy {{ font-size:.85rem; color:var(--pai-muted); line-height:1.55; margin:0; }}

  .pai-sec {{ margin:26px 0 10px; }}
  .pai-sec-title {{
      font-size:.68rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase;
      color:var(--pai-accent); margin:0 0 3px;
  }}
  .pai-sec-sub {{ font-size:.83rem; color:var(--pai-muted); margin:0 0 12px; line-height:1.5; }}

  @media (max-width:640px) {{
    .pai-ev-lab {{ flex-basis:100%; }}
    .pai-hero-main {{ font-size:1.2rem; }}
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
    if src == "world_3d":
        label, cls, note = _SRC_MEASURED
    elif src == "normalized_2d":
        label, cls, note = _SRC_DEGRADED
    elif src == "missing_xyz":
        label, cls, note = _SRC_DEFAULT
    else:
        label, cls, note = _SRC_NONE
    band = entry.get("confidence")
    vis = entry.get("mean_landmark_visibility")
    if band and src != "missing_xyz":
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

    measured = degraded = missing = 0
    for fv_key, entry in prov.items():
        if not isinstance(entry, dict):
            continue
        src = entry.get("source")
        if src == "world_3d":
            measured += 1
        elif src == "normalized_2d":
            degraded += 1
        elif src == "missing_xyz":
            missing += 1
    low_conf = sum(1 for e in prov.values()
                   if isinstance(e, dict) and e.get("confidence") == "low")

    states = []
    if subject is False or reliable is False:
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
        if subject is False:
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


def _section(number: str, title: str, sub: str = ""):
    _html(
        f'<div class="pai-root"><div class="pai-sec">'
        f'<p class="pai-sec-title">{esc(number)} · {esc(title)}</p>'
        + (f'<p class="pai-sec-sub">{esc(sub)}</p>' if sub else "")
        + "</div></div>")


def render_delivery_status(status: DeliveryStatus, result=None) -> None:
    """01 — Can I trust this analysis? Always first."""
    _section("01", "Delivery status",
             "What the quality of this analysis is, before anything it says is read.")

    cls = {"ok": "ok", "warn": "warn", "danger": "danger"}[status.tone]
    chips = []

    if status.subject_verified is True:
        chips.append(("BOWLER CONFIRMED", "ok"))
    elif status.subject_verified is False:
        chips.append(("BOWLER NOT CONFIRMED", "danger"))
    if status.delivery_reliable is True:
        chips.append(("DELIVERY FOUND", "ok"))
    elif status.delivery_reliable is False:
        chips.append(("DELIVERY NOT FOUND", "danger"))
    if status.completeness == "complete":
        chips.append(("FULLY MEASURED", "ok"))
    elif status.completeness == "partial":
        chips.append((f"PARTIAL — {status.degraded} DEGRADED", "warn"))
    else:
        chips.append((f"LIMITED — {status.missing} UNMEASURED", "danger"))
    if status.low_confidence:
        chips.append((f"{status.low_confidence} LOW-CONFIDENCE", "warn"))
    if status.ood_features:
        chips.append(("OUTSIDE MODEL RANGE", "warn"))
    if status.bowler_track_id is not None:
        conf = status.bowler_confidence
        chips.append((f"BOWLER TRACK #{status.bowler_track_id}"
                      + (f" · {conf:.0%}" if isinstance(conf, (int, float)) else ""), "accent"))

    chip_html = "".join(
        f'<span class="pai-chip {c}">{esc(t)}</span>' for t, c in chips)

    _html(
        f'<div class="pai-root"><div class="pai-status {cls}" role="status">'
        f'<p class="pai-status-title">{esc(status.headline)}</p>'
        f'<p class="pai-status-copy">{esc(status.headline_copy)}</p>'
        f'<div class="pai-chips">{chip_html}</div>'
        f"</div></div>")

    if status.blocked_reason:
        _html(
            '<div class="pai-root"><div class="pai-status danger" role="status">'
            '<p class="pai-status-title">What was withheld</p>'
            f'<p class="pai-status-copy">{esc(status.blocked_reason)}</p>'
            "</div></div>")

    for w in status.warnings[:6]:
        st.warning(w)
    if len(status.warnings) > 6:
        st.caption(f"{len(status.warnings) - 6} further pipeline notes are listed in "
                   f"Advanced → pipeline notes.")


def render_insight_hero(result, status: DeliveryStatus, finding: Optional[Finding]) -> None:
    """02 — One main finding, why it matters, what to work on."""
    _section("02", "What this delivery tells you")

    if finding is None:
        _html('<div class="pai-root"><div class="pai-hero danger">'
              '<p class="pai-kicker">No finding was produced</p>'
              '<p class="pai-hero-main">This delivery was not analysed for technique.</p>'
              '<div class="pai-hero-row"><div class="pai-hero-cell">'
              '<p class="pai-hero-lab">Why it matters</p>'
              '<p class="pai-hero-val">The measurements below are shown so you can see what '
              'was captured, but they are not a judgement about a bowling action.</p>'
              "</div></div></div></div>")
        return

    tone = {"high": "danger", "moderate": "warn",
            "low": "warn", "positive": "ok"}.get(finding.severity, "accent")

    notes = _actionable_notes(result)
    action = finding.action or (notes[0] if notes else "")

    obs_cell = ""
    if finding.observed:
        obs_cell = (f'<div class="pai-hero-cell"><p class="pai-hero-lab">What was measured</p>'
                    f'<p class="pai-hero-val">{esc(finding.observed)}</p></div>')

    _html(
        f'<div class="pai-root"><div class="pai-hero {tone}">'
        f'<p class="pai-kicker">Main finding</p>'
        f'<p class="pai-hero-main">{esc(finding.title)}</p>'
        f'<div class="pai-hero-row">'
        f'<div class="pai-hero-cell"><p class="pai-hero-lab">Why it matters</p>'
        f'<p class="pai-hero-val">{esc(finding.why)}</p></div>'
        f"{obs_cell}"
        + (f'<div class="pai-hero-cell"><p class="pai-hero-lab">What to work on</p>'
           f'<p class="pai-hero-val">{esc(action)}</p></div>' if action else "")
        + "</div></div></div>")


def render_top_findings(findings: list) -> None:
    """03 — Up to three findings, no filler."""
    _section("03", "Top findings",
             "The strongest evidence in this delivery, most significant first.")

    if not findings:
        _html('<div class="pai-root"><div class="pai-empty">'
              '<p class="pai-empty-title">Nothing to report</p>'
              '<p class="pai-empty-copy">This delivery did not produce enough trustworthy '
              'evidence to raise a finding. See Delivery status for why.</p></div></div>')
        return

    for f in findings:
        rows = ""
        if f.observed:
            rows += (f'<div class="pai-ev"><span class="pai-ev-lab">Observed</span>'
                     f'<span class="pai-ev-val">{esc(f.observed)}</span></div>')
        rows += (f'<div class="pai-ev"><span class="pai-ev-lab">Why it matters</span>'
                 f'<span class="pai-ev-val">{esc(f.why)}</span></div>')
        if f.action:
            rows += (f'<div class="pai-ev"><span class="pai-ev-lab">Action</span>'
                     f'<span class="pai-ev-val">{esc(f.action)}</span></div>')
        _html(
            f'<div class="pai-root"><div class="pai-finding">'
            f'<div class="pai-finding-top">'
            f'<p class="pai-finding-title">{esc(f.title)}</p>'
            f'<span class="pai-finding-sev {esc(f.severity)}">{esc(f.severity)}</span>'
            f"</div>{rows}</div></div>")


def render_replay(replay_fn: Optional[Callable], replay_path, result,
                  feature_vector: dict, ball_stats: dict) -> None:
    """04 — Visual centrepiece. Delegates to the existing replay renderer."""
    _section("04", "Analysis replay",
             "The evidence behind the findings above. Bowler box, ball track, pose "
             "skeleton and release marker are burned into this render.")
    if replay_fn is None:
        st.caption("Replay is unavailable in this view.")
        return
    replay_fn(replay_path, result, feature_vector, ball_stats)


def render_no_replay() -> None:
    """04 for entries with no clip behind them. States the absence, never fakes it."""
    _section("04", "Analysis replay",
             "The evidence behind the findings above.")
    _html('<div class="pai-root"><div class="pai-empty">'
          '<p class="pai-empty-title">No replay for this entry</p>'
          '<p class="pai-empty-copy">These values were typed in rather than measured '
          'from a clip, so there is no video to trace the findings back to. Upload a '
          'bowling video to get the pose overlay, ball track and release marker.</p>'
          "</div></div>")


def render_key_measurements(measurements: list) -> None:
    """05 — Only the measurements, grouped, each with its source type."""
    _section("05", "Key measurements",
             "Each value carries where it came from, so a measured angle and a "
             "placeholder are never confused.")
    groups = []
    for g in GROUPS:
        items = [m for m in measurements if m.group == g]
        if items:
            groups.append((g, items))

    blocks = []
    for g, items in groups:
        rows = []
        for m in items:
            if m.value is None:
                val = f'<span class="pai-metric-val absent">{esc(m.source_label.split(" —")[0])}</span>'
            else:
                val = f'<span class="pai-metric-val">{esc(fmt(m.value, m.unit))}</span>'
            ctx = f' · {esc(m.context)}' if m.context else ""
            rows.append(
                f'<div class="pai-metric" title="{esc(m.plain)}">'
                f'<div class="pai-metric-top">'
                f'<span class="pai-metric-name">{esc(m.label)}</span>{val}</div>'
                f'<p class="pai-metric-srcline">'
                f'<span class="{esc(m.source_class)}">{esc(m.source_label)}</span>'
                f'{esc(m.source_note)}{ctx}</p></div>')
        blocks.append(f'<div class="pai-mgroup"><p class="pai-mgroup-name">{esc(g)}</p>'
                      + "".join(rows) + "</div>")

    _html(f'<div class="pai-root"><div class="pai-metrics">'
          + "".join(blocks) + "</div></div>")


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
            f'<div class="pai-chain"><span class="pai-chain-step">{i + 1}</span>'
            f'<div class="pai-chain-body"><p class="pai-chain-lab">{esc(lab)}</p>'
            f'<p class="pai-chain-val">{esc(val)}</p></div></div>'
            for i, (lab, val) in enumerate(chain))
        _html(f'<div class="pai-root"><div class="pai-evidence">{steps}</div></div>')


def render_coaching_plan(result, status: DeliveryStatus) -> None:
    """07 — Every note the pipeline produced, as an ordered plan."""
    _section("07", "Coaching plan",
             "Produced by this run's threshold checks. Nothing here is generic advice.")
    notes = _actionable_notes(result)

    if not notes:
        if status.state in ("UNRELIABLE", "SCORING_WITHHELD"):
            _html('<div class="pai-root"><div class="pai-empty">'
                  '<p class="pai-empty-title">No coaching plan produced</p>'
                  '<p class="pai-empty-copy">The pipeline did not generate coaching '
                  'recommendations for this delivery'
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

    for i, note in enumerate(notes[:6]):
        pri = f"p{i + 1}" if i < 2 else ""
        _html(
            f'<div class="pai-root"><div class="pai-plan">'
            f'<div class="pai-plan-head">'
            f'<span class="pai-plan-pri {pri}">Priority {i + 1}</span>'
            f'<span class="pai-plan-title">{esc(note)}</span>'
            f"</div></div></div>")
    if len(notes) > 6:
        st.caption(f"{len(notes) - 6} further notes are in Advanced → coaching output.")


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
        interval_html = (f'<p class="pai-small-note">Model uncertainty: 68% of predictions '
                         f'for inputs like this fall between {interval[0]:.0f} and '
                         f'{interval[1]:.0f}. This is a band on the model, not a confidence '
                         f'interval on a physical measurement.</p>')

    _html(
        f'<div class="pai-root"><div class="pai-small">'
        f'<div class="pai-small-grid">'
        f'<div class="pai-small-cell"><p class="pai-small-lab">Demo score</p>'
        f'<p class="pai-small-val">{score:.0f}<span style="font-size:.9rem;color:{PAI_MUTED}">'
        f" / 100</span></p></div>"
        f'<div class="pai-small-cell"><p class="pai-small-lab">Reading</p>'
        f'<p class="pai-small-val" style="font-size:.92rem">{esc(band)}</p></div>'
        f"</div>"
        f'<p class="pai-small-note">{esc(ctx)}</p>{interval_html}'
        f"</div></div>")


def render_risk_indicators(result, status: DeliveryStatus) -> None:
    """09 — Screening context. Never a diagnosis, never a score when withheld."""
    _section("09", "Biomechanical risk indicators",
             "Published screening bands, checked against this delivery's measurements. "
             "Screening only — not a medical assessment and not an injury prediction.")
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
            triggers = "".join(
                f'<div class="pai-ev"><span class="pai-ev-lab">Triggered factor</span>'
                f'<span class="pai-ev-val">{esc(t)}</span></div>'
                for t in (c.get("trigger_detected") or []))
            incidence = (f'<div class="pai-ev"><span class="pai-ev-lab">Reported incidence</span>'
                         f'<span class="pai-ev-val">{esc(c.get("clinical_incidence"))}</span></div>'
                         if c.get("clinical_incidence") else "")
            recovery = (f'<div class="pai-ev"><span class="pai-ev-lab">Recovery context</span>'
                        f'<span class="pai-ev-val">{esc(c.get("est_recovery_timeline"))}</span></div>'
                        if c.get("est_recovery_timeline") else "")
            limitation = ('<div class="pai-ev"><span class="pai-ev-lab">Limitation</span>'
                          '<span class="pai-ev-val">A screening band, not a diagnosis. '
                          'Reference values in this build are pending source verification '
                          '(see CITATION.md).</span></div>')
            _html(
                f'<div class="pai-root"><div class="pai-status {tone}">'
                f'<p class="pai-status-title">{esc(c.get("injury"))}</p>'
                f'<p class="pai-status-copy">{esc(c.get("anatomical_site"))}</p>'
                f"{triggers}{incidence}{recovery}{limitation}</div></div>")

    risk = getattr(result, "injury_risk", None)
    fv = getattr(result, "feature_vector", None) or {}

    model_block = ""
    if risk is None:
        model_block = (
            '<p class="pai-kicker" style="margin-top:16px">Model risk indicator</p>'
            '<p class="pai-empty-copy">No model risk-indicator output was produced for this '
            'delivery, so no risk level or probability is shown. Nothing is inferred in its '
            'place.</p>')
    else:
        level = _risk_level(result) or ""
        tone = "ok" if level == "low" else ("warn" if level == "moderate" else "danger")
        model_block = (
            f'<p class="pai-kicker" style="margin-top:18px">Model biomechanical risk indicator</p>'
            f'<div class="pai-status {tone}">'
            f'<p class="pai-status-title">{esc(level.upper() or "UNAVAILABLE")}</p>'
            f'<p class="pai-status-copy">Demo model output. Its training labels were derived '
            f'from the benchmark thresholds, so it largely re-states the screening result '
            f'above rather than adding independent evidence. It is a screening context '
            f'indicator, not a diagnosis, and not an injury prediction.</p>'
            f"</div>")

    _html(f'<div class="pai-root">{model_block}</div>')


def render_advanced(result, *, perf_bundle=None, injury_bundle=None,
                    status: Optional[DeliveryStatus] = None,
                    extra_renderers: Optional[dict] = None) -> None:
    """10 — Everything technical, collapsed."""
    with st.expander("Advanced analysis — full technical detail, diagnostics and export",
                     expanded=False):
        _section("10", "Advanced analysis",
                 "Model internals, SHAP contributions, the full reference table and "
                 "run diagnostics. Collapsed by default — none of this is needed to "
                 "act on the findings above.")
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
    """Orchestrator for sections 01-11. Returns the computed status."""
    _html(RESULT_CSS)
    fv = dict(getattr(result, "feature_vector", None) or {})
    bs = dict(ball_stats or {})

    status = assess_delivery(result, perf_bundle=perf_bundle, is_video=is_video)
    findings = build_findings(result, status)
    measurements = build_measurements(result, is_video=is_video)
    chains = build_evidence(result, findings, status)

    st.markdown("---")
    render_delivery_status(status, result)
    render_insight_hero(result, status, findings[0] if findings else None)
    render_top_findings(findings)
    if is_video or replay_path:
        render_replay(replay_fn, replay_path, result, fv, bs)
    else:
        # The hierarchy stays fixed at 11 sections. Silently dropping the replay
        # would leave the user wondering whether evidence was withheld.
        render_no_replay()
    render_key_measurements(measurements)
    render_evidence_panel(chains, status)
    render_coaching_plan(result, status)
    render_performance_indicator(result, perf_bundle, _lazy_ml_models())
    render_risk_indicators(result, status)
    render_advanced(result, perf_bundle=perf_bundle, injury_bundle=injury_bundle,
                    status=status, extra_renderers=advanced_renderers)
    render_history_export(result, session_writer)
    return status


def _lazy_ml_models():
    try:
        from . import ml_models
        return ml_models
    except Exception:
        return None
