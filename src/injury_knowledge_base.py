"""
Clinical knowledge base & biomechanical risk evaluator.

Benchmark data (shipped alongside this module):
  - data/cricket_injury_recovery_benchmarks.json  (machine-readable thresholds)
  - data/cricket_injury_recovery_benchmarks.csv   (tabular, for spreadsheets)

Core API:
  - assess_biomechanical_risks(features, ...) -> list of detected risk dicts
  - workload_risk(acwr=None, seven_day_load=None, rest_days=None) -> workload checks
  - all_benchmarks() -> full injury benchmark table (for dashboards)
  - map_from_pipeline_features(feature_vector) -> clinical-named feature dict
  - benchmark_rows() -> benchmark table as list of row dicts

Clinical feature names expected by `assess_biomechanical_risks`:
  shoulder_counter_rotation (deg), lateral_trunk_flexion (deg),
  knee_angle_ffc (deg, 180 = straight), stride_length_norm (x stature),
  angular_velocity_deg_s, shoulder_abduction_deg, continuous_overs,
  seven_day_ball_load.

WARNING: These are literature-derived benchmark ranges for screening and
coaching purposes, NOT medical diagnoses or individual prognoses.
"""
import json
import os

from . import config

_BENCH_PATH = os.path.join(config.DATA_DIR, "cricket_injury_recovery_benchmarks.json")

_TRIGGER_LABELS = {
    "shoulder_counter_rotation": "Shoulder counter-rotation",
    "lateral_trunk_flexion": "Lateral trunk flexion",
    "knee_angle_ffc": "Knee angle at front-foot contact",
    "stride_length_norm": "Stride length",
    "angular_velocity_deg_s": "Peak angular velocity",
    "shoulder_abduction_deg": "Shoulder abduction",
    "continuous_overs": "Continuous overs in spell",
    "seven_day_ball_load": "7-day bowling load",
}

_KB = None


def _kb() -> dict:
    global _KB
    if _KB is None:
        if not os.path.exists(_BENCH_PATH):
            raise FileNotFoundError(
                f"Benchmark data not found: {_BENCH_PATH}. Expected it in "
                f"{config.DATA_DIR} (data/cricket_injury_recovery_benchmarks.json)."
            )
        with open(_BENCH_PATH, encoding="utf-8") as f:
            _KB = json.load(f)
    return _KB


def _num(features: dict, key: str):
    try:
        val = features[key]
        if val is None:
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None


def _trigger_label(name: str) -> str:
    return _TRIGGER_LABELS.get(name, name.replace("_", " ").title())


def _format_hit(name: str, value: float, threshold: float, unit: str, condition: str) -> str:
    op = condition
    unit_str = unit or ""
    if name in ("continuous_overs", "seven_day_ball_load"):
        return f"{_trigger_label(name)} {value:.0f} {unit_str} ({op} {threshold:.0f})"
    return f"{_trigger_label(name)} {value:.2f} {unit_str} ({op} {threshold:.2f})"


def _check_triggers(features: dict, triggers: dict) -> list:
    """Return list of (name, value, threshold, unit, condition) hits, in trigger-definition order."""
    hits = []
    for name, spec in triggers.items():
        v = _num(features, name)
        if v is None:
            continue
        th = spec["threshold"]
        cond = spec.get("condition", ">")
        if (v > th) if cond == ">" else (v < th):
            hits.append((name, v, th, spec.get("unit", ""), cond))
    return hits


def _severity(hits: list) -> str:
    if not hits:
        return "Low"
    worst = max(abs(v - th) / max(abs(th), 1e-9) for _n, v, th, _u, _c in hits)
    if len(hits) >= 2 or worst >= 0.25:
        return "High"
    return "Moderate"


def assess_biomechanical_risks(features: dict, *, acwr=None,
                               seven_day_load=None, rest_days=None) -> list:
    """
    Evaluate a delivery's features against the clinical benchmark thresholds.

    `features` should use the clinical names above (see
    `map_from_pipeline_features` to convert from pipeline feature vectors).
    Optional workload parameters feed the same thresholds used by
    `workload_risk` (e.g. a 7-day ball count can trigger the lumbar check).

    Returns a list of dicts, one per detected risk, sorted most-severe /
    longest-recovery first. Each dict:
      injury, key, anatomical_site, clinical_incidence,
      trigger_detected (list[str]), severity (Low/Moderate/High),
      est_recovery_timeline, median_days_to_match, avg_days_to_return.
    """
    f = dict(features or {})
    if seven_day_load is not None:
        f["seven_day_ball_load"] = seven_day_load

    results = []
    for entry in _kb().get("injuries", []):
        hits = _check_triggers(f, entry.get("primary_triggers", {}))
        if not hits:
            continue
        results.append({
            "injury": entry["injury"],
            "key": entry.get("key"),
            "anatomical_site": entry.get("anatomical_site"),
            "clinical_incidence": entry.get("clinical_incidence"),
            "trigger_detected": [_format_hit(*h) for h in hits],
            "severity": _severity(hits),
            "est_recovery_timeline": entry.get("recovery_window"),
            "median_days_to_match": entry.get("median_days_to_match"),
            "avg_days_to_return": entry.get("avg_days_to_return"),
        })

    results.sort(key=lambda r: (r["median_days_to_match"] or 0), reverse=True)
    return results


def workload_risk(acwr=None, seven_day_load=None, rest_days=None) -> list:
    """
    Evaluate workload / acute:chronic workload ratio (ACWR) benchmarks.

    Accepts any subset of `acwr` (float), `seven_day_load` (balls in last 7
    days), `rest_days` (days between bowling spells). Returns a list of dicts
    with keys: check, status (at_risk/warning/ok), detail, and multipliers.
    """
    wl = _kb().get("workload", {})
    checks = []

    if acwr is not None:
        lo, hi = wl["acwr_sweet_spot"]
        if acwr > wl["acwr_high_risk_above"]:
            mult = wl["acwr_high_risk_injury_multiplier"]
            checks.append({
                "check": "ACWR spike",
                "status": "at_risk",
                "detail": f"ACWR {acwr:.2f} > {wl['acwr_high_risk_above']:.2f} "
                          f"(injury likelihood {mult[0]}x-{mult[1]}x higher)",
            })
        elif acwr > hi:
            checks.append({
                "check": "ACWR above sweet spot",
                "status": "warning",
                "detail": f"ACWR {acwr:.2f} above sweet spot {hi:.2f} -- reduce acute load",
            })
        else:
            checks.append({
                "check": "ACWR sweet spot",
                "status": "ok",
                "detail": f"ACWR {acwr:.2f} within {lo:.2f}-{hi:.2f} (optimal stimulus)",
            })

    if seven_day_load is not None:
        if seven_day_load > wl["seven_day_spike_balls"]:
            checks.append({
                "check": "7-day delivery spike",
                "status": "at_risk",
                "detail": f"{seven_day_load:.0f} balls > {wl['seven_day_spike_balls']} -- "
                          f"~{wl['seven_day_spike_lumbar_risk_multiplier']}x lumbar stress-"
                          f"fracture risk vs <{wl['seven_day_safe_balls']} balls",
            })
        elif seven_day_load > wl["seven_day_safe_balls"]:
            checks.append({
                "check": "7-day delivery load",
                "status": "warning",
                "detail": f"{seven_day_load:.0f} balls approaching spike threshold "
                          f"({wl['seven_day_spike_balls']})",
            })

    if rest_days is not None:
        if rest_days < wl["rest_days_minimum"]:
            checks.append({
                "check": "Rest days between spells",
                "status": "at_risk",
                "detail": f"{rest_days} rest day(s) < {wl['rest_days_minimum']} -- "
                          f"{wl['rest_days_short_injury_multiplier']}x higher injury rate",
            })
        else:
            checks.append({
                "check": "Rest days between spells",
                "status": "ok",
                "detail": f"{rest_days} rest day(s) >= {wl['rest_days_minimum']} (recommended)",
            })

    return checks


def all_benchmarks() -> list:
    """Full injury benchmark table (injury, site, incidence, triggers, recovery)."""
    out = []
    for entry in _kb().get("injuries", []):
        triggers = "; ".join(
            f"{_trigger_label(n)} {s.get('condition', '>')} {s['threshold']} {s.get('unit', '')}"
            for n, s in entry.get("primary_triggers", {}).items()
        )
        out.append({
            "injury": entry["injury"],
            "anatomical_site": entry.get("anatomical_site"),
            "clinical_incidence": entry.get("clinical_incidence"),
            "primary_triggers": triggers,
            "avg_days_to_return": entry.get("avg_days_to_return"),
            "median_days_to_match": entry.get("median_days_to_match"),
            "recovery_window": entry.get("recovery_window"),
        })
    return out


def map_from_pipeline_features(feature_vector: dict) -> dict:
    """
    Convert a pipeline feature vector (config.FEATURE_NAMES) into the clinical
    names used by the benchmark thresholds.

    Notes:
      - knee_angle_ffc = 180 - knee_flexion_deg (0 deg flexion = straight leg).
      - stride_length_norm is already normalized to bowler stature.
      - Shoulder abduction / continuous overs are not measured by the CV
        pipeline and are left unset (those triggers stay silent unless you
        supply them).
    """
    fv = feature_vector or {}
    knee_flex = fv.get("knee_flexion_deg")
    return {
        "shoulder_counter_rotation": fv.get("shoulder_rotation_deg"),
        "lateral_trunk_flexion": fv.get("trunk_lean_deg"),
        "knee_angle_ffc": (180.0 - knee_flex) if knee_flex is not None else None,
        "stride_length_norm": fv.get("stride_length_norm"),
        "angular_velocity_deg_s": fv.get("angular_velocity_deg_s"),
        "elbow_flexion": fv.get("elbow_flexion_deg"),
        "release_angle": fv.get("release_angle_deg"),
        "ground_contact_time": fv.get("ground_contact_time_s"),
    }
