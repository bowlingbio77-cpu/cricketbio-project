"""
PaceAI P1 -- Independent Ground-Truth Validation Evaluator.

Compares INDEPENDENT human/reference measurements (evaluation/ground_truth/
ground_truth.csv) against PaceAI pipeline predictions. The two inputs are
completely separate:
  * ground truth  -> produced by human annotators via evaluation/annotator.py
  * predictions   -> produced by the PaceAI pipeline, cached per video in
                     evaluation/predictions/<video_id>_pred.json
This module never writes ground truth, never derives one from the other, and
never fills in missing values.

Scientific grounding
---------------------
All statistical methods follow the protocol in
evaluation/ground_truth/ANNOTATION_PROTOCOL.md. Statistics are reported ONLY
when a minimum number of valid paired samples exist; otherwise the evaluator
returns "insufficient data" rather than a misleading number.

Usage:
    python evaluation/ground_truth_validation.py                 # write reports
    python evaluation/ground_truth_validation.py --predictions_dir <dir>
    python -m pytest tests/test_ground_truth_validation.py       # tests
"""
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(EVAL_DIR, "ground_truth")
GT_PATH = os.path.join(GT_DIR, "ground_truth.csv")
PREDICTIONS_DIR = os.path.join(EVAL_DIR, "predictions")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

PROTOCOL_PATH = os.path.join(GT_DIR, "ANNOTATION_PROTOCOL.md")
INVENTORY_PATH = os.path.join(GT_DIR, "video_inventory.csv")

# Reproducibility (ANNOTATION_PROTOCOL.md, section on reproducibility):
# every annotation row records the protocol version it was produced under.
ANNOTATION_PROTOCOL_VERSION = "1.1"
ANNOTATION_PROTOCOL_DATE = "2026-09-11"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

GT_COLUMNS = [
    "video_id",
    "delivery_id",
    "annotator_id",
    "annotation_protocol_version",
    "annotation_date",
    "release_frame",
    "front_contact_frame",
    "front_knee_angle_deg",
    "elbow_flexion_deg",
    "trunk_lean_deg",
    "stride_length_px",
    "release_angle_deg",
    "release_speed_mps",
    "visibility_release",
    "visibility_knee",
    "visibility_elbow",
    "visibility_trunk",
    "annotation_confidence",
    "notes",
]

# All measurement columns one annotator row is expected to fill (or leave
# blank-with-reason). Used by the annotation-status report to list what is
# still missing.
MEASUREMENT_FIELDS = [
    "release_frame",
    "front_contact_frame",
    "front_knee_angle_deg",
    "elbow_flexion_deg",
    "trunk_lean_deg",
    "stride_length_px",
    "release_angle_deg",
    "release_speed_mps",
]

# Physically meaningful bounds for the numeric measurements (unit: degrees or
# pixels). Values outside these are reported, never silently fixed.
ANGLE_BOUNDS = {
    "front_knee_angle_deg": (0.0, 180.0),
    "elbow_flexion_deg": (0.0, 180.0),
    "trunk_lean_deg": (0.0, 90.0),        # upright = 0, horizontal = 90
    "release_angle_deg": (0.0, 90.0),     # positive = downward from horizontal
    "stride_length_px": (0.0, None),      # non-negative planar distance
}

VALID_VISIBILITY = ["good", "partial", "occluded", "out_of_frame", "not_established"]

# Continuous kinematic features that can be compared GT-vs-Prediction, with the
# matching PaceAI prediction key (see evaluation/runs/after/<clip>.json).
CONTINUOUS_FEATURES = {
    "front_knee_angle_deg": "knee_flexion_deg",
    "elbow_flexion_deg": "elbow_flexion_deg",
    "trunk_lean_deg": "trunk_lean_deg",
    "release_angle_deg": "release_angle_deg",
}

# Frame-event comparisons (prediction key aliases accepted).
FRAME_EVENTS = {
    "release_frame": ("release_frame", "release_idx"),
    "front_contact_frame": ("front_contact_frame", "front_foot_contact_frame"),
}

# Features present in the schema but NOT numerically validatable with the
# current prediction output, and why.
EXCLUDED_FEATURES = {
    "stride_length_px": (
        "unit mismatch: ground truth is planar pixels, PaceAI estimates a "
        "normalized value. Comparable only after a shared normalized unit is "
        "agreed in the protocol."
    ),
    "release_speed_mps": (
        "not measurable from uncalibrated camera footage (needs pixel-per-metre "
        "calibration) and not produced by the PaceAI prediction cache."
    ),
}

# Minimum valid paired samples per statistic. Below these we return
# "insufficient data" instead of a misleading number.
MIN_SAMPLES_DESCRIPTIVE = 1   # bias / MAE / RMSE
MIN_SAMPLES_SD = 2            # std of error
MIN_SAMPLES_LOA = 3           # Bland-Altman limits of agreement
MIN_SAMPLES_CORRELATION = 4   # Pearson r, R^2
MIN_SAMPLES_ICC = 3           # ICC(2,1)

REFERENCE_ANNOTATOR = "adjudicator"


# --------------------------------------------------------------------------- #
# Ground-truth parsing
# --------------------------------------------------------------------------- #

def _to_number(value) -> Optional[float]:
    """Coerce a CSV cell to float; returns None for blank/invalid."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "unknown":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _valid_frame(value) -> Optional[int]:
    """Coerce a frame cell to a non-negative int; None if blank/invalid."""
    num = _to_number(value)
    if num is None or num < 0:
        return None
    return int(round(num))


def parse_ground_truth_csv(path: str) -> Tuple[List[dict], List[str]]:
    """Parse ground_truth.csv into records.

    Returns (records, errors). Records keep raw values where present; numeric
    cells that cannot be parsed are recorded as None with an error string so a
    missing vs invalid distinction stays visible and nothing is invented.
    """
    if not os.path.exists(path):
        return [], ["ground_truth.csv not found"]
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], ["empty ground_truth.csv (no header)"]
        rows = list(reader)

    errors: List[str] = []
    records: List[dict] = []
    for i, raw in enumerate(rows):
        video_id = (raw.get("video_id") or "").strip()
        if not video_id:
            errors.append(f"row {i + 2}: missing video_id -- skipped")
            continue
        record = {c: (raw.get(c) or "").strip() for c in GT_COLUMNS}

        # Frame fields: coerce to a non-negative int. A present-but-invalid
        # value is reported loudly, never silently "corrected" or ignored.
        for field in ["release_frame", "front_contact_frame"]:
            raw_val = record[field]
            parsed = _valid_frame(raw_val)
            if raw_val != "" and parsed is None:
                errors.append(
                    f"row {i + 2}: invalid {field} value {raw_val!r} -- recorded "
                    f"as missing (no value invented)")
            record[field] = parsed

        # numeric continuous fields (blank -> missing; invalid -> missing + reported)
        numeric_fields = [
            "front_knee_angle_deg", "elbow_flexion_deg", "trunk_lean_deg",
            "stride_length_px", "release_angle_deg", "release_speed_mps",
        ]
        for field in numeric_fields:
            raw_val = record[field]
            record[field] = _to_number(raw_val)
            if raw_val != "" and record[field] is None:
                errors.append(
                    f"row {i + 2}: invalid numeric {field} value {raw_val!r} -- "
                    f"recorded as missing (no value invented)")

        if not record["video_id"]:
            errors.append(f"row {i + 2}: empty video_id -- skipped")
            continue
        records.append(record)
    return records, errors


def load_clip_metadata(inventory_path: str = None) -> Dict[str, dict]:
    """Read per-clip probe metadata (fps, frame_count, ...) from the inventory.

    Returns {video_id: {fps, frame_count, ...}}. Missing or unreadable values
    are dropped so frame-bound checks are only performed when a real count
    exists -- the pipeline never invents a frame count.
    """
    inventory_path = inventory_path or INVENTORY_PATH
    meta: Dict[str, dict] = {}
    if not os.path.exists(inventory_path):
        return meta
    with open(inventory_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue
            entry = {"fps": None, "frame_count": None}
            fps = _to_number(row.get("fps"))
            if fps is not None and fps > 0:
                entry["fps"] = fps
            fc = _to_number(row.get("frame_count"))
            if fc is not None and fc > 0:
                entry["frame_count"] = int(round(fc))
            meta[video_id] = entry
    return meta


def validate_annotations(records: List[dict],
                         clip_metadata: Optional[Dict[str, dict]] = None,
                         ) -> List[dict]:
    """Report data-quality issues in ground-truth rows (never fixes them).

    Checks (per ANNOTATION_PROTOCOL.md):
      * every row has an annotator_id (required for inter-rater reliability)
      * no duplicate (video_id, delivery_id, annotator_id) rows
      * frame events are >= 0 and within the clip's frame count when known
      * fps metadata is > 0
      * numeric measurements stay within physically meaningful bounds
      * visibility flags use the documented vocabulary
    Returns a list of issue dicts:
        {"row": "video_id/delivery_id/annotator_id", "field", "code", "message"}
    """
    issues: List[dict] = []
    clip_metadata = clip_metadata or {}

    def add(row, field, code, message):
        issues.append({
            "row": f"{row.get('video_id')}/{row.get('delivery_id')}/{row.get('annotator_id')}",
            "field": field, "code": code, "message": message,
        })

    seen_keys = set()
    for r in records:
        key = (r.get("video_id"), r.get("delivery_id"), r.get("annotator_id"))
        if not r.get("annotator_id"):
            add(r, "annotator_id", "missing_annotator_id",
                "annotator_id is required for inter-rater reliability")
        if key in seen_keys:
            add(r, "row", "duplicate_row",
                f"duplicate (video_id, delivery_id, annotator_id) row {key}")
        seen_keys.add(key)

        meta = clip_metadata.get(r.get("video_id")) or {}
        frame_count = meta.get("frame_count")
        for field in ["release_frame", "front_contact_frame"]:
            val = r.get(field)
            if val is None:
                continue
            if val < 0:
                add(r, field, "negative_frame",
                    f"{field} = {val} is negative")
            elif frame_count is not None and val >= frame_count:
                add(r, field, "frame_out_of_range",
                    f"{field} = {val} >= clip frame_count {frame_count}")

        for field, (lo, hi) in ANGLE_BOUNDS.items():
            val = r.get(field)
            if val is None:
                continue
            if lo is not None and val < lo:
                add(r, field, "out_of_bounds",
                    f"{field} = {val} below physical bound {lo}")
            elif hi is not None and val > hi:
                add(r, field, "out_of_bounds",
                    f"{field} = {val} above physical bound {hi}")

        for field in ["visibility_release", "visibility_knee",
                      "visibility_elbow", "visibility_trunk"]:
            val = (r.get(field) or "").strip()
            if val and val not in VALID_VISIBILITY:
                add(r, field, "invalid_visibility",
                    f"{field} = {val!r} not in {VALID_VISIBILITY}")

        conf = (r.get("annotation_confidence") or "").strip()
        if conf and (not conf.isdigit() or not 1 <= int(conf) <= 5):
            add(r, "annotation_confidence", "invalid_confidence",
                f"annotation_confidence = {conf!r} not in 1..5")

    for video_id, meta in clip_metadata.items():
        if meta.get("fps") is None or meta.get("fps") <= 0:
            add({"video_id": video_id}, "fps", "invalid_fps",
                f"clip {video_id} has no positive FPS metadata")
    return issues


def load_prediction(video_id: str, predictions_dir: str = PREDICTIONS_DIR) -> Optional[dict]:
    """Load the cached PaceAI prediction for a video (None if missing)."""
    path = os.path.join(predictions_dir, f"{video_id}_pred.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_frame(pred: dict, gt_feature: str) -> Optional[int]:
    """Extract a predicted frame-event value from a prediction dict."""
    if pred is None:
        return None
    biomech = pred.get("biomechanics") or {}
    aliases = FRAME_EVENTS.get(gt_feature, ())
    for alias in aliases:
        val = pred.get(alias)
        if val is None:
            val = biomech.get(alias)
        val = _valid_frame(val)
        if val is not None:
            return val
    return None


def prediction_feature(pred: dict, gt_feature: str) -> Optional[float]:
    """Extract a predicted continuous kinematic feature from a prediction dict."""
    if pred is None:
        return None
    key = CONTINUOUS_FEATURES.get(gt_feature)
    if key is None:
        return None
    biomech = pred.get("biomechanics") or {}
    return _to_number(biomech.get(key))


# --------------------------------------------------------------------------- #
# Reference-row selection (multi-annotator support)
# --------------------------------------------------------------------------- #

def reference_rows(records: List[dict]) -> List[dict]:
    """Select the reference rows for GT-vs-Prediction comparison.

    If any row for a clip carries annotator_id == 'adjudicator', those rows
    alone are the reference. Otherwise all independently annotated rows are
    kept (each is an independent sample).
    """
    adjudicated_ids = {r["video_id"] for r in records
                       if r.get("annotator_id") == REFERENCE_ANNOTATOR}
    if not adjudicated_ids:
        return records
    return [r for r in records
            if r["video_id"] in adjudicated_ids
            and r.get("annotator_id") == REFERENCE_ANNOTATOR]


# --------------------------------------------------------------------------- #
# Pairing ground truth vs predictions
# --------------------------------------------------------------------------- #

def paired_samples(records: List[dict],
                   predictions: Dict[str, Optional[dict]],
                   gt_feature: str,
                   ) -> List[Tuple[str, float, float]]:
    """Return (video_id, gt_value, pred_value) triples for a feature.

    A pair is used only when BOTH the ground-truth value and the predicted
    value exist and are finite. Nothing is interpolated or imputed.
    """
    if gt_feature in FRAME_EVENTS:
        samples = []
        for r in reference_rows(records):
            gt_val = r.get(gt_feature)
            if gt_val is None:
                continue
            pred = predictions.get(r["video_id"]) if predictions is not None else None
            p_val = prediction_frame(pred, gt_feature)
            if p_val is None:
                continue
            samples.append((r["video_id"], float(gt_val), float(p_val)))
        return samples

    if gt_feature not in CONTINUOUS_FEATURES:
        return []
    samples = []
    for r in reference_rows(records):
        gt_val = r.get(gt_feature)
        if gt_val is None or not np.isfinite(gt_val):
            continue
        pred = predictions.get(r["video_id"]) if predictions is not None else None
        p_val = prediction_feature(pred, gt_feature)
        if p_val is None or not np.isfinite(p_val):
            continue
        samples.append((r["video_id"], float(gt_val), float(p_val)))
    return samples


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def icc_two_way_random_single(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    """ICC(2,1) -- two-way random-effects, single measures (Shrout & Fleiss).

    Treats (ground truth, prediction) as two raters and each delivery as a
    target. Returns None when the model is degenerate (zero variance), which
    happens when every target has identical ratings.
    """
    n = len(gt)
    k = 2
    if n < 2:
        return None
    x = np.column_stack([gt, pred])
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_total = float(((x - grand) ** 2).sum())
    ss_target = float((k * (row_means - grand) ** 2).sum())
    ss_rater = float((n * (col_means - grand) ** 2).sum())
    ss_resid = ss_total - ss_target - ss_rater

    ms_target = ss_target / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_resid = ss_resid / max(1, (n - 1) * (k - 1))

    denom = ms_target + (k - 1) * ms_resid + k * (ms_rater - ms_resid) / n
    if denom <= 0:
        return None
    icc = (ms_target - ms_resid) / denom
    return float(icc)


def bland_altman(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Bland-Altman difference vs mean; returns LoA stats."""
    mean = (gt + pred) / 2.0
    diff = pred - gt
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) >= 2 else None
    if sd is None:
        return {"bias": bias, "sd_diff": None,
                "loa_lower": None, "loa_upper": None}
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
    }


def compute_feature_agreement(gt_feature: str,
                              samples: List[Tuple[str, float, float]],
                              ) -> dict:
    """Compute agreement metrics for one feature across paired samples."""
    n = len(samples)
    result = {
        "feature": gt_feature,
        "n": n,
        "status": "measured" if n > 0 else "no_paired_samples",
        "bias": None, "mae": None, "rmse": None, "sd_error": None,
        "pearson_r": None, "r2": None,
        "loa_lower": None, "loa_upper": None,
        "icc21": None,
        "statistics_status": "measured" if n > 0 else "insufficient data",
    }
    if n == 0:
        return result

    gt = np.array([s[1] for s in samples], dtype=float)
    pred = np.array([s[2] for s in samples], dtype=float)
    err = pred - gt

    if n >= MIN_SAMPLES_DESCRIPTIVE:
        result["bias"] = float(np.mean(err))
        result["mae"] = float(np.mean(np.abs(err)))
        result["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    if n >= MIN_SAMPLES_SD:
        result["sd_error"] = float(np.std(err, ddof=1))

    if n >= MIN_SAMPLES_CORRELATION:
        if float(np.std(gt)) > 0 and float(np.std(pred)) > 0:
            r = float(np.corrcoef(gt, pred)[0, 1])
            result["pearson_r"] = r
            result["r2"] = r * r

    if n >= MIN_SAMPLES_LOA:
        ba = bland_altman(gt, pred)
        result["loa_lower"] = ba["loa_lower"]
        result["loa_upper"] = ba["loa_upper"]

    if n >= MIN_SAMPLES_ICC:
        result["icc21"] = icc_two_way_random_single(gt, pred)

    return result


def compute_frame_event_metrics(gt_feature: str,
                                samples: List[Tuple[str, float, float]],
                                fps: float) -> dict:
    """Frame-event (release/front-contact) error summary incl. milliseconds."""
    n = len(samples)
    result = {
        "feature": gt_feature,
        "n": n,
        "fps": fps,
        "status": "measured" if n > 0 else "no_paired_samples",
        "absolute_errors": [] if n else None,
        "errors_ms": [] if n else None,
        "mean_abs_frame_error": None,
        "median_abs_frame_error": None,
        "mean_signed_frame_error": None,
        "median_signed_frame_error": None,
        "mean_abs_error_ms": None,
        "median_abs_error_ms": None,
    }
    if n == 0:
        return result
    abs_err = np.abs(np.array([s[1] - s[2] for s in samples], dtype=float))
    signed_err = np.array([s[1] - s[2] for s in samples], dtype=float)
    fps = fps if fps and fps > 0 else 1.0
    result["absolute_errors"] = [int(e) for e in abs_err]
    result["errors_ms"] = [round(float(e / fps * 1000.0), 2) for e in abs_err]
    result["mean_abs_frame_error"] = float(abs_err.mean())
    result["median_abs_frame_error"] = float(np.median(abs_err))
    result["mean_signed_frame_error"] = float(signed_err.mean())
    result["median_signed_frame_error"] = float(np.median(signed_err))
    result["mean_abs_error_ms"] = float((abs_err / fps * 1000.0).mean())
    result["median_abs_error_ms"] = float(np.median(abs_err / fps * 1000.0))
    return result


# --------------------------------------------------------------------------- #
# Prediction cache writer (from existing PaceAI run output)
# --------------------------------------------------------------------------- #

def build_predictions_from_runs(runs_dir: Optional[str] = None,
                                predictions_dir: str = PREDICTIONS_DIR,
                                fps: float = 20.0) -> List[dict]:
    """Copy REAL PaceAI pipeline output (evaluation/runs/after/<clip>.json) into
    the canonical per-video prediction cache used by the evaluator.

    This only re-formats output instead of re-running the slow pipeline. It is a
    prediction artifact, never ground truth.
    """
    if runs_dir is None:
        runs_dir = os.path.join(EVAL_DIR, "runs", "after")
    if not os.path.isdir(runs_dir):
        return []
    os.makedirs(predictions_dir, exist_ok=True)
    written = []
    for filename in sorted(os.listdir(runs_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(runs_dir, filename), "r", encoding="utf-8") as f:
            run = json.load(f)
        video_id = os.path.splitext(filename)[0]
        diag = run.get("diagnostics") or {}
        canonical = {
            "video_id": video_id,
            "fps": fps,
            "frame_space": "preprocessed_20fps_640x360",
            "release_frame": diag.get("release_frame_idx"),
            "front_contact_frame": diag.get("front_foot_contact_frame"),
            "reliable": diag.get("reliable"),
            "biomechanics": run.get("features") or {},
            "note": ("PaceAI pipeline prediction output -- NOT ground truth. "
                     "Frames are indices in the preprocessed space (20 FPS)."),
        }
        out_path = os.path.join(predictions_dir, f"{video_id}_pred.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(canonical, f, indent=2)
        written.append({"video_id": video_id, "path": out_path})
    return written


# --------------------------------------------------------------------------- #
# End-to-end validation run + reports
# --------------------------------------------------------------------------- #

def load_all_predictions(predictions_dir: str) -> Dict[str, Optional[dict]]:
    preds = {}
    if os.path.isdir(predictions_dir):
        for name in os.listdir(predictions_dir):
            if name.endswith("_pred.json"):
                video_id = name[: -len("_pred.json")]
                with open(os.path.join(predictions_dir, name), "r",
                          encoding="utf-8") as f:
                    preds[video_id] = json.load(f)
    return preds


def run_ground_truth_validation(
    gt_path: str = GT_PATH,
    predictions_dir: str = PREDICTIONS_DIR,
    results_dir: str = RESULTS_DIR,
    inventory_path: str = None,
    write_reports: bool = True,
) -> dict:
    """Run GT-vs-Prediction validation and (optionally) write reports.

    Honest behavior: if no independent annotations exist, every statistic
    reports "insufficient data"/"no annotations" -- never a fabricated number.
    Ground-truth rows are validated (frames in range, physics bounds,
    duplicates, annotator_id) but never silently corrected.
    """
    records, parse_errors = parse_ground_truth_csv(gt_path)
    predictions = load_all_predictions(predictions_dir)

    clip_metadata = load_clip_metadata(inventory_path)
    validation_issues = validate_annotations(records, clip_metadata)

    annotated_video_ids = sorted({r["video_id"] for r in records})
    predicted_video_ids = sorted(set(predictions.keys()))

    # per-video fps for frame-event conversions (preprocessed space).
    fps_by_clip = {}
    for vid in annotated_video_ids:
        pred = predictions.get(vid)
        fps_by_clip[vid] = float(pred.get("fps", 20.0) or 20.0) if pred else 20.0

    # Frame events: pair per-clip fps.
    frame_metrics = {}
    for gt_feature in FRAME_EVENTS:
        samples = paired_samples(records, predictions, gt_feature)
        fps = None
        if samples:
            fps = max(fps_by_clip.get(v[0], 20.0) for v in samples)
        frame_metrics[gt_feature] = compute_frame_event_metrics(
            gt_feature, samples, fps or 20.0)

    feature_metrics = {}
    feature_errors = []
    bland_altman_rows = []
    for gt_feature in CONTINUOUS_FEATURES:
        samples = paired_samples(records, predictions, gt_feature)
        feature_metrics[gt_feature] = compute_feature_agreement(gt_feature, samples)
        for video_id, gt_val, pred_val in samples:
            err = pred_val - gt_val
            feature_errors.append({
                "video_id": video_id,
                "feature": gt_feature,
                "ground_truth": gt_val,
                "prediction": pred_val,
                "error": err,
                "abs_error": abs(err),
            })
            bland_altman_rows.append({
                "video_id": video_id,
                "feature": gt_feature,
                "ground_truth": gt_val,
                "prediction": pred_val,
                "mean": (gt_val + pred_val) / 2.0,
                "difference": err,
            })
    for gt_feature in FRAME_EVENTS:
        samples = paired_samples(records, predictions, gt_feature)
        fps = fps_by_clip.get(samples[0][0], 20.0) if samples else 20.0
        for video_id, gt_val, pred_val in samples:
            abs_err = abs(gt_val - pred_val)
            feature_errors.append({
                "video_id": video_id,
                "feature": gt_feature,
                "ground_truth": gt_val,
                "prediction": pred_val,
                "error": gt_val - pred_val,
                "abs_error": abs_err,
                "error_ms": abs_err / fps * 1000.0,
            })
            bland_altman_rows.append({
                "video_id": video_id,
                "feature": gt_feature,
                "ground_truth": gt_val,
                "prediction": pred_val,
                "mean": (gt_val + pred_val) / 2.0,
                "difference": gt_val - pred_val,
            })

    summary_rows = []
    for gt_feature, m in feature_metrics.items():
        summary_rows.append({
            "feature": gt_feature,
            "n": m["n"],
            "status": m["status"],
            "bias": m["bias"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "sd_error": m["sd_error"],
            "pearson_r": m["pearson_r"],
            "r2": m["r2"],
            "loa_lower": m["loa_lower"],
            "loa_upper": m["loa_upper"],
            "icc21": m["icc21"],
        })
    for gt_feature, m in frame_metrics.items():
        summary_rows.append({
            "feature": gt_feature,
            "n": m["n"],
            "status": m["status"],
            "fps": m["fps"],
            "mean_abs_frame_error": m["mean_abs_frame_error"],
            "median_abs_frame_error": m["median_abs_frame_error"],
            "mean_signed_frame_error": m["mean_signed_frame_error"],
            "median_signed_frame_error": m["median_signed_frame_error"],
            "mean_abs_error_ms": m["mean_abs_error_ms"],
            "median_abs_error_ms": m["median_abs_error_ms"],
        })

    results = {
        "status": "PILOT_VALIDATION",
        "ground_truth_rows": len(records),
        "annotated_video_ids": annotated_video_ids,
        "prediction_video_ids": predicted_video_ids,
        "paired_samples_available": bool(feature_errors),
        "parse_errors": parse_errors,
        "validation_issues": validation_issues,
        "feature_metrics": feature_metrics,
        "frame_metrics": frame_metrics,
        "excluded_features": {k: {"reason": v} for k, v in EXCLUDED_FEATURES.items()},
    }

    if write_reports:
        write_reports_(
            results=results,
            feature_errors=feature_errors,
            bland_altman_rows=bland_altman_rows,
            summary_rows=summary_rows,
            gt_path=gt_path,
            predictions_dir=predictions_dir,
            results_dir=results_dir,
        )
    return results


def _fmt(x) -> str:
    if x is None:
        return "insufficient data"
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def write_reports_(results, feature_errors, bland_altman_rows, summary_rows,
                   gt_path, predictions_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # validation_summary.csv
    summary_path = os.path.join(results_dir, "validation_summary.csv")
    summary_fields = sorted({k for r in summary_rows for k in r})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: _fmt(r.get(k)) if not isinstance(r.get(k), int)
                             else r.get(k) for k in summary_fields})

    # feature_errors.csv
    errors_path = os.path.join(results_dir, "feature_errors.csv")
    with open(errors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "feature", "ground_truth",
                         "prediction", "error", "abs_error", "error_ms"])
        for r in feature_errors:
            writer.writerow([
                r["video_id"], r["feature"], _fmt(r["ground_truth"]),
                _fmt(r["prediction"]), _fmt(r["error"]),
                _fmt(r["abs_error"]), _fmt(r.get("error_ms")),
            ])

    # bland_altman_data.csv
    ba_path = os.path.join(results_dir, "bland_altman_data.csv")
    with open(ba_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "feature", "ground_truth",
                         "prediction", "mean", "difference"])
        for r in bland_altman_rows:
            writer.writerow([r["video_id"], r["feature"],
                             _fmt(r["ground_truth"]), _fmt(r["prediction"]),
                             _fmt(r["mean"]), _fmt(r["difference"])])

    # agreement_metrics.json
    metrics_path = os.path.join(results_dir, "agreement_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # validation_report.md
    report_path = os.path.join(results_dir, "validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_validation_report(
            results, summary_rows, gt_path, predictions_dir))

    return {
        "validation_summary.csv": summary_path,
        "feature_errors.csv": errors_path,
        "bland_altman_data.csv": ba_path,
        "agreement_metrics.json": metrics_path,
        "validation_report.md": report_path,
    }


def build_validation_report(results, summary_rows, gt_path, predictions_dir) -> str:
    n_ann = results["ground_truth_rows"]
    paired = results["paired_samples_available"]
    lines = []
    lines.append("# PaceAI P1 -- Pilot Validation Report")
    lines.append("")
    lines.append("> **Ground-truth validation is performed independently of the "
                 "PaceAI prediction pipeline.**")
    lines.append(">")
    lines.append("> **The current 8-clip dataset represents pilot validation and "
                 "is insufficient by itself to establish population-level "
                 "generalization.**")
    lines.append(">")
    lines.append("> **The validation evaluator is implemented, but no real "
                 "accuracy statistics are reported until independent annotations "
                 "are available.**")
    lines.append("")
    if n_ann == 0:
        lines.append("## Status")
        lines.append("")
        lines.append("**NO REAL GROUND TRUTH -- INSUFFICIENT DATA.**")
        lines.append("")
        lines.append("No independent ground-truth annotations have been entered "
                     "yet (`evaluation/ground_truth/ground_truth.csv` is "
                     "schema-only), so no accuracy statistic of any kind is "
                     "reported.")
        lines.append("")
        lines.append("The 67.5 -> 80.0 value previously reported is a "
                     "**plausibility-vs-published-ranges** score and a bug-fix "
                     "verification -- it is NOT measurement accuracy and is NOT "
                     "validation.")
        lines.append("")
        lines.append("All feature statistics below report **insufficient data**.")
    elif not paired:
        lines.append("## Status")
        lines.append("")
        lines.append(f"{n_ann} ground-truth row(s) exist but **no paired "
                     "samples** (matching independent + prediction values) are "
                     "available yet -- every statistic reports insufficient data.")
        lines.append("")
    else:
        lines.append("## Status")
        lines.append("")
        lines.append(f"Ground-truth rows: {n_ann} | Paired samples present.")
        lines.append("")

    lines.append("## Cross-checks")
    lines.append("")
    lines.append(f"- Ground truth source: `{gt_path}`")
    lines.append(f"- Prediction cache: `{predictions_dir}`")
    lines.append(f"- Annotated clips: {results['annotated_video_ids'] or 'none'}")
    lines.append(f"- Clips with predictions: {results['prediction_video_ids'] or 'none'}")
    lines.append("")
    lines.append("### Excluded features (no valid comparison declared)")
    lines.append("")
    lines.append("| Feature | Reason |")
    lines.append("|---|---|")
    for feature, info in (results.get("excluded_features") or {}).items():
        reason = info.get("reason") if isinstance(info, dict) else info
        lines.append(f"| {feature} | {reason} |")
    lines.append("")

    issues = results.get("validation_issues") or []
    lines.append("## Ground-truth data validation")
    lines.append("")
    if issues:
        lines.append(f"{len(issues)} issue(s) reported (never silently "
                     "corrected):")
        lines.append("")
        lines.append("| Row | Field | Code | Message |")
        lines.append("|---|---|---|---|")
        for it in issues:
            lines.append(f"| {it['row']} | {it['field']} | {it['code']} | "
                         f"{it['message']} |")
    else:
        lines.append("No validation issues in the current ground-truth rows.")
    lines.append("")

    lines.append("## Per-feature agreement summary")
    lines.append("")
    lines.append("`insufficient data` = fewer valid paired samples than the "
                 "statistic requires.")
    lines.append("")
    lines.append("| Feature | n | status | bias | MAE | RMSE | sd(err) | r | "
                 "R^2 | LoA low | LoA high | ICC(2,1) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        lines.append("| {feature} | {n} | {status} | {bias} | {mae} | {rmse} | "
                     "{sd} | {r} | {r2} | {loa_l} | {loa_u} | {icc} |".format(
            feature=r["feature"], n=r.get("n", "insufficient data"),
            status=r.get("status", ""),
            bias=_fmt(r.get("bias")), mae=_fmt(r.get("mae")),
            rmse=_fmt(r.get("rmse")), sd=_fmt(r.get("sd_error")),
            r=_fmt(r.get("pearson_r")), r2=_fmt(r.get("r2")),
            loa_l=_fmt(r.get("loa_lower")), loa_u=_fmt(r.get("loa_upper")),
            icc=_fmt(r.get("icc21"))))
    lines.append("")
    lines.append("### Frame-event errors (frames; milliseconds at preprocessed "
                 "20 FPS for GT-vs-prediction)")
    lines.append("")
    lines.append("| Feature | n | mean abs (frames) | median abs (frames) | "
                 "mean abs (ms) | median abs (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary_rows:
        if "mean_abs_frame_error" not in r or r.get("mean_abs_frame_error") is None:
            continue
        lines.append("| {f} | {n} | {ma} | {med} | {ms} | {msm} |".format(
            f=r["feature"], n=r.get("n", "insufficient data"),
            ma=_fmt(r.get("mean_abs_frame_error")),
            med=_fmt(r.get("median_abs_frame_error")),
            ms=_fmt(r.get("mean_abs_error_ms")),
            msm=_fmt(r.get("median_abs_error_ms"))))
    lines.append("")
    lines.append("## Statistical integrity")
    lines.append("")
    lines.append("- Min samples for bias/MAE/RMSE: "
                 f"{MIN_SAMPLES_DESCRIPTIVE}; sd of error: {MIN_SAMPLES_SD}; "
                 f"Bland-Altman LoA: {MIN_SAMPLES_LOA}; correlation/R^2: "
                 f"{MIN_SAMPLES_CORRELATION}; ICC(2,1): {MIN_SAMPLES_ICC}.")
    lines.append("- ICC(2,1): two-way random-effects, single measures "
                 "(ground-truth and prediction treated as two raters).")
    lines.append("- R^2 reported here is the squared Pearson correlation.")
    lines.append("- Results are **PILOT VALIDATION**, not a population-level "
                 "generalization claim.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="PaceAI P1 Ground-Truth Validation Evaluator")
    parser.add_argument("--write", action="store_true",
                        help="write evaluation/results/* reports")
    parser.add_argument("--predictions_dir", default=PREDICTIONS_DIR)
    parser.add_argument("--build_predictions", action="store_true",
                        help="copy evaluation/runs/after/*.json into the "
                             "prediction cache (PaceAI output, NOT ground truth)")
    args = parser.parse_args()

    if args.build_predictions:
        written = build_predictions_from_runs(predictions_dir=args.predictions_dir)
        print(f"Built {len(written)} prediction cache file(s).")

    results = run_ground_truth_validation(
        predictions_dir=args.predictions_dir, write_reports=args.write)

    print("\n=== PACEAI P1 GROUND-TRUTH VALIDATION ===")
    print(f"Ground-truth rows : {results['ground_truth_rows']}")
    print(f"Annotated clips   : {results['annotated_video_ids'] or 'none'}")
    print(f"Prediction clips  : {results['prediction_video_ids'] or 'none'}")
    if results["ground_truth_rows"] == 0:
        print("Paired samples    : NO REAL GROUND TRUTH -- INSUFFICIENT DATA")
    else:
        paired = results["paired_samples_available"]
        print("Paired samples    : %s" % ("yes" if paired else "no -> insufficient data"))
    for feature in list(results["feature_metrics"].keys()) + list(results["frame_metrics"].keys()):
        m = (results["feature_metrics"] or results["frame_metrics"]).get(feature) or \
            results["frame_metrics"].get(feature) or results["feature_metrics"].get(feature)
        if feature in results["feature_metrics"]:
            m = results["feature_metrics"][feature]
            print(f"  {feature:<22} n={m['n']:<3} status={m['status']}")
        else:
            m = results["frame_metrics"][feature]
            print(f"  {feature:<22} n={m['n']:<3} status={m['status']}")
    if results["parse_errors"]:
        print("\nParse errors:")
        for e in results["parse_errors"]:
            print("  -", e)
    if args.write:
        print("\nReports written to evaluation/results/")


if __name__ == "__main__":
    main()