"""
Validation comparison harness: BEFORE vs AFTER real-clip run.

Scores every clip's 10 biomechanical features against the published fast-bowling
ranges encoded in `app.py:FEATURE_LABELS` (literature-derived). The SAME rubric
is applied to both phases, so a change in score reflects a change in how many
feature values fall inside the published plausible band for an elite/intl
fast-bowling delivery -- it is a plausibility measurement, NOT ground-truth
validation (no annotated reference dataset exists yet, see evaluation/README.md).

Usage:
    python scripts/compare_validation.py [--before evaluation/runs/before_summary.csv] [--after evaluation/runs/after_summary.csv]

Output:
    - prints per-clip before/after tables + overall scores
    - writes evaluation/runs/validation_scores.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

# Published feature bands (0.0 = lower bound, 1.0 = upper bound) from app.py.
FEATURE_BANDS = {
    "shoulder_rotation_deg": (0.0, 90.0),
    "elbow_flexion_deg": (0.0, 45.0),
    "wrist_angle_deg": (90.0, 180.0),
    "hip_rotation_deg": (0.0, 80.0),
    "knee_flexion_deg": (0.0, 60.0),
    "trunk_lean_deg": (0.0, 60.0),
    "stride_length_norm": (0.3, 1.6),
    "release_angle_deg": (30.0, 90.0),
    "angular_velocity_deg_s": (100.0, 1500.0),
    "ground_contact_time_s": (0.05, 0.35),
}
# Centre of the published band for reporting; NOT used for pass/fail scoring.
ELITE_TARGETS = {
    "shoulder_rotation_deg": 18.0, "elbow_flexion_deg": 8.0,
    "wrist_angle_deg": 165.0, "hip_rotation_deg": 45.0,
    "knee_flexion_deg": 10.0, "trunk_lean_deg": 25.0,
    "stride_length_norm": 1.05, "release_angle_deg": 78.0,
    "angular_velocity_deg_s": 1100.0, "ground_contact_time_s": 0.11,
}


def read_summary(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_clip(row):
    """Per-clip: fraction of features within the published band (None = missing)."""
    scored, passed = 0, 0
    per_feature = {}
    for feat, (lo, hi) in FEATURE_BANDS.items():
        raw = row.get(feat)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            per_feature[feat] = None
            continue
        scored += 1
        ok = lo <= val <= hi
        passed += int(ok)
        per_feature[feat] = {"value": round(val, 2), "in_range": ok,
                             "target": ELITE_TARGETS[feat]}
    score = (passed / scored * 100.0) if scored else 0.0
    return score, passed, scored, per_feature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="evaluation/runs/before_summary.csv")
    ap.add_argument("--after", default="evaluation/runs/after_summary.csv")
    args = ap.parse_args()

    before = read_summary(args.before)
    after = read_summary(args.after)
    if not before or not after:
        print("Missing before/after summary CSV; run scripts/real_clip_baseline.py "
              "for both --phase before and --phase after first.")
        sys.exit(1)

    per_clip_by_phase = {}
    for phase, data in (("before", before), ("after", after)):
        per_clip = {}
        for row in data:
            score, passed, scored, per_feature = score_clip(row)
            per_clip[row["video"]] = {
                "score": round(score, 1), "passed": passed, "scored": scored,
                "features": per_feature,
                "reliable": row.get("reliable"), "release_frame": row.get("release_frame"),
                "complete_frac": round(float(row.get("complete_frac", 0) or 0), 3),
            }
        per_clip_by_phase[phase] = per_clip

    before_map = per_clip_by_phase["before"]
    after_map = per_clip_by_phase["after"]
    before_overall = round(float(np.mean([v["score"] for v in before_map.values()])), 1) if before_map else 0.0
    after_overall = round(float(np.mean([v["score"] for v in after_map.values()])), 1) if after_map else 0.0

    def feature_summary(per_clip):
        agg = {}
        for feat in FEATURE_BANDS:
            vals = [c["features"].get(feat) for c in per_clip.values()]
            valid = [v for v in vals if v is not None]
            agg[feat] = {
                "in_range_frac": (sum(1 for v in valid if v["in_range"]) / len(valid) if valid else 0.0),
                "n_valid": len(valid),
                "mean_value": (float(np.mean([v["value"] for v in valid])) if valid else None),
            }
        return agg

    fb, fa = feature_summary(before_map), feature_summary(after_map)

    print("=" * 78)
    print("Per-clip biomechanics-plausibility score (0-100, published ranges)")
    print(f"  overall BEFORE: {before_overall}/100   AFTER: {after_overall}/100   "
          f"delta: {after_overall - before_overall:+.1f}")
    print("=" * 78)
    print(f"{'clip':<24}{'before':>8}{'after':>8}{'delta':>8}   reliable->")
    for vid in sorted(set(before_map) | set(after_map)):
        b = before_map.get(vid, {}).get("score")
        a = after_map.get(vid, {}).get("score")
        br = before_map.get(vid, {}).get("reliable")
        ar = after_map.get(vid, {}).get("reliable")
        d = (a - b) if (b is not None and a is not None) else None
        d_str = f"{d:+.1f}" if d is not None else "  -  "
        print(f"{vid:<24}{('--' if b is None else f'{b:6.1f}'):>8}"
              f"{('--' if a is None else f'{a:6.1f}'):>8}{d_str:>9}   {br}->{ar}")

    print()
    print("Feature-level in-range fraction of clips (before -> after):")
    for feat in FEATURE_BANDS:
        b = fb[feat]
        a = fa[feat]
        print(f"  {feat:<22} {b['in_range_frac']:.0%} -> {a['in_range_frac']:.0%}   "
              f"(mean val {b['mean_value']:.2f} -> {a['mean_value']:.2f}, n={a['n_valid']})")

    out = {
        "rubric": "published fast-bowling ranges (app.py:FEATURE_LABELS)",
        "n_clips": len(before_map),
        "before": {"overall": before_overall, "clips": before_map, "features": fb},
        "after": {"overall": after_overall, "clips": after_map, "features": fa},
    }
    out_path = os.path.join("evaluation", "runs", "validation_scores.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()