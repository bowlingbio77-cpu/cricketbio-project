"""
Real-clip baseline harness for PaceAI biomechanics.

Runs the CURRENT pose -> feature-engineering pipeline on real bowling clips and
records every feature + delivery diagnostics + landmark-quality statistics so
that BEFORE/AFTER comparisons can be made after scientifically-justified fixes.

Usage:
    python scripts/real_clip_baseline.py [--clips a.avi b.mp4 ...] [--phase before]

Outputs one JSON per clip under evaluation/runs/<phase>/ plus a summary CSV.
This harness intentionally does NOT run YOLO/ball tracking (heavy, optional);
it measures the scientific core: pose -> release frame -> 10 biomechanical
features -> reliability-relevant landmark statistics.

It is a measurement tool, NOT the scoring rubric. It never fabricates
ground truth.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, preprocessing, pose_estimation, feature_engineering as feateng

# Key landmarks required for the 10-feature vector.
KEY_LANDMARKS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_index", "right_index",
]
DELIVERY_KEY_LANDMARKS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]


def bowling_arm_from_name(path: str) -> str:
    base = os.path.basename(path).lower()
    if "_left" in base or base.startswith("left_"):
        return "left"
    if "_right" in base or base.startswith("right_"):
        return "right"
    return "right"


def landmark_stats(pose_sequence):
    """Mean visibility + fraction of frames with all key landmarks above a
    visibility threshold (0.4) -- the per-frame completeness of the limb
    chain the 10 features depend on."""
    if not pose_sequence:
        return {"n_pose_frames": 0, "mean_visibility": 0.0,
                "complete_fraction": 0.0, "key_visibility": {}}
    vis = {name: [] for name in KEY_LANDMARKS}
    complete = 0
    for pf in pose_sequence:
        lm = pf.landmarks
        ok = True
        for name in KEY_LANDMARKS:
            v = float(lm[config.POSE_LANDMARK_NAMES.index(name), 3])
            vis[name].append(v)
            if v < 0.4:
                ok = False
        if ok:
            complete += 1
    out = {"n_pose_frames": len(pose_sequence),
           "complete_fraction": complete / max(1, len(pose_sequence))}
    out["key_visibility"] = {name: round(float(np.mean(vs)), 3)
                             for name, vs in vis.items()}
    return out


def analyze_clip(path: str, bowling_arm: str) -> dict:
    frames = list(preprocessing.preprocess_video(
        path, target_fps=config.TARGET_FPS, resize_dim=config.RESIZE_DIM))
    result = {"video": path, "bowling_arm": bowling_arm,
              "n_frames_preprocessed": len(frames), "features": None,
              "diagnostics": None, "landmark_stats": {}, "error": None,
              "elapsed_s": None}
    t0 = time.perf_counter()
    try:
        with pose_estimation.PoseEstimator() as estimator:
            pose_sequence = estimator.process_video_frames(iter(frames))
        if len(pose_sequence) < 3:
            result["error"] = f"Only {len(pose_sequence)} pose frames"
            result["elapsed_s"] = round(time.perf_counter() - t0, 1)
            return result
        fv, diag = feateng.analyze_delivery(pose_sequence,
                                            bowling_arm=bowling_arm,
                                            camera_view="behind")
        result["features"] = {k: (None if v is None else round(float(v), 4))
                              for k, v in fv.items()}
        d = dict(diag)
        d.pop("frame_features", None)
        result["diagnostics"] = d
        result["landmark_stats"] = landmark_stats(pose_sequence)
    except Exception as exc:  # record, never hide
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return result


DEFAULT_DIR = os.path.join("corrected_all_data", "bowling")


def pick_default_clips(n=8):
    candidates = []
    for pat in ["fast_right_00000001.avi", "fast_right_00000002.avi",
                "fast_left_0000001.avi", "fast_left_00000001.avi",
                "off_left_00000001.mp4", "off_right_00000042.avi",
                "leg_right_00000029.avi", "leg_right_00000001.mp4"]:
        p = os.path.join(DEFAULT_DIR, pat)
        if os.path.exists(p):
            candidates.append(p)
    if len(candidates) < n:
        # fall back to whatever is available
        import glob
        allv = sorted(glob.glob(os.path.join(DEFAULT_DIR, "*")))
        for p in allv:
            if p not in candidates:
                candidates.append(p)
            if len(candidates) >= n:
                break
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--phase", default="before")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    clips = args.clips or pick_default_clips(args.n)
    clips = [c for c in clips if os.path.exists(c)]
    if not clips:
        print("No clips found.")
        return

    out_dir = os.path.join("evaluation", "runs", args.phase)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for c in clips:
        res = analyze_clip(c, bowling_arm_from_name(c))
        name = os.path.splitext(os.path.basename(c))[0]
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        fv = res["features"] or {}
        d = res["diagnostics"] or {}
        rows.append({
            "video": name, "bowling_arm": res["bowling_arm"],
            "n_pose_frames": res["landmark_stats"].get("n_pose_frames", 0),
            "complete_frac": res["landmark_stats"].get("complete_fraction", 0.0),
            "reliable": d.get("reliable"),
            "release_frame": d.get("release_frame_idx"),
            "release_at_edge": d.get("release_frame_idx") is not None and (
                d.get("n_frames") and d.get("release_frame_idx") >= (d.get("n_frames") or 0) - 2
            ),
            "front_leg": d.get("front_leg"),
            **fv,
            "elapsed_s": res["elapsed_s"],
            "error": res["error"],
        })
        print(f"[{args.phase}] {name}: {res['elapsed_s']}s "
              f"reliable={d.get('reliable')} release={d.get('release_frame_idx')} "
              f"err={res['error']}")

    summary = os.path.join("evaluation", "runs", f"{args.phase}_summary.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary -> {summary} ({len(rows)} clips)")


if __name__ == "__main__":
    main()