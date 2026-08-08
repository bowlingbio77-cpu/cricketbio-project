"""
Extract the 10 biomechanical feature vectors from every bowling video.

Pipeline per video:
    preprocessing (resize, optional denoise)
      -> MediaPipe Pose (largest person selected per frame = bowler proxy)
      -> feature_engineering.build_feature_vector

Usage:
    python scripts/extract_features.py --source <dir> --out <csv>
    python scripts/extract_features.py --limit 10   # dry-run on N videos
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, feature_engineering as feateng
from src import preprocessing

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=config.POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

FEATURE_NAMES = config.FEATURE_NAMES


def parse_meta(filename: str):
    """fast_left_00000012.avi -> (bowler_type='fast', arm='left')"""
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 2)
    if len(parts) == 3 and parts[1] in ("left", "right"):
        return parts[0], parts[1]
    return "", ""


def largest_pose_landmarks(result):
    """Pick the pose with the biggest 2D bounding box (proxy for the bowler)."""
    best, best_area = None, -1.0
    for lm in result.pose_landmarks:
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best, best_area = lm, area
    return best


def extract_one_video(path, landmarker, denoise, target_fps):
    pose_sequence = []
    for idx, ts, frame in preprocessing.preprocess_video(
        path, target_fps=target_fps, denoise=denoise
    ):
        rgb = (frame[:, :, ::-1]).copy()  # BGR -> RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if not res.pose_landmarks:
            continue
        lm = largest_pose_landmarks(res)
        landmarks = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])
        pose_sequence.append((idx, ts, landmarks))

    if len(pose_sequence) < 3:
        return None, len(pose_sequence)

    seq = [type("PF", (), {"frame_idx": i, "timestamp_sec": t, "landmarks": L})
           for i, t, L in pose_sequence]
    try:
        vec = feateng.build_feature_vector(seq, bowling_arm="right")
    except Exception:
        return None, len(pose_sequence)
    return vec, len(pose_sequence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"D:\corrected_all_data\bowling")
    ap.add_argument("--out", default=os.path.join(config.DATA_DIR, "video_features.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--denoise", action="store_true", help="enable heavy denoise (slow)")
    ap.add_argument("--target_fps", type=float, default=config.TARGET_FPS)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.source) if f.lower().endswith((".avi", ".mp4")))
    if args.limit:
        files = files[: args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["filename"])

    header = ["filename", "bowler_type", "arm"] + FEATURE_NAMES + [
        "status", "n_pose_frames", "runtime_s"
    ]
    write_header = not (os.path.exists(args.out) and args.resume)

    times = []
    with PoseEstimator() as estimator, open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for i, fn in enumerate(files, 1):
            if fn in done:
                continue
            t0 = time.time()
            vec, npose = extract_one_video(
                os.path.join(args.source, fn), estimator, args.denoise, args.target_fps
            )
            dt = time.time() - t0
            times.append(dt)
            row = {"filename": fn, "bowler_type": "", "arm": "", "status": "ok",
                   "n_pose_frames": npose, "runtime_s": round(dt, 2)}
            btype, arm = parse_meta(fn)
            row["bowler_type"], row["arm"] = btype, arm
            if vec is not None:
                row.update({k: round(float(vec[k]), 4) for k in FEATURE_NAMES})
            else:
                row["status"] = "failed"
            writer.writerow(row)
            if i % 20 == 0 or i == len(files):
                avg = np.mean(times) if times else 0
                print(f"[{i}/{len(files)}] avg {avg:.2f}s/video, last={fn} {row['status']}",
                      flush=True)

    print(f"done. wrote {len(files)} to {args.out}")
    if times:
        print(f"mean {np.mean(times):.2f}s/video, total {np.sum(times)/60:.1f} min (this batch)")


if __name__ == "__main__":
    main()
