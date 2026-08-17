"""
Stage 1: Auto-label cricket ball bounding boxes from real bowling clips.

Runs the existing ball tracker on every clip in corrected_all_data/bowling/.
For clips where the tracker succeeds (outcome == "ok" or "release_no_impact"),
extracts the ball bounding boxes from the winning track and saves them as
YOLO-format labels.

Output structure:
    data/cricket_ball_dataset/
        auto_labeled/
            images/<clip_name>_<frame_idx>.png
            labels/<clip_name>_<frame_idx>.txt

Usage:
    python scripts/auto_label.py [--limit N] [--min-conf 0.3]
"""
import argparse
import os
import sys
import json
import random

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import preprocessing
from src.ball_tracking_v2 import BallTracker, BallPoint

BOWLING_DIR = os.path.join("corrected_all_data", "bowling")
OUTPUT_DIR = os.path.join("data", "cricket_ball_dataset", "auto_labeled")
BALL_CLASS_ID = 0  # single class: cricket ball


def load_clip_frames(video_path, resize_dim=(640, 360)):
    frames = list(preprocessing.preprocess_video(video_path, target_fps=20,
                                                  resize_dim=resize_dim, denoise=False))
    if not frames:
        return []
    return frames


def run_tracker_on_clip(video_path):
    frames = load_clip_frames(video_path)
    if not frames:
        return None, None, []
    tracker = BallTracker()
    track, stats = tracker.track(frames)
    return track, stats, frames


def extract_labels_for_clip(track, frames, clip_name, output_dir):
    """Save YOLO-format labels + extracted frames for a single clip."""
    if not track:
        return 0

    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Build frame lookup by frame_idx
    frame_map = {idx: (ts, fr) for idx, ts, fr in frames}
    count = 0

    for pt in track:
        if pt.frame_idx not in frame_map:
            continue
        ts, frame = frame_map[pt.frame_idx]
        h, w = frame.shape[:2]  # should be 360x640

        # Save frame as image
        img_name = f"{clip_name}_f{pt.frame_idx:04d}.png"
        lbl_name = f"{clip_name}_f{pt.frame_idx:04d}.txt"

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        cv2.imwrite(img_path, frame)

        # YOLO format: class x_center y_center width height (all normalized 0-1)
        cx = pt.x / w
        cy = pt.y / h
        bw = pt.w / w
        bh = pt.h / h
        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.01, min(0.5, bw))
        bh = max(0.01, min(0.5, bh))

        with open(lbl_path, "w") as f:
            f.write(f"{BALL_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max clips to process (0=all)")
    parser.add_argument("--min-confidence", type=float, default=0.2,
                        help="Min track confidence to keep labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover all bowling clips
    clips = []
    for fname in sorted(os.listdir(BOWLING_DIR)):
        base, ext = os.path.splitext(fname)
        if ext.lower() in (".avi", ".mp4"):
            clips.append((base, os.path.join(BOWLING_DIR, fname)))

    random.seed(args.seed)
    random.shuffle(clips)
    if args.limit > 0:
        clips = clips[:args.limit]

    print(f"Processing {len(clips)} clips...")

    total_labels = 0
    success_count = 0
    fail_count = 0
    stats_summary = {"ok": 0, "release_no_impact": 0, "track_too_short": 0, "other": 0}

    for i, (clip_name, video_path) in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] {clip_name} ... ", end="", flush=True)
        try:
            track, stats, frames = run_tracker_on_clip(video_path)
            outcome = stats.get("outcome", "unknown") if stats else "unknown"

            if outcome in ("ok", "release_no_impact") and track:
                n = extract_labels_for_clip(track, frames, clip_name, OUTPUT_DIR)
                total_labels += n
                success_count += 1
                stats_summary[stats_summary.get(outcome, "other") is not None and outcome or "other"] += 1
                if outcome == "ok":
                    stats_summary["ok"] += 1
                elif outcome == "release_no_impact":
                    stats_summary["release_no_impact"] += 1
                print(f"{outcome} -> {n} labels")
            else:
                fail_count += 1
                stats_summary["track_too_short" if outcome == "track_too_short" else "other"] += 1
                print(f"{outcome} (skipped)")
        except Exception as e:
            fail_count += 1
            stats_summary["other"] += 1
            print(f"ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"AUTO-LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Clips processed: {len(clips)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed/skipped:  {fail_count}")
    print(f"  Total labels:    {total_labels}")
    print(f"  Output:          {OUTPUT_DIR}")
    print(f"\n  Outcome breakdown:")
    for k, v in sorted(stats_summary.items()):
        if v > 0:
            print(f"    {k:25s}: {v}")

    # Save metadata
    meta = {
        "total_clips": len(clips),
        "successful": success_count,
        "failed": fail_count,
        "total_labels": total_labels,
        "outcome_breakdown": stats_summary,
        "class_names": ["cricket_ball"],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {OUTPUT_DIR}/metadata.json")


if __name__ == "__main__":
    main()
