"""
Stage 1: Auto-label bowler/batsman roles from a small set of bowling clips.

Strategy (reuses the repo's cricket-aware tracking heuristics):
  * Every clip is processed through YOLO person detection + ByteTrack.
  * The BOWLER is the track with the highest cricket-evidence score
    (run-up motion, straight line, growth, vertical bias -- see src/tracking.py).
  * The BATSMAN is the "anti-bowler": the long-lived track that is NOT the
    bowler, stays near the crease, and barely moves (high span, low motion,
    moderate-to-large bbox). This is exactly the person the old "biggest
    person" heuristic used to wrongly crown as the bowler.

Frames are sampled every `step` frames (default 2) to keep the training set
small and de-correlated. Images are resized to 640x360 (the pipeline's standard
coordinate space) and labels are written in YOLO format:
    class x_center y_center width height    (normalized 0..1)
    class 0 = bowler, class 1 = batsman

Output layout (consumed by scripts/train_bowler_batsman.py):
    data/bowler_batsman_dataset/
        auto_labeled/
            images/<clip>_<frame>.png
            labels/<clip>_<frame>.txt
        preview/                      # annotated review frames
        metadata.json

Usage:
    python scripts/prepare_bowler_batsman_dataset.py
    python scripts/prepare_bowler_batsman_dataset.py --clips fast_left_0000036.avi \
        fast_right_0000084.avi --step 2
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, preprocessing, tracking as trackmod

BOWLING_DIR = os.path.join("corrected_all_data", "bowling")
OUTPUT_DIR = os.path.join("data", "bowler_batsman_dataset")
AUTO_DIR = os.path.join(OUTPUT_DIR, "auto_labeled")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview")
N_CLIPS_DEFAULT = 25
CLASS_NAMES = ["bowler", "batsman"]
BOWLER_CLASS_ID = 0
BATSMAN_CLASS_ID = 1
SEED = 42

# Batsman anti-bowler scoring constants.
BATSMAN_MIN_SPAN = 0.35       # must be present in >= 35% of frames
BATSMAN_MAX_MOTION = 0.45     # must not move around like the run-up
BATSMAN_MIN_AREA_FRACTION = 0.008  # far-end batsmen are small in delivery-cam shots


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def sample_clips(n: int) -> list:
    """Deterministically pick n diverse clips (mix of fast/leg/off, left/right)."""
    random.seed(SEED)
    by_type = {}
    for fname in sorted(os.listdir(BOWLING_DIR)):
        if not fname.lower().endswith((".avi", ".mp4")):
            continue
        t, _, arm = fname.rpartition("_")[0].replace("_00000", "_"), "", ""
        parts = fname.split("_")
        key = (parts[0], parts[1] if len(parts) > 1 and parts[1] in ("left", "right") else "")
        by_type.setdefault(key, []).append(fname)

    picked = []
    for key in sorted(by_type, key=lambda k: -len(by_type[k])):
        pool = [f for f in by_type[key] if f not in picked]
        if not pool:
            continue
        picked.append(random.choice(pool))
        if len(picked) >= n:
            break
    if len(picked) < n:
        pool = [f for f in sorted(os.listdir(BOWLING_DIR))
                if f.lower().endswith((".avi", ".mp4")) and f not in picked]
        picked.extend(random.sample(pool, min(n - len(picked), len(pool))))
    return picked[:n]


def compute_role_tracks(tracks: dict, frame_dims, n_total) -> tuple:
    """Return (bowler_track_id, batsman_track_id) using the repo's heuristics.

    Bowler via select_bowler_track_with_meta; batsman = best anti-bowler track.
    """
    bowler, _meta = trackmod.select_bowler_track_with_meta(
        tracks, frame_dims=frame_dims, total_frames=n_total)
    if bowler is None:
        return None, None

    frame_h, frame_w = frame_dims
    frame_area = frame_h * frame_w
    best_id, best_score = None, -1.0
    for tid, tr in tracks.items():
        if tid == bowler.track_id:
            continue
        span = len(tr) / max(1, n_total)
        if span < BATSMAN_MIN_SPAN:
            continue
        steps = trackmod._consecutive_steps(tr)
        motion = float(np.sum(steps)) / float(np.hypot(frame_w, frame_h) + 1e-6)
        motion = float(np.tanh(motion))
        if motion > BATSMAN_MAX_MOTION:
            continue
        areas = np.asarray([_bbox_area(b) for b in tr.bboxes], dtype=float)
        med_area = float(np.median(areas)) if len(areas) else 0.0
        if med_area / max(1.0, frame_area) < BATSMAN_MIN_AREA_FRACTION:
            continue
        size = float(np.tanh(med_area / max(0.01 * frame_area, 1e-6)))
        score = span * size * (1.0 - motion)
        if score > best_score:
            best_score, best_id = score, tid
    return bowler.track_id if bowler else None, best_id


def annotate_frame(frame, bboxes, labels):
    """Draw the two role boxes on the frame (red=bowler, blue=batsman)."""
    img = frame.copy()
    colors = {BOWLER_CLASS_ID: (0, 0, 255), BATSMAN_CLASS_ID: (255, 0, 0)}
    for lbl, (x1, y1, x2, y2) in zip(labels, bboxes):
        c = colors[lbl]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(img, CLASS_NAMES[lbl], (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return img


def process_clip(video_path: str, step: int, stats: dict) -> int:
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = list(preprocessing.preprocess_video(
        video_path, target_fps=config.TARGET_FPS,
        resize_dim=config.RESIZE_DIM, denoise=False))
    if not frames:
        return 0

    # IMPORTANT: always label with the generic COCO person model. The fine-tuned
    # role model is the *target* we are generating labels for; using it here
    # would leak its biases (and its current low-confidence boxes) into the labels.
    tracker = trackmod.BowlerTracker(weights=config.YOLO_WEIGHTS)
    tracks = tracker.track_frames([(idx, fr) for idx, ts, fr in frames])
    if not tracks:
        print("      no person tracks -> skipped")
        return 0

    h, w = frames[0][2].shape[:2]
    bowler_id, batsman_id = compute_role_tracks(
        tracks, frame_dims=(h, w), n_total=len(frames))
    if bowler_id is None:
        print("      no bowler found -> skipped")
        return 0

    frame_map = {idx: (ts, fr) for idx, ts, fr in frames}
    img_dir = os.path.join(AUTO_DIR, "images")
    lbl_dir = os.path.join(AUTO_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    bowler_tr = tracks[bowler_id]
    batsman_tr = tracks[batsman_id] if batsman_id is not None else None
    bowler_by_f = dict(zip(bowler_tr.frames, bowler_tr.bboxes))
    batsman_by_f = dict(zip(batsman_tr.frames, batsman_tr.bboxes)) if batsman_tr else {}

    count = 0
    preview_paths = []
    for idx in sorted(frame_map):
        if idx % step != 0 and idx != sorted(frame_map)[0]:
            continue
        bboxes, labels = [], []
        if idx in bowler_by_f:
            bboxes.append(bowler_by_f[idx])
            labels.append(BOWLER_CLASS_ID)
        if idx in batsman_by_f:
            bboxes.append(batsman_by_f[idx])
            labels.append(BATSMAN_CLASS_ID)
        if not bboxes:
            continue
        ts, frame = frame_map[idx]
        img_name = f"{clip_name}_f{idx:04d}.png"
        lbl_name = f"{clip_name}_f{idx:04d}.txt"
        cv2.imwrite(os.path.join(img_dir, img_name), frame)
        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            for lbl, (x1, y1, x2, y2) in zip(labels, bboxes):
                nx1, ny1 = max(0.0, x1 / w), max(0.0, y1 / h)
                nx2, ny2 = min(1.0, x2 / w), min(1.0, y2 / h)
                nw, nh = nx2 - nx1, ny2 - ny1
                if nw <= 0 or nh <= 0:
                    continue
                f.write(f"{lbl} {(nx1 + nx2) / 2:.6f} {(ny1 + ny2) / 2:.6f} "
                        f"{nw:.6f} {nh:.6f}\n")
        count += 1
        if count <= 8:
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            p = os.path.join(PREVIEW_DIR, f"{clip_name}_preview_f{idx:04d}.jpg")
            cv2.imwrite(p, annotate_frame(frame, bboxes, labels))
            preview_paths.append(p)

    stats["frames"] += len(frames)
    stats["labeled"] += count
    stats["bowler_frames"] += len(bowler_by_f)
    stats["batsman_frames"] += len(batsman_by_f)
    print(f"      bowler=#{bowler_id} batsman=#{batsman_id} "
          f"-> {count} labeled frames")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="exact clip filenames (default: sampled)")
    ap.add_argument("--step", type=int, default=2, help="frame sampling step (default 2)")
    ap.add_argument("--limit", type=int, default=N_CLIPS_DEFAULT,
                    help="number of clips when not specified explicitly")
    args = ap.parse_args()

    if args.clips:
        clips = []
        for c in args.clips:
            p = c if os.path.exists(c) else os.path.join(BOWLING_DIR, c)
            if os.path.exists(p):
                clips.append(os.path.basename(p))
        if not clips:
            sys.exit("none of the --clips resolved; expected files in corrected_all_data/bowling/")
    else:
        clips = sample_clips(args.limit)

    print("=" * 60)
    print("AUTO-LABEL BOWLER/BATSMAN ROLES")
    print("=" * 60)
    print(f"Clips ({len(clips)}):")
    for c in clips:
        print(f"  - {c}")

    stats = {"clips": [], "frames": 0, "labeled": 0,
             "bowler_frames": 0, "batsman_frames": 0}
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip} ...")
        try:
            n = process_clip(os.path.join(BOWLING_DIR, clip), args.step, stats)
            stats["clips"].append({"clip": clip, "labeled_frames": n})
        except Exception as exc:  # noqa: BLE001
            stats["clips"].append({"clip": clip, "labeled_frames": 0, "error": str(exc)})
            print(f"      ERROR: {exc}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta = {
        "class_names": CLASS_NAMES,
        "n_clips": len(clips),
        **stats,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE")
    print("=" * 60)
    n_imgs = len(os.listdir(os.path.join(AUTO_DIR, "images")))
    print(f"  Clips processed: {len(clips)}")
    print(f"  Total frames read: {stats['frames']}")
    print(f"  Labeled frames saved: {n_imgs} (labels in {os.path.join(AUTO_DIR, 'labels')})")
    print(f"  Bowler boxes: {stats['bowler_frames']}, Batsman boxes: {stats['batsman_frames']}")
    print(f"  Review frames: {PREVIEW_DIR}")
    print(f"  metadata: {os.path.join(OUTPUT_DIR, 'metadata.json')}")
    print("\nNext: python scripts/review_role_labels.py        (correct mistakes)")
    print("Next: python scripts/train_bowler_batsman.py        (fine-tune YOLO)")


if __name__ == "__main__":
    main()