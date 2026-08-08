"""
Verifies the YOLO11 detection + ByteTrack tracking stages work on a real
video, and saves annotated frames so you can visually confirm the bowler
is being picked up correctly.

Usage:
    python scripts/test_yolo_detection.py path/to/video.mp4 [--out_dir out]

Run scripts/setup_yolo.sh first (needs internet, one-time).
"""
import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.detection import BowlerDetector
from src.tracking import BowlerTracker, select_bowler_track


def annotate_and_save_detections(video_path: str, out_dir: str, n_preview_frames: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    detector = BowlerDetector()
    print(f"Detection backend: {detector.backend}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = sorted(set(int(total * f) for f in
                              [i / (n_preview_frames - 1) for i in range(n_preview_frames)]))
    saved = 0
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        dets = detector.detect(frame, frame_idx=idx)
        primary = detector.select_primary_bowler(dets)
        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            is_primary = (d is primary)
            color = (0, 255, 0) if is_primary else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_primary else 1)
            cv2.putText(frame, f"{d.confidence:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out_path = os.path.join(out_dir, f"frame_{idx:04d}_detections.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  frame {idx}: {len(dets)} detection(s) -> {out_path}")
        saved += 1
    cap.release()
    return saved


def run_tracking(video_path: str):
    tracker = BowlerTracker()
    print(f"\nTracking backend: {tracker.backend}")
    tracks = tracker.track_video(video_path)
    print(f"Found {len(tracks)} track(s): {[(tid, len(tr)) for tid, tr in tracks.items()]}")
    bowler = select_bowler_track(tracks)
    if bowler:
        print(f"Selected bowler track_id={bowler.track_id}, "
              f"present in {len(bowler)}/{tracks and max(len(t) for t in tracks.values())} frames")
    else:
        print("No track selected -- check confidence threshold / video framing.")
    return tracks, bowler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to a bowling video clip")
    parser.add_argument("--out_dir", default="./yolo_test_output")
    args = parser.parse_args()

    print(f"=== Step 1: Detection on sampled frames ({args.video}) ===")
    annotate_and_save_detections(args.video, args.out_dir)

    print(f"\n=== Step 2: Full-video tracking ===")
    run_tracking(args.video)

    print(f"\nDone. Check {args.out_dir}/ for annotated frames -- "
          f"the GREEN box (thicker) should be the bowler.")
