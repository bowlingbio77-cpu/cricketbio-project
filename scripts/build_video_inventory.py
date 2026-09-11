"""
Build the real-video inventory for the P1 ground-truth annotation pipeline.

Probes the 8 real bowling clips (the PARI-F1 sweep set identified in
evaluation/validation_report.md) and writes:

    evaluation/ground_truth/video_inventory.csv

Only reads video headers / frame metadata -- it never modifies the videos and
never invents values.  Fields that require human visual inspection (view_type,
usable_for_annotation) are recorded as UNKNOWN / pending review.

Usage:
    python scripts/build_video_inventory.py
"""
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The 8 real clips used in the PARI-F1 plausibility sweep.
REAL_CLIPS = [
    "fast_right_00000001.avi",
    "fast_right_00000002.avi",
    "fast_left_0000001.avi",
    "fast_left_00000001.avi",
    "off_left_00000001.mp4",
    "off_right_00000042.avi",
    "leg_right_00000029.avi",
    "leg_right_00000001.mp4",
]

BOWLING_DIR = os.path.join("corrected_all_data", "bowling")
OUT_DIR = os.path.join("evaluation", "ground_truth")
OUT_PATH = os.path.join(OUT_DIR, "video_inventory.csv")

COLUMNS = [
    "video_id",
    "filename",
    "path",
    "fps",
    "frame_count",
    "width",
    "height",
    "duration_s",
    "view_type",
    "usable_for_annotation",
    "notes",
    "review_date",
]


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return {"error": "open_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or fps <= 0:
        fps = None
    return {
        "fps": None if fps is None else round(float(fps), 4),
        "width": None if not width else int(width),
        "height": None if not height else int(height),
        "frame_count": None if not frame_count else int(frame_count),
    }


def notes_for(filename: str, probe_result: dict) -> str:
    kind = filename.split("_")[0]  # fast / leg / off
    return (
        f"bowling_type={kind}; PARI-F1 plausibility sweep clip "
        f"(see evaluation/validation_report.md). No per-delivery ground truth "
        f"exists for this clip."
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for filename in sorted(REAL_CLIPS):
        path = os.path.join(BOWLING_DIR, filename)
        probe_result = probe(path)
        if "error" in probe_result or not os.path.exists(path):
            rows.append({
                "video_id": os.path.splitext(filename)[0],
                "filename": filename,
                "path": path,
                "fps": "UNKNOWN",
                "frame_count": "UNKNOWN",
                "width": "UNKNOWN",
                "height": "UNKNOWN",
                "duration_s": "UNKNOWN",
                "view_type": "UNKNOWN",
                "usable_for_annotation": "UNKNOWN",
                "notes": "UNKNOWN",
                "review_date": "",
            })
            continue
        fps = probe_result["fps"]
        frame_count = probe_result["frame_count"]
        rows.append({
            "video_id": os.path.splitext(filename)[0],
            "filename": filename,
            "path": path,
            "fps": fps if fps is not None else "UNKNOWN",
            "frame_count": frame_count if frame_count is not None else "UNKNOWN",
            "width": probe_result["width"],
            "height": probe_result["height"],
            "duration_s": (round(frame_count / fps, 3)
                           if fps and frame_count else "UNKNOWN"),
            "view_type": "UNKNOWN",
            "usable_for_annotation": "UNKNOWN",
            "notes": notes_for(filename, probe_result),
            "review_date": "",
        })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips -> {OUT_PATH}")


if __name__ == "__main__":
    main()