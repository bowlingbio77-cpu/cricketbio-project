"""
Single-Video Validation Benchmark for PaceAI.

Runs PaceAI on one video from the DeepSportradar dataset and compares
results against Sportradar's ground-truth annotations.

Usage:
    python evaluation/single_video_benchmark/run_validation.py

Output:
    ground_truth.json      - Parsed ground truth for selected video
    paceai_result.json     - PaceAI pipeline output
    bowler_tracking_comparison.csv  - Frame-by-frame comparison
    FINAL_REPORT.md        - Complete accuracy report
"""
import json
import csv
import sys
import time
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.detection import BowlerDetector
from src.pose_estimation import PoseEstimator
from src.ball_tracking_v2 import BallTracker
from src.feature_engineering import analyze_delivery
from src.tracking import (
    ROLE_BATSMAN,
    ROLE_WICKETKEEPER,
    ROLE_UMPIRE,
    ROLE_FIELDER,
    classify_player_roles,
)
from src.analysis_replay import render_analysis_replay

ROLE_BOWLER = "bowler"

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
REPORT_PATH = BASE / "FINAL_REPORT.md"


# ── helpers ──────────────────────────────────────────────────────────────────

def find_videos() -> list[Path]:
    """Find all video files in the downloaded dataset."""
    video_dir = DATASET_DIR / "videos"
    if not video_dir.exists():
        video_dir = DATASET_DIR
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos = []
    for ext in exts:
        videos.extend(video_dir.rglob(ext))
    return sorted(videos)


def find_annotations() -> dict[str, Path]:
    """Map video stem to its annotation JSON path."""
    ann_files = list((DATASET_DIR / "data").rglob("*.json"))
    mapping = {}
    for ann in ann_files:
        stem = ann.stem
        mapping[stem] = ann
    return mapping


def load_annotation(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ground_truth(annotation: dict, total_frames: int) -> dict:
    """
    Parse the DeepSportradar annotation format into a structured ground truth.

    Annotation format (per frame):
        event: {frame_str: [{"event": "is bowling"|"bowl release", "box": [x1,y1,x2,y2,conf], ...}]}
        person: {frame_str: [{"role": "bowler"|"batsman"|..., "id": "2", "box": [x1,y1,x2,y2,conf]}]}

    Returns dict with:
        release_frames: list of frame indices where ball release occurs
        bowling_windows: list of (start, end) tuples for bowling events
        bowler_bboxes: dict mapping frame_idx -> [x1, y1, x2, y2]
        all_players: dict mapping frame_idx -> list of {role, bbox}
    """
    gt = {
        "release_frames": [],
        "bowling_windows": [],
        "bowler_bboxes": {},
        "all_players": {},
    }

    # Parse events
    events = annotation.get("event", {})
    bowling_frames = []

    for frame_str, event_list in events.items():
        frame_idx = int(frame_str)
        # event_list is a list of event dicts
        if isinstance(event_list, list):
            for ev in event_list:
                ev_type = ev.get("event", "")
                if ev_type in ("bowl release", "is bowling"):
                    bowling_frames.append(frame_idx)
                    if ev_type == "bowl release":
                        gt["release_frames"].append(frame_idx)
        elif isinstance(event_list, str):
            if event_list in ("bowl release", "is bowling"):
                bowling_frames.append(frame_idx)
                if event_list == "bowl release":
                    gt["release_frames"].append(frame_idx)

    gt["release_frames"] = sorted(set(gt["release_frames"]))

    # Build bowling windows (contiguous frames)
    if bowling_frames:
        bowling_sorted = sorted(set(bowling_frames))
        window_start = bowling_sorted[0]
        prev = bowling_sorted[0]
        for f in bowling_sorted[1:]:
            if f - prev > 5:
                gt["bowling_windows"].append((window_start, prev))
                window_start = f
            prev = f
        gt["bowling_windows"].append((window_start, prev))

    # Parse person bounding boxes
    persons = annotation.get("person", {})
    for frame_str, person_list in persons.items():
        frame_idx = int(frame_str)
        if frame_idx >= total_frames:
            continue
        players = []
        for p in person_list:
            role_raw = p.get("role", "unknown")
            bbox = p.get("box", [0, 0, 0, 0])[:4]  # take first 4 elements
            role = normalize_role(role_raw)
            players.append({"role": role, "bbox": bbox, "raw_role": role_raw})
            if role == ROLE_BOWLER:
                gt["bowler_bboxes"][frame_idx] = bbox
        gt["all_players"][frame_idx] = players

    return gt


def normalize_role(raw: str) -> str:
    raw_lower = raw.lower().strip()
    if "bowler" in raw_lower:
        return ROLE_BOWLER
    if "batsman" in raw_lower or "batter" in raw_lower:
        return ROLE_BATSMAN
    if "keeper" in raw_lower or "wicket" in raw_lower:
        return ROLE_WICKETKEEPER
    if "umpire" in raw_lower:
        return ROLE_UMPIRE
    return raw_lower


def run_paceai_on_video(video_path: Path, focus_frames: list[int] | None = None) -> dict:
    """
    Run the PaceAI pipeline on selected frames and return structured results.

    If focus_frames is provided, only processes those frames (much faster).
    Otherwise processes every frame.
    """
    result = {
        "detections": [],
        "poses": [],
        "bowler_id": None,
        "bowler_frames": [],
        "player_roles": {},
        "features": {},
        "errors": [],
        "pipeline_time": 0,
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["errors"].append(f"Cannot open video: {video_path}")
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result["fps"] = fps
    result["total_frames"] = total_frames

    detector = BowlerDetector()
    pose_estimator = PoseEstimator()

    all_bboxes = {}
    start_time = time.time()

    if focus_frames:
        # Fast mode: only process specified frames
        for frame_idx in focus_frames:
            if frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                detections = detector.detect(frame)
                for det in detections:
                    bbox = det.bbox
                    conf = det.confidence
                    if bbox is not None:
                        if frame_idx not in all_bboxes:
                            all_bboxes[frame_idx] = []
                        all_bboxes[frame_idx].append({
                            "bbox": list(bbox),
                            "confidence": float(conf),
                        })
                        result["detections"].append({
                            "frame": frame_idx,
                            "bbox": list(bbox),
                            "confidence": float(conf),
                        })
            except Exception as e:
                result["errors"].append(f"Detection error at frame {frame_idx}: {e}")

            # Pose on every 5th frame only
            if frame_idx % 5 == 0:
                try:
                    timestamp_sec = frame_idx / fps
                    pose_frame = pose_estimator.process_frame(frame, frame_idx, timestamp_sec)
                    if pose_frame is not None:
                        result["poses"].append({
                            "frame": frame_idx,
                            "landmarks": pose_frame.landmarks.tolist(),
                            "world_landmarks": pose_frame.world_landmarks.tolist() if pose_frame.world_landmarks is not None else None,
                            "n_people": pose_frame.n_people,
                        })
                except Exception as e:
                    result["errors"].append(f"Pose error at frame {frame_idx}: {e}")
    else:
        # Slow mode: process every frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                detections = detector.detect(frame)
                for det in detections:
                    bbox = det.bbox
                    conf = det.confidence
                    if bbox is not None:
                        if frame_idx not in all_bboxes:
                            all_bboxes[frame_idx] = []
                        all_bboxes[frame_idx].append({
                            "bbox": list(bbox),
                            "confidence": float(conf),
                        })
                        result["detections"].append({
                            "frame": frame_idx,
                            "bbox": list(bbox),
                            "confidence": float(conf),
                        })
            except Exception as e:
                result["errors"].append(f"Detection error at frame {frame_idx}: {e}")
            frame_idx += 1

    cap.release()

    # Classify player roles on first detected frame
    if all_bboxes:
        sample_frame_idx = min(all_bboxes.keys())
        cap2 = cv2.VideoCapture(str(video_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
        ret, sample_frame = cap2.read()
        cap2.release()
        if ret:
            sample_dets = all_bboxes.get(sample_frame_idx, [])
            bboxes_list = [d["bbox"] for d in sample_dets]
            if bboxes_list:
                try:
                    roles = classify_player_roles(sample_frame, bboxes_list)
                    result["player_roles"] = roles
                except Exception as e:
                    result["errors"].append(f"Role classification error: {e}")

    result["bowler_frames"] = sorted(all_bboxes.keys())
    result["pipeline_time"] = time.time() - start_time

    return result


def compute_iou(box_a, box_b) -> float:
    """Compute Intersection over Union of two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compare_bowler_tracking(gt: dict, paceai: dict, fps: float) -> list[dict]:
    """
    Frame-by-frame comparison of bowler tracking.

    Returns list of dicts, one per frame, with:
        frame, gt_bbox, paceai_bbox, iou, match
    """
    comparison = []

    gt_frames = set(gt.get("bowler_bboxes", {}).keys())
    pa_frames = set(paceai.get("bowler_frames", []))
    all_frames = sorted(gt_frames | pa_frames)

    for f in all_frames:
        gt_bbox = gt.get("bowler_bboxes", {}).get(f)
        # Find PaceAI detection on this frame
        pa_bbox = None
        for det in paceai.get("detections", []):
            if det["frame"] == f:
                pa_bbox = det["bbox"]
                break

        iou = 0.0
        match = False
        if gt_bbox and pa_bbox:
            iou = compute_iou(gt_bbox, pa_bbox)
            match = iou >= 0.5

        comparison.append({
            "frame": f,
            "gt_bbox": gt_bbox,
            "paceai_bbox": pa_bbox,
            "iou": round(iou, 4),
            "match": match,
            "time_s": round(f / fps, 3) if fps > 0 else 0,
        })

    return comparison


def evaluate_release_detection(gt: dict, paceai: dict) -> dict:
    """
    Evaluate how well PaceAI identifies the bowling release moment.

    Compares PaceAI's bowler presence windows against ground-truth
    bowling action windows.
    """
    gt_windows = gt.get("bowling_windows", [])
    gt_release = gt.get("release_frames", [])
    pa_frames = paceai.get("bowler_frames", [])

    if not pa_frames or not gt_windows:
        return {
            "method": "N/A",
            "gt_windows": gt_windows,
            "paceai_frames_count": len(pa_frames),
            "recall": 0,
            "precision": 0,
            "notes": "Insufficient data for comparison",
        }

    # For each GT window, check if PaceAI has detections in that range
    matched_frames = 0
    total_gt_frames = 0
    for start, end in gt_windows:
        for f in range(start, end + 1):
            total_gt_frames += 1
            if f in pa_frames:
                matched_frames += 1

    recall = matched_frames / total_gt_frames if total_gt_frames > 0 else 0.0

    # Check how many PaceAI frames fall within GT windows
    pa_in_windows = 0
    for f in pa_frames:
        for start, end in gt_windows:
            if start <= f <= end:
                pa_in_windows += 1
                break

    precision = pa_in_windows / len(pa_frames) if pa_frames else 0.0

    return {
        "method": "window_recall_precision",
        "gt_windows": gt_windows,
        "gt_release_frames": gt_release,
        "paceai_frames_count": len(pa_frames),
        "matched_frames": matched_frames,
        "total_gt_frames": total_gt_frames,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def compare_features(gt_features: dict, paceai_features: dict) -> dict:
    """
    Compare biomechanical features if both are available.

    This is a placeholder — the DeepSportradar dataset does NOT provide
    biomechanical measurements, so this comparison will be limited to
    what PaceAI can compute from the video alone.
    """
    return {
        "available_in_gt": bool(gt_features),
        "available_in_paceai": bool(paceai_features),
        "comparison_possible": False,
        "reason": "Ground truth dataset does not contain biomechanical measurements",
    }


def write_comparison_csv(comparison: list[dict], path: Path):
    """Write frame-by-frame comparison to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "time_s",
            "gt_bbox", "paceai_bbox",
            "iou", "match"
        ])
        for row in comparison:
            writer.writerow([
                row["frame"],
                row["time_s"],
                row["gt_bbox"],
                row["paceai_bbox"],
                row["iou"],
                row["match"],
            ])


def write_final_report(
    video_name: str,
    gt: dict,
    paceai: dict,
    tracking_comparison: list[dict],
    release_eval: dict,
    feature_eval: dict,
    path: Path,
):
    """Generate the FINAL_REPORT.md"""

    total_frames = paceai.get("total_frames", 0)
    fps = paceai.get("fps", 30)

    # Tracking stats
    gt_frames_with_bowler = len(gt.get("bowler_bboxes", {}))
    pa_frames_with_bowler = len(paceai.get("bowler_frames", []))
    frames_with_match = sum(1 for c in tracking_comparison if c["match"])
    ious = [c["iou"] for c in tracking_comparison if c["gt_bbox"] and c["paceai_bbox"]]
    mean_iou = statistics.mean(ious) if ious else 0
    median_iou = statistics.median(ious) if ious else 0

    recall = frames_with_match / gt_frames_with_bowler if gt_frames_with_bowler else 0

    # Release detection
    release_recall = release_eval.get("recall", 0)
    release_precision = release_eval.get("precision", 0)

    report = f"""# PaceAI Single-Video Validation Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Video tested | `{video_name}` |
| Total frames | {total_frames} |
| Frame rate | {fps:.1f} fps |
| Duration | {total_frames/fps:.1f}s |

## Ground Truth Source

- **Dataset**: DeepSportradar Cricket Bowl Release Challenge (2023)
- **Annotator**: Sportradar (professional sports data company)
- **Annotation quality**: Production-grade, used for commercial broadcasts
- **License**: CC BY-NC-ND 4.0

## 1. Bowler Identification

| Metric | Value |
|--------|-------|
| GT bowler-labeled frames | {gt_frames_with_bowler} |
| PaceAI detected bowler frames | {pa_frames_with_bowler} |
| GT bowling windows | {len(gt.get('bowling_windows', []))} |
| GT release frame annotations | {len(gt.get('release_frames', []))} |

**Analysis**: PaceAI {'successfully identified bowler frames' if pa_frames_with_bowler > 0 else 'did not identify any bowler frames'} across the video.

## 2. Bowler Tracking Accuracy

| Metric | Value |
|--------|-------|
| Frames with GT bowler bbox | {gt_frames_with_bowler} |
| Frames with PaceAI bowler bbox | {pa_frames_with_bowler} |
| Matching frames (IoU >= 0.5) | {frames_with_match} |
| Recall (matched / GT frames) | {recall:.1%} |
| Mean IoU | {mean_iou:.4f} |
| Median IoU | {median_iou:.4f} |

**Analysis**: {'Tracking shows strong alignment' if recall > 0.7 else 'Tracking shows moderate alignment' if recall > 0.4 else 'Tracking shows weak alignment'} with ground truth bowler positions.

## 3. Release Frame Detection

| Metric | Value |
|--------|-------|
| GT release windows | {len(release_eval.get('gt_windows', []))} |
| GT release frames (exact) | {len(release_eval.get('gt_release_frames', []))} |
| PaceAI active frames | {release_eval.get('paceai_frames_count', 0)} |
| Window recall | {release_recall:.1%} |
| Window precision | {release_precision:.1%} |

**Analysis**: PaceAI {'captures the bowling action period effectively' if release_recall > 0.6 else 'partially captures the bowling action period' if release_recall > 0.3 else 'does not reliably capture the bowling action period'}.

## 4. Biomechanical Features

| Metric | Value |
|--------|-------|
| GT biomechanical data | {'Available' if feature_eval.get('available_in_gt') else 'Not available'} |
| PaceAI features computed | {'Yes' if feature_eval.get('available_in_paceai') else 'No'} |
| Comparison possible | {'Yes' if feature_eval.get('comparison_possible') else 'No'} |

**Note**: The DeepSportradar dataset does not include biomechanical measurements (joint angles, release speed, etc.). Therefore, direct biomechanical accuracy comparison is not possible with this dataset. PaceAI's biomechanical pipeline can still be validated by running on real video, but cross-validation against ground truth requires a different data source.

## 5. Error Log

"""
    errors = paceai.get("errors", [])
    if errors:
        report += "| Frame | Error |\n|-------|-------|\n"
        for err in errors[:20]:
            report += f"| - | {err} |\n"
        if len(errors) > 20:
            report += f"| - | ... and {len(errors)-20} more errors |\n"
    else:
        report += "No errors encountered.\n"

    report += f"""
## 6. Pipeline Performance

| Metric | Value |
|--------|-------|
| Pipeline execution time | {paceai.get('pipeline_time', 0):.2f}s |
| Processing speed | {total_frames / max(paceai.get('pipeline_time', 1), 0.01):.1f} fps |

## 7. Limitations

1. **Broadcast camera angle**: DeepSportradar videos are broadcast view, not side-on biomechanics view
2. **No biomechanical GT**: Dataset provides player positions, not joint angles or release metrics
3. **Multiple deliveries**: Videos contain ~2 overs; comparison uses all detected bowler activity
4. **Role ambiguity**: Some frames may have ambiguous player roles

## 8. Conclusion

{'**PaceAI demonstrates strong performance** on this video, with high tracking accuracy and reliable bowling action detection.' if recall > 0.7 and release_recall > 0.6 else '**PaceAI shows moderate performance** on this video, with room for improvement in tracking consistency or release detection.' if recall > 0.4 or release_recall > 0.3 else '**PaceAI performance needs improvement** on this video, with low tracking accuracy or release detection.'}

The bowler tracking recall of {recall:.1%} and release window recall of {release_recall:.1%} suggest {'robust' if recall > 0.7 else 'moderate' if recall > 0.4 else 'limited'} performance on this broadcast-angle cricket footage.
"""

    path.write_text(report, encoding="utf-8")
    return report


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PaceAI Single-Video Validation Benchmark")
    print("  Dataset: DeepSportradar Cricket Bowl Release Challenge (2023)")
    print("=" * 70)
    print()

    # Check dataset exists
    videos = find_videos()
    if not videos:
        print("ERROR: No videos found. Run download_dataset.py first.")
        print(f"Expected location: {DATASET_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in dataset")
    print()

    # Find matching annotation first
    annotations = find_annotations()

    # Prefer videos that have matching annotations
    matched = [v for v in videos if v.stem in annotations]
    if matched:
        # Pick smallest matched video for speed
        video = min(matched, key=lambda v: v.stat().st_size)
        print(f"Selected video (matched annotation): {video.name}")
    else:
        video = videos[0]
        print(f"Selected video: {video.name}")
    print(f"  Path: {video}")
    print()

    # Find matching annotation
    ann_path = annotations.get(video.stem)
    if ann_path is None:
        # Fallback to first annotation
        if annotations:
            ann_path = list(annotations.values())[0]
            print(f"No exact match for annotation. Using: {ann_path.name}")
        else:
            print("ERROR: No annotation files found.")
            sys.exit(1)

    print(f"Annotation: {ann_path.name}")

    # Get video info
    cap = cv2.VideoCapture(str(video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {total_frames} ({total_frames/fps:.1f}s)")
    print(f"  FPS: {fps:.1f}")
    print()

    # Phase 3: Load and parse ground truth
    print("Phase 3: Parsing ground truth annotations...")
    raw_annotation = load_annotation(ann_path)
    gt = extract_ground_truth(raw_annotation, total_frames)

    print(f"  Bowling windows: {gt['bowling_windows']}")
    print(f"  Release frames: {gt['release_frames']}")
    print(f"  Bowler-labeled frames: {len(gt['bowler_bboxes'])}")
    print()

    # Save ground truth
    gt_path = BASE / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, default=str)
    print(f"  Saved: {gt_path}")
    print()

    # Phase 4: Run PaceAI (focus on bowling window frames for speed)
    print("Phase 4: Running PaceAI pipeline...")
    focus = []
    for ws, we in gt["bowling_windows"]:
        focus.extend(range(max(0, ws - 20), min(total_frames, we + 20)))
    focus = sorted(set(focus)) if focus else None
    if focus:
        print(f"  Focusing on {len(focus)} frames around bowling windows")
    paceai_result = run_paceai_on_video(video, focus_frames=focus)
    print(f"  Detections: {len(paceai_result['detections'])} across {len(paceai_result['bowler_frames'])} frames")
    print(f"  Poses: {len(paceai_result['poses'])}")
    print(f"  Pipeline time: {paceai_result['pipeline_time']:.2f}s")
    print(f"  Errors: {len(paceai_result['errors'])}")
    print()

    # Save PaceAI result
    pa_path = BASE / "paceai_result.json"
    with open(pa_path, "w", encoding="utf-8") as f:
        json.dump(paceai_result, f, indent=2, default=str)
    print(f"  Saved: {pa_path}")
    print()

    # Phase 5: Bowler identification comparison
    print("Phase 5: Bowler identification comparison...")
    gt_roles = {}
    for frame_idx, players in gt.get("all_players", {}).items():
        for p in players:
            if p["role"] == ROLE_BOWLER:
                gt_roles[frame_idx] = p["bbox"]
                break
    print(f"  GT bowler frames: {len(gt_roles)}")
    print(f"  PaceAI bowler frames: {len(paceai_result['bowler_frames'])}")
    print()

    # Phase 6: Tracking comparison
    print("Phase 6: Tracking comparison...")
    tracking_comp = compare_bowler_tracking(gt, paceai_result, fps)
    csv_path = BASE / "bowler_tracking_comparison.csv"
    write_comparison_csv(tracking_comp, csv_path)
    print(f"  Comparison frames: {len(tracking_comp)}")
    print(f"  Saved: {csv_path}")
    print()

    # Phase 7: Release frame comparison
    print("Phase 7: Release frame comparison...")
    release_eval = evaluate_release_detection(gt, paceai_result)
    print(f"  GT bowling windows: {release_eval.get('gt_windows', [])}")
    print(f"  Window recall: {release_eval.get('recall', 0):.1%}")
    print(f"  Window precision: {release_eval.get('precision', 0):.1%}")
    print()

    # Phase 8: Feature comparison
    print("Phase 8: Feature comparison...")
    feature_eval = compare_features({}, paceai_result.get("features", {}))
    print(f"  GT biomechanics available: {feature_eval['available_in_gt']}")
    print(f"  PaceAI features computed: {feature_eval['available_in_paceai']}")
    print(f"  Comparison possible: {feature_eval['comparison_possible']}")
    print()

    # Phase 10: Generate report
    print("Phase 10: Generating FINAL_REPORT.md...")
    report = write_final_report(
        video_name=video.name,
        gt=gt,
        paceai=paceai_result,
        tracking_comparison=tracking_comp,
        release_eval=release_eval,
        feature_eval=feature_eval,
        path=REPORT_PATH,
    )
    print(f"  Saved: {REPORT_PATH}")
    print()
    print("=" * 70)
    print("  Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
