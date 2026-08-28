"""
PaceAI Evaluation Framework

Computes detection, tracking, pose, and release-frame metrics against
ground-truth annotations. Requires annotated bowling videos in the
evaluation/ directory structure.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --video_id fast_right_1
"""
import os
import sys
import csv
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(EVAL_DIR, "videos")
ANNOTATIONS_DIR = os.path.join(EVAL_DIR, "annotations")
PREDICTIONS_DIR = os.path.join(EVAL_DIR, "predictions")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@dataclass
class BallAnnotation:
    frame_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class PoseAnnotation:
    frame_number: int
    landmark_name: str
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class VideoMetadata:
    video_id: str
    video_path: str
    fps: float
    width: int
    height: int
    bowling_arm: str
    camera_view: str
    release_frame_annotated: Optional[int] = None


@dataclass
class EvaluationDataset:
    videos: List[VideoMetadata] = field(default_factory=list)
    ball_annotations: Dict[str, List[BallAnnotation]] = field(default_factory=dict)
    pose_annotations: Dict[str, List[PoseAnnotation]] = field(default_factory=dict)


def load_metadata() -> List[VideoMetadata]:
    """Load video metadata from evaluation/metadata.csv."""
    path = os.path.join(EVAL_DIR, "metadata.csv")
    if not os.path.exists(path):
        return []
    videos = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("video_id") or not row.get("video_path"):
                continue
            videos.append(VideoMetadata(
                video_id=row["video_id"],
                video_path=row["video_path"],
                fps=float(row.get("fps", 20)),
                width=int(row.get("width", 640)),
                height=int(row.get("height", 360)),
                bowling_arm=row.get("bowling_arm", "right"),
                camera_view=row.get("camera_view", "behind"),
                release_frame_annotated=int(row["release_frame_annotated"])
                if row.get("release_frame_annotated") else None,
            ))
    return videos


def load_ball_annotations(video_id: str) -> List[BallAnnotation]:
    """Load frame-level ball bounding box annotations for a video."""
    path = os.path.join(ANNOTATIONS_DIR, f"{video_id}.csv")
    if not os.path.exists(path):
        return []
    annotations = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(BallAnnotation(
                frame_number=int(row["frame_number"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                confidence=float(row.get("confidence", 1.0)),
            ))
    return annotations


def load_pose_annotations(video_id: str) -> List[PoseAnnotation]:
    """Load pose landmark annotations for a video."""
    path = os.path.join(ANNOTATIONS_DIR, f"{video_id}_pose.csv")
    if not os.path.exists(path):
        return []
    annotations = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(PoseAnnotation(
                frame_number=int(row["frame_number"]),
                landmark_name=row["landmark_name"],
                x=float(row["x"]),
                y=float(row["y"]),
                confidence=float(row.get("confidence", 1.0)),
            ))
    return annotations


def load_dataset() -> EvaluationDataset:
    """Load the complete evaluation dataset."""
    ds = EvaluationDataset()
    ds.videos = load_metadata()
    for vid in ds.videos:
        ds.ball_annotations[vid.video_id] = load_ball_annotations(vid.video_id)
        ds.pose_annotations[vid.video_id] = load_pose_annotations(vid.video_id)
    return ds


# --------------------------------------------------------------------------- #
# Pipeline-prediction cache
# --------------------------------------------------------------------------- #
# Running the full CV pipeline on real clips is slow (~minutes). To keep the
# eval loop fast and re-runnable, `predict_video()` runs the REAL pipeline once
# and caches the relevant outputs to evaluation/predictions/<video_id>.json.
# `run_evaluation()` then loads the cached predictions and feeds REAL values
# into the metric functions below -- no fabricated data anywhere.


class SimplePoint:
    """Lightweight BallPoint stand-in (frame_idx/x/y/detected/source/w/h/conf)."""
    __slots__ = ("frame_idx", "x", "y", "detected", "source", "w", "h", "confidence")

    def __init__(self, frame_idx, x, y, detected, source, w, h, confidence=1.0):
        self.frame_idx = frame_idx
        self.x = x
        self.y = y
        self.detected = detected
        self.source = source
        self.w = w
        self.h = h
        self.confidence = confidence


def predict_video(vid: VideoMetadata) -> dict:
    """Run the real analyze_video pipeline, extract predictions, cache to JSON.

    Only CONSUMES pipeline output -- does not modify the ML/CV pipeline.
    """
    from src import pipeline

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    result = pipeline.analyze_video(
        vid.video_path,
        bowling_arm=vid.bowling_arm,
        target_fps=int(vid.fps),
        camera_view=vid.camera_view,
        run_ml=False,
    )

    traj = []
    for p in (result.ball_stats or {}).get("trajectory") or []:
        traj.append({
            "frame_idx": p.frame_idx,
            "x": float(p.x), "y": float(p.y),
            "detected": bool(getattr(p, "detected", True)),
            "source": getattr(p, "source", "motion"),
            "w": float(getattr(p, "w", 0.0)),
            "h": float(getattr(p, "h", 0.0)),
            "confidence": float(getattr(p, "confidence", 1.0)),
        })

    pred = {
        "video_id": vid.video_id,
        "fps": vid.fps,
        "release_idx": result.ball_stats.get("release_idx"),
        "n_frames": result.ball_stats.get("n_frames"),
        "trajectory": traj,
    }
    path = os.path.join(PREDICTIONS_DIR, f"{vid.video_id}_pred.json")
    with open(path, "w") as f:
        json.dump(pred, f, indent=2)
    return pred


def load_predictions(video_id: str) -> Optional[dict]:
    """Load cached pipeline predictions for a video (None if missing)."""
    path = os.path.join(PREDICTIONS_DIR, f"{video_id}_pred.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def trajectory_to_points(traj: list) -> list:
    """Convert cached trajectory dicts back into SimplePoint objects."""
    out = []
    for p in traj:
        out.append(SimplePoint(
            int(p["frame_idx"]),
            float(p["x"]), float(p["y"]),
            bool(p.get("detected", True)),
            p.get("source", "motion"),
            float(p.get("w", 0.0)), float(p.get("h", 0.0)),
            float(p.get("confidence", 1.0)),
        ))
    return out


def trajectory_to_boxes(traj: list, box_pad: float = 6.0):
    """Predict per-frame ball boxes (frame_idx -> (x1,y1,x2,y2,conf)) from the
    tracked trajectory centers, using the tracked width/height (or a small pad).
    conf = 1.0 for YOLO-detected points, 0.5 for interpolated/predicted."""
    boxes = {}
    for p in traj:
        w = float(p.get("w", 0.0)) or box_pad
        h = float(p.get("h", 0.0)) or box_pad
        x, y = float(p["x"]), float(p["y"])
        conf = 1.0 if p.get("detected", True) and p.get("source", "") == "yolo" else 0.5
        boxes[int(p["frame_idx"])] = [(x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf)]
    return boxes


# --------------------------------------------------------------------------- #
# Metric: Ball Detection mAP50
# --------------------------------------------------------------------------- #

def compute_iou(box_a, box_b) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_detection_metrics(
    predictions: Dict[int, List[Tuple[float, float, float, float, float]]],
    ground_truth: List[BallAnnotation],
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.1,
) -> dict:
    """
    Compute precision, recall, and mAP@50 for ball detection.

    Parameters
    ----------
    predictions : dict mapping frame_number -> list of (x1, y1, x2, y2, confidence)
    ground_truth : list of BallAnnotation for the video
    iou_threshold : IoU threshold for a match
    conf_threshold : minimum confidence to consider a detection

    Returns
    -------
    dict with precision, recall, ap50, true_positives, false_positives, false_negatives
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "ap50": 0.0,
                "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                "status": "no_ground_truth"}

    gt_by_frame = {}
    for ann in ground_truth:
        gt_by_frame.setdefault(ann.frame_number, []).append(ann)

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for frame_idx, gt_boxes in gt_by_frame.items():
        pred_boxes = predictions.get(frame_idx, [])
        pred_boxes = [(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in pred_boxes
                       if c >= conf_threshold]
        pred_boxes.sort(key=lambda b: b[4], reverse=True)

        matched_gt = set()
        matched_pred = set()

        for pi, (px1, py1, px2, py2, pconf) in enumerate(pred_boxes):
            best_iou = 0.0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou((px1, py1, px2, py2),
                                  (gt.x1, gt.y1, gt.x2, gt.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold:
                tp_total += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_total += 1

        fn_total += len(gt_boxes) - len(matched_gt)

    total_gt = len(ground_truth)
    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, total_gt)
    ap50 = precision * recall  # simplified AP for single-threshold

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ap50": round(ap50, 4),
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "total_ground_truth": total_gt,
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Metric: Tracking Consistency
# --------------------------------------------------------------------------- #

def compute_tracking_metrics(
    trajectory,  # list of BallPoint
    ground_truth: List[BallAnnotation],
    fps: float = 20.0,
) -> dict:
    """
    Compute tracking consistency metrics.

    Parameters
    ----------
    trajectory : list of BallPoint from the tracker
    ground_truth : list of BallAnnotation
    fps : video FPS for time conversion

    Returns
    -------
    dict with coverage_pct, detection_ratio, id_switches, trajectory_length
    """
    if not trajectory or not ground_truth:
        return {"coverage_pct": 0.0, "detection_ratio": 0.0,
                "id_switches": 0, "status": "no_data"}

    gt_frames = set(a.frame_number for a in ground_truth)
    tracked_frames = set(p.frame_idx for p in trajectory)
    detected_frames = set(p.frame_idx for p in trajectory if p.detected)

    coverage = len(tracked_frames & gt_frames) / max(1, len(gt_frames))
    detection_ratio = len(detected_frames & gt_frames) / max(1, len(gt_frames))

    id_switches = 0
    prev_source = None
    for p in trajectory:
        if p.source != prev_source and prev_source is not None:
            if p.source in ("yolo", "wrist_proxy") and prev_source in ("yolo", "wrist_proxy"):
                id_switches += 1
        prev_source = p.source

    return {
        "coverage_pct": round(coverage * 100, 1),
        "detection_ratio": round(detection_ratio, 4),
        "id_switches": id_switches,
        "trajectory_length": len(trajectory),
        "gt_frames_count": len(gt_frames),
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Metric: Release Frame Accuracy
# --------------------------------------------------------------------------- #

def compute_release_frame_metrics(
    predicted_release: Optional[int],
    annotated_release: Optional[int],
    fps: float = 20.0,
) -> dict:
    """
    Compute release frame accuracy.

    Parameters
    ----------
    predicted_release : frame index from the pipeline
    annotated_release : ground-truth frame index
    fps : video FPS

    Returns
    -------
    dict with absolute_error, time_error_s
    """
    if predicted_release is None or annotated_release is None:
        return {"absolute_error": None, "time_error_s": None,
                "status": "not_available"}

    abs_error = abs(predicted_release - annotated_release)
    time_error = abs_error / fps if fps > 0 else 0.0

    return {
        "absolute_error": abs_error,
        "time_error_s": round(time_error, 4),
        "predicted": predicted_release,
        "annotated": annotated_release,
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Metric: Pose Accuracy
# --------------------------------------------------------------------------- #

def compute_pose_metrics(
    predicted_landmarks: Dict[str, Tuple[float, float]],  # name -> (x, y)
    ground_truth: List[PoseAnnotation],
    image_width: int = 640,
    image_height: int = 360,
) -> dict:
    """
    Compute pose landmark accuracy.

    Parameters
    ----------
    predicted_landmarks : dict mapping landmark name -> (x, y) pixel coords
    ground_truth : list of PoseAnnotation
    image_width, image_height : for normalization

    Returns
    -------
    dict with per-landmark RMSE and normalized RMSE
    """
    if not ground_truth or not predicted_landmarks:
        return {"status": "not_available", "per_landmark": {}}

    diag = np.hypot(image_width, image_height)
    gt_by_name = {}
    for ann in ground_truth:
        gt_by_name[ann.landmark_name] = (ann.x, ann.y)

    per_landmark = {}
    errors = []
    for name, (gx, gy) in gt_by_name.items():
        if name not in predicted_landmarks:
            continue
        px, py = predicted_landmarks[name]
        err = np.hypot(px - gx, py - gy)
        norm_err = err / diag
        per_landmark[name] = {
            "pixel_error": round(float(err), 2),
            "normalized_error": round(float(norm_err), 4),
        }
        errors.append(err)

    if not errors:
        return {"status": "no_matching_landmarks", "per_landmark": {}}

    return {
        "mean_pixel_error": round(float(np.mean(errors)), 2),
        "median_pixel_error": round(float(np.median(errors)), 2),
        "max_pixel_error": round(float(np.max(errors)), 2),
        "mean_normalized_error": round(float(np.mean(errors) / diag), 4),
        "per_landmark": per_landmark,
        "landmarks_evaluated": len(errors),
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Metric: Wrist-Proxy Reliability
# --------------------------------------------------------------------------- #

def compute_wrist_proxy_reliability(
    wrist_visibilities: List[float],
    delivery_reliable: bool,
    release_frame_detected: bool,
) -> dict:
    """
    Quality flag for wrist-proxy tracking.

    Parameters
    ----------
    wrist_visibilities : per-frame wrist landmark visibility scores
    delivery_reliable : whether the delivery phase detection was reliable
    release_frame_detected : whether a release frame was found

    Returns
    -------
    dict with quality_level, avg_visibility, confidence_reason
    """
    if not wrist_visibilities:
        return {"quality_level": "LOW", "avg_visibility": 0.0,
                "confidence_reason": "No wrist landmark data available",
                "status": "not_available"}

    avg_vis = float(np.mean(wrist_visibilities))

    if avg_vis > 0.8 and delivery_reliable and release_frame_detected:
        level = "HIGH"
        reason = "High wrist visibility with reliable delivery detection"
    elif avg_vis > 0.5 or (delivery_reliable and release_frame_detected):
        level = "MEDIUM"
        reasons = []
        if avg_vis <= 0.5:
            reasons.append(f"moderate wrist visibility ({avg_vis:.2f})")
        if not delivery_reliable:
            reasons.append("delivery phase detection unreliable")
        if not release_frame_detected:
            reasons.append("release frame not detected")
        reason = "Wrist-proxy: " + "; ".join(reasons)
    else:
        level = "LOW"
        reason = (f"Low wrist visibility ({avg_vis:.2f}) and "
                  "unreliable delivery detection")

    return {
        "quality_level": level,
        "avg_visibility": round(avg_vis, 4),
        "confidence_reason": reason,
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Metric: Reels Quality
# --------------------------------------------------------------------------- #

def compute_reels_quality(trajectory, slow_factor: float = 2.5) -> dict:
    """
    Compute slow-motion rendering quality metrics.

    Parameters
    ----------
    trajectory : list of BallPoint (the display trajectory)
    slow_factor : applied slow-motion factor

    Returns
    -------
    dict with frame_duplication_ratio, max_center_displacement
    """
    if not trajectory or len(trajectory) < 2:
        return {"status": "not_available"}

    pts = np.array([[p.x, p.y] for p in trajectory], dtype=float)
    displacements = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))

    return {
        "max_center_displacement_px": round(float(np.max(displacements)), 2),
        "mean_center_displacement_px": round(float(np.mean(displacements)), 2),
        "median_center_displacement_px": round(float(np.median(displacements)), 2),
        "trajectory_points": len(trajectory),
        "status": "measured",
    }


# --------------------------------------------------------------------------- #
# Full evaluation runner
# --------------------------------------------------------------------------- #

def run_evaluation(video_id: Optional[str] = None) -> dict:
    """
    Run the full evaluation pipeline.

    If video_id is None, evaluates all videos in the dataset.
    If no data is available, reports NOT MEASURED for all metrics.
    """
    ds = load_dataset()

    if not ds.videos:
        return {
            "status": "NO_DATA",
            "message": ("Evaluation dataset not yet available. "
                        "Add annotated videos to evaluation/videos/ and "
                        "annotations to evaluation/annotations/."),
            "detection": {"status": "NOT_MEASURED"},
            "tracking": {"status": "NOT_MEASURED"},
            "release_frame": {"status": "NOT_MEASURED"},
            "pose": {"status": "NOT_MEASURED"},
            "wrist_proxy": {"status": "NOT_MEASURED"},
            "reels": {"status": "NOT_MEASURED"},
        }

    videos_to_eval = ds.videos
    if video_id:
        videos_to_eval = [v for v in ds.videos if v.video_id == video_id]
        if not videos_to_eval:
            return {"status": "VIDEO_NOT_FOUND", "video_id": video_id}

    results = {
        "status": "measured",
        "n_videos": len(videos_to_eval),
        "videos": {},
        "aggregate": {},
    }

    all_release = []

    for vid in videos_to_eval:
        v_result = {}
        pred = load_predictions(vid.video_id)

        gt = ds.ball_annotations.get(vid.video_id, [])
        v_result["ground_truth_count"] = len(gt)
        v_result["has_predictions"] = pred is not None

        if pred is not None:
            fps = vid.fps or pred.get("fps") or 20.0
            traj = trajectory_to_points(pred.get("trajectory") or [])
            boxes = trajectory_to_boxes(pred.get("trajectory") or [])

            v_result["detection"] = compute_detection_metrics(boxes, gt)
            v_result["tracking"] = compute_tracking_metrics(traj, gt, fps)

            pr_release = pred.get("release_idx")
            v_result["release_frame"] = compute_release_frame_metrics(
                pr_release, vid.release_frame_annotated, fps)
            if v_result["release_frame"]["status"] == "measured":
                all_release.append(v_result["release_frame"])

            v_result["reels"] = compute_reels_quality(traj)
            v_result["wrist_proxy"] = compute_wrist_proxy_reliability(
                [p.confidence for p in traj if getattr(p, "source", "") == "wrist_proxy"],
                v_result["tracking"].get("status") == "measured",
                pr_release is not None,
            )
        else:
            v_result["detection"] = compute_detection_metrics({}, gt)
            v_result["tracking"] = compute_tracking_metrics([], gt, vid.fps)
            v_result["release_frame"] = compute_release_frame_metrics(
                None, vid.release_frame_annotated, vid.fps)
            v_result["reels"] = compute_reels_quality([])
            v_result["wrist_proxy"] = compute_wrist_proxy_reliability([], False, False)

        pose_gt = ds.pose_annotations.get(vid.video_id, [])
        v_result["pose"] = compute_pose_metrics({}, pose_gt, vid.width, vid.height)

        results["videos"][vid.video_id] = v_result

    # Aggregate metrics across all evaluated videos (only from measured values --
    # videos without predictions/annotations are excluded so we never report a
    # made-up number).
    det_measured = [v["detection"] for v in results["videos"].values()
                    if v["detection"].get("status") == "measured"]
    if det_measured:
        results["aggregate"]["detection"] = {
            "mean_precision": round(float(np.mean([d["precision"] for d in det_measured])), 4),
            "mean_recall": round(float(np.mean([d["recall"] for d in det_measured])), 4),
            "mean_ap50": round(float(np.mean([d["ap50"] for d in det_measured])), 4),
            "n_videos": len(det_measured),
        }

    trk_measured = [v["tracking"] for v in results["videos"].values()
                    if v["tracking"].get("status") == "measured"]
    if trk_measured:
        results["aggregate"]["tracking"] = {
            "mean_coverage_pct": round(float(np.mean([t["coverage_pct"] for t in trk_measured])), 1),
            "mean_id_switches": round(float(np.mean([t["id_switches"] for t in trk_measured])), 2),
            "n_videos": len(trk_measured),
        }

    if all_release:
        errors = [r["absolute_error"] for r in all_release]
        results["aggregate"]["release_frame"] = {
            "mean_error": round(float(np.mean(errors)), 2),
            "median_error": round(float(np.median(errors)), 2),
            "max_error": int(np.max(errors)),
        }

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="PaceAI Evaluation Framework")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Evaluate a specific video (default: all)")
    parser.add_argument("--predict", action="store_true",
                        help="Run the real CV pipeline on each annotated video and "
                             "cache predictions before evaluating (slow).")
    args = parser.parse_args()

    if args.predict:
        ds = load_dataset()
        if not ds.videos:
            print("No videos in evaluation/metadata.csv -- nothing to predict.")
            return
        ids = [args.video_id] if args.video_id else [v.video_id for v in ds.videos]
        sel = [v for v in ds.videos if v.video_id in ids]
        print(f"Running pipeline predictions on {len(sel)} video(s)...")
        for vid in sel:
            print(f"  -> {vid.video_id} ...")
            predict_video(vid)
        print("Predictions cached in evaluation/predictions/")

    results = run_evaluation(args.video_id)

    print("\n" + "=" * 60)
    print("PACEAI EVALUATION RESULTS")
    print("=" * 60)

    if results["status"] == "NO_DATA":
        print(f"\n{results['message']}")
        print("\nAll metrics: NOT MEASURED")
        return

    print(f"\nVideos evaluated: {results['n_videos']}")

    for vid_id, v_result in results.get("videos", {}).items():
        print(f"\n--- {vid_id} ---")
        print(f"  Ground truth annotations: {v_result['ground_truth_count']}")
        det = v_result["detection"]
        print(f"  Detection: P={det.get('precision', 'N/A')}, "
              f"R={det.get('recall', 'N/A')}, mAP50={det.get('ap50', 'N/A')} "
              f"[{det['status']}]")
        trk = v_result["tracking"]
        print(f"  Tracking: coverage={trk.get('coverage_pct', 'N/A')}%, "
              f"ID switches={trk.get('id_switches', 'N/A')} [{trk['status']}]")
        rel = v_result["release_frame"]
        print(f"  Release frame: error={rel.get('absolute_error', 'N/A')} frames "
              f"[{rel['status']}]")
        pose = v_result["pose"]
        print(f"  Pose: mean_error={pose.get('mean_pixel_error', 'N/A')} px "
              f"[{pose['status']}]")
        wp = v_result["wrist_proxy"]
        print(f"  Wrist proxy: {wp.get('quality_level', 'N/A')} [{wp['status']}]")
        reels = v_result["reels"]
        print(f"  Reels: max_displacement={reels.get('max_center_displacement_px', 'N/A')} px "
              f"[{reels['status']}]")

    if "aggregate" in results:
        print("\n--- AGGREGATE ---")
        for metric, vals in results["aggregate"].items():
            print(f"  {metric}: {vals}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
