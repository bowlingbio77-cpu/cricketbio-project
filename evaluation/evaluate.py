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
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(EVAL_DIR, "videos")
ANNOTATIONS_DIR = os.path.join(EVAL_DIR, "annotations")


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
                "confidence_reason": "No wrist landmark data available"}

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

    all_detection = []
    all_tracking = []
    all_release = []

    for vid in videos_to_eval:
        v_result = {}

        gt = ds.ball_annotations.get(vid.video_id, [])
        v_result["ground_truth_count"] = len(gt)

        v_result["detection"] = compute_detection_metrics({}, gt)
        all_detection.append(v_result["detection"])

        v_result["tracking"] = compute_tracking_metrics([], gt, vid.fps)
        all_tracking.append(v_result["tracking"])

        v_result["release_frame"] = compute_release_frame_metrics(
            None, vid.release_frame_annotated, vid.fps)
        if v_result["release_frame"]["status"] == "measured":
            all_release.append(v_result["release_frame"])

        pose_gt = ds.pose_annotations.get(vid.video_id, [])
        v_result["pose"] = compute_pose_metrics({}, pose_gt, vid.width, vid.height)
        v_result["wrist_proxy"] = compute_wrist_proxy_reliability([], False, False)
        v_result["reels"] = compute_reels_quality([])

        results["videos"][vid.video_id] = v_result

    if all_detection:
        measured = [d for d in all_detection if d["status"] == "measured"]
        if measured:
            results["aggregate"]["detection"] = {
                "mean_precision": round(float(np.mean([d["precision"] for d in measured])), 4),
                "mean_recall": round(float(np.mean([d["recall"] for d in measured])), 4),
                "mean_ap50": round(float(np.mean([d["ap50"] for d in measured])), 4),
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
    args = parser.parse_args()

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
