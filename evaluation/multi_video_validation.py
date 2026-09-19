"""
Multi-video validation script for PaceAI pipeline.

Runs detection, tracking, and pose estimation on multiple videos
and aggregates results for statistical analysis.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.detection import BowlerDetector
from src.pose_estimation import PoseEstimator


def validate_single_video(video_path, detector, pose_estimator):
    """Run full pipeline on a single video and return results."""
    video_name = os.path.basename(video_path)
    print(f"\nProcessing: {video_name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    result = {
        "video": video_name,
        "fps": fps,
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "detections": [],
        "poses": [],
        "tracking_ids": set(),
        "errors": [],
    }
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection (every 10th frame for speed)
        if frame_idx % 10 == 0:
            try:
                detections = detector.detect(frame, frame_idx)
                for det in detections:
                    result["detections"].append({
                        "frame": frame_idx,
                        "bbox": det.bbox.tolist() if hasattr(det.bbox, 'tolist') else det.bbox,
                        "confidence": float(det.confidence),
                    })
            except Exception as e:
                result["errors"].append(f"Detection error at frame {frame_idx}: {e}")
        
        # Pose (every 10th frame to save time)
        if frame_idx % 10 == 0:
            try:
                timestamp_sec = frame_idx / fps
                pose_frame = pose_estimator.process_frame(frame, frame_idx, timestamp_sec)
                if pose_frame is not None:
                    result["poses"].append({
                        "frame": frame_idx,
                        "n_people": pose_frame.n_people,
                        "landmarks_shape": pose_frame.landmarks.shape,
                    })
            except Exception as e:
                result["errors"].append(f"Pose error at frame {frame_idx}: {e}")
        
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")
    
    cap.release()
    result["processing_time"] = time.time() - start_time
    result["detection_count"] = len(result["detections"])
    result["pose_count"] = len(result["poses"])
    
    print(f"  Done: {result['detection_count']} detections, {result['pose_count']} poses, {result['processing_time']:.1f}s")
    
    return result


def main():
    dataset_dir = Path(__file__).parent / "single_video_benchmark" / "dataset" / "videos"
    output_path = Path(__file__).parent / "multi_video_results.json"
    
    videos = sorted(dataset_dir.glob("*.mp4"))
    
    if not videos:
        print(f"No videos found in {dataset_dir}")
        return
    
    print(f"Found {len(videos)} videos")
    
    # Initialize detector and pose estimator
    detector = BowlerDetector()
    pose_estimator = PoseEstimator()
    
    all_results = {
        "description": "Multi-video validation results",
        "videos": [],
        "summary": {},
    }
    
    total_detections = 0
    total_poses = 0
    total_time = 0
    
    for video_path in videos:
        result = validate_single_video(video_path, detector, pose_estimator)
        all_results["videos"].append(result)
        
        if "error" not in result:
            total_detections += result["detection_count"]
            total_poses += result["pose_count"]
            total_time += result["processing_time"]
    
    # Summary
    successful_videos = [v for v in all_results["videos"] if "error" not in v]
    all_results["summary"] = {
        "videos_processed": len(successful_videos),
        "total_detections": total_detections,
        "total_poses": total_poses,
        "total_processing_time": round(total_time, 1),
        "avg_detections_per_video": round(total_detections / max(1, len(successful_videos)), 1),
        "avg_poses_per_video": round(total_poses / max(1, len(successful_videos)), 1),
    }
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Videos processed: {all_results['summary']['videos_processed']}")
    print(f"Total detections: {all_results['summary']['total_detections']}")
    print(f"Total poses: {all_results['summary']['total_poses']}")
    print(f"Total time: {all_results['summary']['total_processing_time']}s")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
