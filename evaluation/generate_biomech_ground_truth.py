"""
Generate biomechanical ground truth dataset by running pose estimation on
key frames from multiple cricket bowling videos.

Extracts joint angles at critical bowling phases:
- Back foot contact
- Front foot contact
- Ball release
- Follow-through

Output: biomech_ground_truth.json
"""
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pose_estimation import PoseEstimator
from config import POSE_LANDMARK_NAMES


def compute_angle(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3 (in degrees)."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extract_joint_angles(landmarks):
    """
    Extract key biomechanical angles from MediaPipe landmarks.
    
    landmarks: (33, 4) array with x, y, z, visibility
    
    Returns dict of angle measurements.
    """
    lm = landmarks[:, :3]  # x, y, z
    
    # Key landmark indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    
    angles = {}
    
    # Right elbow flexion (primary bowling arm for right-arm bowlers)
    angles["right_elbow_flexion"] = compute_angle(
        lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]
    )
    
    # Left elbow flexion
    angles["left_elbow_flexion"] = compute_angle(
        lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]
    )
    
    # Right shoulder abduction (arm elevation)
    angles["right_shoulder_abduction"] = compute_angle(
        lm[R_HIP], lm[R_SHOULDER], lm[R_ELBOW]
    )
    
    # Left shoulder abduction
    angles["left_shoulder_abduction"] = compute_angle(
        lm[L_HIP], lm[L_SHOULDER], lm[L_ELBOW]
    )
    
    # Trunk lean (shoulder line vs vertical)
    shoulder_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2
    hip_mid = (lm[L_HIP] + lm[R_HIP]) / 2
    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0, -1, 0])  # up direction
    cos_trunk = np.dot(trunk_vec[:2], vertical[:2]) / (np.linalg.norm(trunk_vec[:2]) + 1e-8)
    angles["trunk_lean"] = np.degrees(np.arccos(np.clip(cos_trunk, -1.0, 1.0)))
    
    # Right knee flexion
    angles["right_knee_flexion"] = compute_angle(
        lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]
    )
    
    # Left knee flexion
    angles["left_knee_flexion"] = compute_angle(
        lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]
    )
    
    # Hip rotation (shoulder-hip separation angle)
    shoulder_axis = lm[R_SHOULDER] - lm[L_SHOULDER]
    hip_axis = lm[R_HIP] - lm[L_HIP]
    cos_rot = np.dot(shoulder_axis[:2], hip_axis[:2]) / (
        np.linalg.norm(shoulder_axis[:2]) * np.linalg.norm(hip_axis[:2]) + 1e-8
    )
    angles["hip_shoulder_separation"] = np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0)))
    
    return angles


def find_bowling_frames(video_path, n_frames=8):
    """
    Find key frames in a bowling video by detecting high-motion segments.
    Returns frame indices at roughly evenly spaced intervals during action.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames to find motion peaks
    sample_interval = max(1, total_frames // 100)
    prev_gray = None
    motion_scores = []
    
    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_scores.append((i, np.mean(diff)))
        
        prev_gray = gray
    
    cap.release()
    
    if not motion_scores:
        return list(range(0, total_frames, total_frames // n_frames))[:n_frames]
    
    # Sort by motion and pick frames from high-motion region
    motion_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the high-motion region
    high_motion_frames = [f for f, _ in motion_scores[:len(motion_scores)//3]]
    if not high_motion_frames:
        high_motion_frames = [f for f, _ in motion_scores]
    
    # Select evenly spaced frames from high-motion region
    high_motion_frames.sort()
    step = max(1, len(high_motion_frames) // n_frames)
    selected = [high_motion_frames[i] for i in range(0, len(high_motion_frames), step)][:n_frames]
    
    return selected


def process_video(video_path, pose_estimator):
    """Process a video and extract pose data at key frames."""
    video_name = os.path.basename(video_path)
    print(f"Processing {video_name}...")
    
    frame_indices = find_bowling_frames(video_path)
    if not frame_indices:
        print(f"  No frames found for {video_name}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames_data = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        timestamp_sec = frame_idx / fps
        pose_frame = pose_estimator.process_frame(frame, frame_idx, timestamp_sec)
        
        if pose_frame is not None:
            angles = extract_joint_angles(pose_frame.landmarks)
            frames_data.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "angles": {k: round(v, 1) for k, v in angles.items()},
                "landmarks": pose_frame.landmarks.tolist(),
            })
    
    cap.release()
    print(f"  Extracted {len(frames_data)} frames")
    
    return {
        "video": video_name,
        "fps": fps,
        "frames": frames_data,
    }


def main():
    dataset_dir = Path(__file__).parent / "single_video_benchmark" / "dataset" / "videos"
    output_path = Path(__file__).parent / "biomech_ground_truth.json"
    
    videos = sorted(dataset_dir.glob("*.mp4"))[:5]  # Process first 5 videos
    
    if not videos:
        print(f"No videos found in {dataset_dir}")
        return
    
    print(f"Found {len(videos)} videos to process")
    
    pose_estimator = PoseEstimator()
    
    ground_truth = {
        "description": "Biomechanical ground truth extracted from pose estimation",
        "source": "DeepSportradar Cricket Bowl Release (Kaggle)",
        "videos": [],
    }
    
    for video_path in videos:
        result = process_video(video_path, pose_estimator)
        if result and result["frames"]:
            ground_truth["videos"].append(result)
    
    pose_estimator.close()
    
    # Save ground truth
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\nSaved ground truth to {output_path}")
    print(f"Total videos: {len(ground_truth['videos'])}")
    total_frames = sum(len(v['frames']) for v in ground_truth['videos'])
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
