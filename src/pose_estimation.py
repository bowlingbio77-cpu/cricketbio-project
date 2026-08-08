"""
Stage 4 & 5: Pose Estimation + 33 Body Landmark Coordinate Extraction

Uses MediaPipe's Tasks API (mediapipe>=0.10) PoseLandmarker, which needs a
`.task` model file. Download once (needs internet):

    wget -O models/pose_landmarker_heavy.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

(use pose_landmarker_lite.task or _full.task for faster/less accurate variants)
"""
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from . import config


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp_sec: float
    landmarks: np.ndarray  # shape (33, 4) -> x, y, z, visibility (x,y normalized 0-1)
    world_landmarks: Optional[np.ndarray] = None  # shape (33, 3) -> metric x,y,z (meters)


class PoseEstimator:
    def __init__(self, model_path: str = config.POSE_MODEL_PATH,
                 min_detection_confidence: float = config.POSE_MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = config.POSE_MIN_TRACKING_CONFIDENCE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe pose model not found at {model_path}. "
                "Download it first (see module docstring) -- requires internet access."
            )
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int,
                       timestamp_sec: float) -> Optional[PoseFrame]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp_sec * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]  # primary detected person
        landmarks = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])

        world = None
        if result.pose_world_landmarks:
            wlm = result.pose_world_landmarks[0]
            world = np.array([[p.x, p.y, p.z] for p in wlm])

        return PoseFrame(frame_idx, timestamp_sec, landmarks, world)

    def process_video_frames(self, frames_iter) -> List[PoseFrame]:
        """frames_iter yields (frame_idx, timestamp_sec, frame_bgr), e.g. from preprocessing.py"""
        pose_sequence = []
        for frame_idx, ts, frame in frames_iter:
            pf = self.process_frame(frame, frame_idx, ts)
            if pf is not None:
                pose_sequence.append(pf)
        return pose_sequence

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def landmark_dict(pose_frame: PoseFrame) -> dict:
    """Convenience: {landmark_name: (x, y, z, visibility)}"""
    return {
        name: tuple(pose_frame.landmarks[i])
        for i, name in enumerate(config.POSE_LANDMARK_NAMES)
    }
