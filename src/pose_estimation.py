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


def _primary_person(pose_landmarks) -> int:
    """
    Index of the person to analyze. MediaPipe can detect several people in one
    frame (fielders, keeper, umpire); blindly taking [0] is not guaranteed to
    be the bowler. We pick the person with the LARGEST landmark bounding box:
    in a bowler-cropped frame that is the bowler, and in a full scene it is the
    dominant (usually nearest) person.
    """
    best_idx, best_area = 0, -1.0
    for i, lm in enumerate(pose_landmarks):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_idx, best_area = i, area
    return best_idx


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

        idx = _primary_person(result.pose_landmarks)
        lm = result.pose_landmarks[idx]
        landmarks = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])

        world = None
        if result.pose_world_landmarks:
            wlm = result.pose_world_landmarks[idx]
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


# MediaPipe pose bone connections (index pairs into POSE_LANDMARK_NAMES).
# Same topology as the official PoseLandmarker POSE_CONNECTIONS.
_SKELETON_CONNECTIONS = [
    (11, 12),   # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # right arm
    (11, 23), (12, 24),  # shoulders -> hips
    (23, 24),  # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    (0, 11), (0, 12),   # nose -> shoulders
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # face
]
_SKELETON_POINTS = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def draw_skeleton(frame_bgr: np.ndarray, pose_frame: PoseFrame,
                  min_visibility: float = 0.4,
                  bone_color=(0, 204, 255), joint_color=(255, 80, 80),
                  thickness: int = 2) -> np.ndarray:
    """Draw the pose skeleton (bones + joints) onto a BGR frame.

    ``pose_frame.landmarks`` holds normalized (x, y, z, visibility) up to
    (33, 4); landmarks below ``min_visibility`` are skipped so weak/temporary
    joints don't flicker jagged bones. Returns a copy of the frame with the
    overlay applied -- the caller decides where to draw it.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    pts = {}
    for i, name in enumerate(config.POSE_LANDMARK_NAMES):
        if i >= pose_frame.landmarks.shape[0]:
            break
        x, y, z, vis = pose_frame.landmarks[i]
        if vis < min_visibility:
            continue
        pts[i] = (int(round(x * w)), int(round(y * h)))

    for a, b in _SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)

    for i in _SKELETON_POINTS:
        if i in pts:
            px, py = pts[i]
            cv2.circle(img, (px, py), 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return img
