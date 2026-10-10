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
    n_people: int = 1  # how many simultaneous people the pose model saw in this frame
    #: How the analysed subject was chosen: "subject_overlap" (the caller's locked
    #: region decided it), "largest_area_fallback" (no pose overlapped the locked
    #: region, so the legacy heuristic decided), or "largest_area" (no region was
    #: supplied). Surfaced so the pipeline can refuse to score a fallback frame.
    subject_selection: str = "largest_area"


def _landmark_bbox(pose_landmarks) -> tuple:
    """Axis-aligned bbox of one person's landmarks, in normalized coords."""
    xs = [p.x for p in pose_landmarks]
    ys = [p.y for p in pose_landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _largest_area_index(pose_landmarks) -> int:
    best_idx, best_area = 0, -1.0
    for i, lm in enumerate(pose_landmarks):
        x1, y1, x2, y2 = _landmark_bbox(lm)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_idx, best_area = i, area
    return best_idx


def _select_subject(pose_landmarks, subject_bbox=None,
                    frame_size=None) -> tuple:
    """Choose which detected person to analyse.

    This is the authoritative subject decision. It is *not* allowed to contradict
    the caller's locked identity: when ``subject_bbox`` (the bowler's region, in
    pixel coordinates of the frame being analysed) is supplied, the person whose
    pose actually overlaps that region wins.

    Selecting by landmark area alone was the second, independent identity decision
    in this pipeline and the direct cause of the measured failure: on side-on
    broadcast footage the nearest, largest person is the batsman, not a small
    distant bowler. Area is now only a fallback for frames where no pose overlaps
    the locked region, and that fallback is reported so the caller can refuse to
    score it.

    Returns ``(index, method)``.
    """
    if not pose_landmarks:
        return 0, "none"

    if subject_bbox is not None and frame_size is not None:
        fh, fw = frame_size
        if fw > 0 and fh > 0:
            sx1, sy1, sx2, sy2 = subject_bbox
            region = (sx1 / fw, sy1 / fh, sx2 / fw, sy2 / fh)
            best_idx, best_iou = -1, 0.0
            for i, lm in enumerate(pose_landmarks):
                overlap = _iou(_landmark_bbox(lm), region)
                if overlap > best_iou:
                    best_idx, best_iou = i, overlap
            if best_idx >= 0 and best_iou > 0.0:
                return best_idx, "subject_overlap"
            return _largest_area_index(pose_landmarks), "largest_area_fallback"

    return _largest_area_index(pose_landmarks), "largest_area"


def _primary_person(pose_landmarks) -> int:
    """Index of the person to analyse, by largest landmark bounding box.

    Retained for backward compatibility. Prefer :func:`_select_subject`, which
    honours a caller-supplied identity lock; this function has no notion of one
    and must not be used on full broadcast frames.
    """
    return _largest_area_index(pose_landmarks)


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
            num_poses=config.POSE_MAX_PEOPLE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int,
                       timestamp_sec: float,
                       subject_bbox=None) -> Optional[PoseFrame]:
        """Estimate pose for ``frame_bgr``.

        ``subject_bbox`` is the caller's locked identity region as
        ``(x1, y1, x2, y2)`` in *pixel coordinates of this frame*. When given, the
        analysed person is the one overlapping that region, so pose extraction
        cannot silently disagree with the bowler lock. Pass ``None`` only when no
        identity has been established -- and note the result is then flagged
        ``largest_area`` and must not be treated as a verified subject.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp_sec * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        n_people = len(result.pose_landmarks)
        frame_size = frame_bgr.shape[:2]
        idx, method = _select_subject(
            result.pose_landmarks, subject_bbox=subject_bbox, frame_size=frame_size)
        lm = result.pose_landmarks[idx]
        landmarks = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])

        world = None
        if result.pose_world_landmarks:
            wlm = result.pose_world_landmarks[idx]
            world = np.array([[p.x, p.y, p.z] for p in wlm])

        return PoseFrame(frame_idx, timestamp_sec, landmarks, world, n_people, method)

    def process_video_frames(self, frames_iter,
                             subject_box_by_frame: Optional[dict] = None) -> List[PoseFrame]:
        """Run pose over ``frames_iter`` yielding ``(frame_idx, timestamp_sec, frame)``.

        ``subject_box_by_frame`` maps ``frame_idx`` to that frame's locked
        identity region in the *same coordinate space as the frame being passed*.
        This matters: cropped frames and full frames need different boxes, and
        supplying the wrong space is how a bowler lock silently degrades into a
        whole-frame search.
        """
        pose_sequence = []
        for frame_idx, ts, frame in frames_iter:
            subject_bbox = None
            if subject_box_by_frame:
                subject_bbox = subject_box_by_frame.get(frame_idx)
            pf = self.process_frame(frame, frame_idx, ts, subject_bbox=subject_bbox)
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
