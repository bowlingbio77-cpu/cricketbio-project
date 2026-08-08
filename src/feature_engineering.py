"""
Stage 6: Biomechanical Feature Engineering Module

Converts a sequence of 33-landmark PoseFrames (from pose_estimation.py) into the
10 biomechanical features used in the diagram:
    shoulder_rotation, elbow_flexion, wrist_angle, hip_rotation, knee_flexion,
    trunk_lean, stride_length, release_angle, angular_velocity, ground_contact_time

This module has no dependency on MediaPipe/OpenCV -- it operates on plain numpy
arrays of shape (33, 3+) so it can be unit-tested with synthetic landmarks.
"""
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from . import config

L = {name: i for i, name in enumerate(config.POSE_LANDMARK_NAMES)}


def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (at vertex b) in degrees, given 2D or 3D points."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _vec_angle_to_horizontal(v: np.ndarray) -> float:
    """Angle of a 2D vector relative to the horizontal axis, in degrees."""
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _xy(landmarks: np.ndarray, name: str) -> np.ndarray:
    return landmarks[L[name], :2]


def _xyz(landmarks: np.ndarray, name: str) -> np.ndarray:
    return landmarks[L[name], :3]


@dataclass
class FrameFeatures:
    frame_idx: int
    timestamp_sec: float
    shoulder_rotation_deg: float
    elbow_flexion_deg: float
    wrist_angle_deg: float
    hip_rotation_deg: float
    knee_flexion_deg: float
    trunk_lean_deg: float

    def to_dict(self):
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "shoulder_rotation_deg": self.shoulder_rotation_deg,
            "elbow_flexion_deg": self.elbow_flexion_deg,
            "wrist_angle_deg": self.wrist_angle_deg,
            "hip_rotation_deg": self.hip_rotation_deg,
            "knee_flexion_deg": self.knee_flexion_deg,
            "trunk_lean_deg": self.trunk_lean_deg,
        }


def compute_frame_features(landmarks: np.ndarray, frame_idx: int = 0,
                            timestamp_sec: float = 0.0,
                            bowling_arm: str = "right") -> FrameFeatures:
    """
    Compute the per-frame joint-angle features from a single (33, >=3) landmark array.
    `bowling_arm`: "right" or "left" -- selects which arm/leg drives release-side metrics.
    """
    side = bowling_arm
    other = "left" if side == "right" else "right"

    shoulder = _xy(landmarks, f"{side}_shoulder")
    elbow = _xy(landmarks, f"{side}_elbow")
    wrist = _xy(landmarks, f"{side}_wrist")
    hip = _xy(landmarks, f"{side}_hip")
    knee = _xy(landmarks, f"{side}_knee")
    ankle = _xy(landmarks, f"{side}_ankle")
    index = _xy(landmarks, f"{side}_index")

    opp_shoulder = _xy(landmarks, f"{other}_shoulder")
    opp_hip = _xy(landmarks, f"{other}_hip")

    # Shoulder rotation: angle of the shoulder line (L-shoulder -> R-shoulder)
    # relative to the hip line -- captures trunk/shoulder counter-rotation through delivery.
    shoulder_line = shoulder - opp_shoulder
    hip_line = hip - opp_hip
    shoulder_rotation = abs(_vec_angle_to_horizontal(shoulder_line) -
                             _vec_angle_to_horizontal(hip_line))
    shoulder_rotation = min(shoulder_rotation, 360 - shoulder_rotation)

    # Elbow flexion: angle at elbow between (shoulder-elbow) and (wrist-elbow).
    # 180 = fully straight (legal per ICC at point of release), smaller = more bend.
    elbow_flexion = _angle_3pt(shoulder, elbow, wrist)

    # Wrist angle: angle at wrist between (elbow-wrist) and (index-wrist) -- cocking angle.
    wrist_angle = _angle_3pt(elbow, wrist, index)

    # Hip rotation: angle of hip line relative to horizontal (proxy for hip drive).
    hip_rotation = abs(_vec_angle_to_horizontal(hip_line))

    # Knee flexion: angle at front-leg knee between hip-knee and ankle-knee.
    knee_flexion = _angle_3pt(hip, knee, ankle)

    # Trunk lean: angle of the trunk (mid-hip -> mid-shoulder) from vertical.
    mid_hip = (hip + opp_hip) / 2
    mid_shoulder = (shoulder + opp_shoulder) / 2
    trunk_vec = mid_shoulder - mid_hip
    trunk_lean = abs(90 - abs(_vec_angle_to_horizontal(trunk_vec)))

    return FrameFeatures(
        frame_idx=frame_idx, timestamp_sec=timestamp_sec,
        shoulder_rotation_deg=shoulder_rotation,
        elbow_flexion_deg=180.0 - elbow_flexion,  # report as *flexion from straight*
        wrist_angle_deg=wrist_angle,
        hip_rotation_deg=hip_rotation,
        knee_flexion_deg=180.0 - knee_flexion,
        trunk_lean_deg=trunk_lean,
    )


def compute_sequence_features(pose_sequence: Sequence, bowling_arm: str = "right") -> List[FrameFeatures]:
    """pose_sequence: list of PoseFrame (from pose_estimation.py) OR objects with
    .landmarks (ndarray), .frame_idx, .timestamp_sec attributes."""
    return [
        compute_frame_features(pf.landmarks, pf.frame_idx, pf.timestamp_sec, bowling_arm)
        for pf in pose_sequence
    ]


def find_release_frame(frame_features: List[FrameFeatures]) -> int:
    """
    Heuristic: ball release occurs at peak elbow extension (min elbow_flexion_deg)
    just after peak shoulder rotation in the delivery stride. Returns index into
    frame_features (not the raw video frame_idx).
    """
    if not frame_features:
        raise ValueError("Empty feature sequence")
    flexions = [f.elbow_flexion_deg for f in frame_features]
    return int(np.argmin(flexions))


def compute_stride_length(pose_sequence: Sequence, release_idx: int,
                           bowling_arm: str = "right") -> float:
    """
    Front-foot-contact stride length, normalized by the bowler's own height
    (shoulder-to-ankle at release), so it's comparable across camera distances/bowlers.
    Measured as horizontal distance between back-foot and front-foot ankles at
    front-foot-contact (approximated as the frame just before release).
    """
    other = "left" if bowling_arm == "right" else "right"
    contact_idx = max(0, release_idx - 2)
    pf = pose_sequence[contact_idx]
    front_ankle = _xy(pf.landmarks, f"{other}_ankle")   # front (non-bowling-arm side) leg lands first
    back_ankle = _xy(pf.landmarks, f"{bowling_arm}_ankle")
    stride_px = np.linalg.norm(front_ankle - back_ankle)

    shoulder = _xy(pf.landmarks, f"{bowling_arm}_shoulder")
    ankle_ref = _xy(pf.landmarks, f"{bowling_arm}_ankle")
    height_px = np.linalg.norm(shoulder - ankle_ref) * 4.0  # rough shoulder-to-ankle -> full height scale
    if height_px < 1e-6:
        return 0.0
    return float(stride_px / height_px)


def compute_release_angle(pose_sequence: Sequence, release_idx: int,
                           bowling_arm: str = "right") -> float:
    """Angle of the bowling-arm (shoulder->wrist) relative to vertical at release frame."""
    pf = pose_sequence[release_idx]
    shoulder = _xy(pf.landmarks, f"{bowling_arm}_shoulder")
    wrist = _xy(pf.landmarks, f"{bowling_arm}_wrist")
    arm_vec = wrist - shoulder
    angle_from_horizontal = _vec_angle_to_horizontal(arm_vec)
    return float(abs(angle_from_horizontal))  # 0 = horizontal arm, 90 = straight overhead


def compute_angular_velocity(frame_features: List[FrameFeatures], release_idx: int,
                              window: int = 3) -> float:
    """
    Peak shoulder angular velocity (deg/s) around release, via finite differences
    of shoulder_rotation_deg over timestamp_sec.
    """
    lo = max(0, release_idx - window)
    hi = min(len(frame_features), release_idx + window + 1)
    seg = frame_features[lo:hi]
    if len(seg) < 2:
        return 0.0
    velocities = []
    for i in range(1, len(seg)):
        dt = seg[i].timestamp_sec - seg[i - 1].timestamp_sec
        if dt <= 0:
            continue
        dtheta = seg[i].shoulder_rotation_deg - seg[i - 1].shoulder_rotation_deg
        velocities.append(abs(dtheta / dt))
    return float(max(velocities)) if velocities else 0.0


def compute_ground_contact_time(pose_sequence: Sequence, release_idx: int,
                                 bowling_arm: str = "right",
                                 ankle_velocity_threshold: float = 0.01) -> float:
    """
    Front-foot ground contact time: duration (seconds) the front ankle's vertical
    position stays near-stationary (velocity below threshold, normalized coords/sec)
    around the release frame -- approximates stance phase duration.
    """
    other = "left" if bowling_arm == "right" else "right"
    lo = max(0, release_idx - 6)
    hi = min(len(pose_sequence), release_idx + 3)
    seg = pose_sequence[lo:hi]
    if len(seg) < 2:
        return 0.0

    contact_frames = 0
    for i in range(1, len(seg)):
        y0 = _xy(seg[i - 1].landmarks, f"{other}_ankle")[1]
        y1 = _xy(seg[i].landmarks, f"{other}_ankle")[1]
        dt = seg[i].timestamp_sec - seg[i - 1].timestamp_sec
        if dt <= 0:
            continue
        vel = abs(y1 - y0) / dt
        if vel < ankle_velocity_threshold:
            contact_frames += 1

    if contact_frames == 0:
        return 0.0
    total_time = seg[-1].timestamp_sec - seg[0].timestamp_sec
    return float(total_time * (contact_frames / max(1, len(seg) - 1)))


def build_feature_vector(pose_sequence: Sequence, bowling_arm: str = "right") -> dict:
    """
    Top-level entry point: takes a full delivery's pose sequence and returns the
    single feature vector (dict) matching config.FEATURE_NAMES, ready for the ML module.
    """
    if len(pose_sequence) < 3:
        raise ValueError("Need at least 3 pose frames to compute delivery features")

    frame_feats = compute_sequence_features(pose_sequence, bowling_arm)
    release_idx = find_release_frame(frame_feats)
    release_frame = frame_feats[release_idx]

    vector = {
        "shoulder_rotation_deg": release_frame.shoulder_rotation_deg,
        "elbow_flexion_deg": release_frame.elbow_flexion_deg,
        "wrist_angle_deg": release_frame.wrist_angle_deg,
        "hip_rotation_deg": release_frame.hip_rotation_deg,
        "knee_flexion_deg": release_frame.knee_flexion_deg,
        "trunk_lean_deg": release_frame.trunk_lean_deg,
        "stride_length_norm": compute_stride_length(pose_sequence, release_idx, bowling_arm),
        "release_angle_deg": compute_release_angle(pose_sequence, release_idx, bowling_arm),
        "angular_velocity_deg_s": compute_angular_velocity(frame_feats, release_idx),
        "ground_contact_time_s": compute_ground_contact_time(pose_sequence, release_idx, bowling_arm),
    }
    return vector
