"""
Stage 6: Biomechanical Feature Engineering Module

Converts a sequence of 33-landmark PoseFrames (from pose_estimation.py) into the
10 biomechanical features used in the diagram:
    shoulder_rotation, elbow_flexion, wrist_angle, hip_rotation, knee_flexion,
    trunk_lean, stride_length, release_angle, angular_velocity, ground_contact_time

View-independence (fixes perspective bias):
    MediaPipe returns *normalized* landmarks (pixel-space, x,y in [0,1]) AND
    *world* landmarks (metric, meters, y-up). All joint angles and length ratios
    here are computed from the world landmarks when available, so they are not
    perspective-distorted by the camera angle. The 2D normalized landmarks are
    only used as a fallback when a caller supplies plain arrays.

Front/back-foot logic (fixes handedness/camera-view bugs):
    The front (planted) leg is detected from motion -- the ankle that goes
    stationary for the longest stretch around the delivery -- rather than being
    assumed to be the non-bowling-arm leg. `bowling_arm` selects MediaPipe's
    anatomical sides (the subject's own right/left), which is correct for any
    camera view; the app additionally records `camera_view` so this is explicit.

This module has no dependency on MediaPipe/OpenCV -- it operates on plain numpy
arrays of shape (33, 3+) so it can be unit-tested with synthetic landmarks.
"""
from dataclasses import dataclass
from typing import List, Sequence, Optional
import numpy as np
from . import config

L = {name: i for i, name in enumerate(config.POSE_LANDMARK_NAMES)}


# --------------------------------------------------------------------------- #
# Coordinate selection
# --------------------------------------------------------------------------- #
def _resolve(frame) -> tuple:
    """Return (landmarks, is_world).

    - PoseFrame with world_landmarks  -> (33, 3) metric meters, is_world=True
    - PoseFrame with only landmarks   -> (33, 2) normalized x,y, is_world=False
    - raw ndarray                     -> as-is; is_world = shape[1] >= 3
    Normalized landmarks are sliced to (x, y) because MediaPipe's normalized z
    is a perspective-relative depth, not a metric coordinate.
    """
    if frame is None:
        return None, False
    if isinstance(frame, np.ndarray):
        arr = np.asarray(frame, dtype=float)
        return arr, arr.ndim >= 2 and arr.shape[1] >= 3
    world = getattr(frame, "world_landmarks", None)
    if world is not None and len(world) == 33:
        return np.asarray(world, dtype=float), True
    lm = getattr(frame, "landmarks", None)
    if lm is not None:
        arr = np.asarray(lm, dtype=float)
        return arr[:, :2], False
    return None, False


def _array_for(frame) -> Optional[np.ndarray]:
    """The resolved landmark array for a PoseFrame-like object (or ndarray)."""
    arr, _ = _resolve(frame)
    return arr


def _is_3d(landmarks: np.ndarray) -> bool:
    return landmarks is not None and landmarks.shape[1] >= 3


def _pt(landmarks: np.ndarray, name: str, horizontal: bool = False) -> np.ndarray:
    """3D point (x,y,z) in world space, or the 2D (x,y) fallback.

    When `horizontal=True` and coords are 3D, returns (x, z) -- the ground-plane
    projection used for stride / rotation measurements.
    """
    v = landmarks[L[name]]
    if horizontal and _is_3d(landmarks):
        return np.array([v[0], v[2]])
    return v[:2] if not _is_3d(landmarks) else v[:3]


def _body_height(landmarks: np.ndarray) -> float:
    """Vertical extent of the person: max_y - min_y over all landmarks.

    In world space this is metric (meters); in normalized space it is a fraction
    of frame height. Used as the size-independent normalization for stride length
    and as the unit for velocity thresholds.
    """
    if landmarks is None or landmarks.shape[0] < 33:
        return 0.0
    ys = landmarks[:, 1]
    h = float(np.max(ys) - np.min(ys))
    return h if np.isfinite(h) else 0.0


# --------------------------------------------------------------------------- #
# Angle helpers (work for 2D or 3D points)
# --------------------------------------------------------------------------- #
def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (at vertex b) in degrees, given 2D or 3D points."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _angle_to_horizontal(v: np.ndarray) -> float:
    """Angle of a vector from the horizontal plane, in degrees.

    3D (world): elevation from the ground plane (0 = horizontal, 90 = up).
    2D (normalized): angle from the image x-axis (matches the legacy fallback).
    """
    if v.ndim == 1 and v.shape[0] >= 3:
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return 0.0
        return float(np.degrees(np.arcsin(np.clip(v[1] / norm, -1.0, 1.0))))
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees, mapped to [0, 180]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _horizontal(vec3: np.ndarray) -> np.ndarray:
    """Project a 3D vector onto the ground plane (x, z)."""
    return np.array([vec3[0], vec3[2]])


# --------------------------------------------------------------------------- #
# Per-frame joint-angle features
# --------------------------------------------------------------------------- #
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


def compute_frame_features(frame, frame_idx: int = 0,
                            timestamp_sec: float = 0.0,
                            bowling_arm: str = "right",
                            front_leg: Optional[str] = None) -> FrameFeatures:
    """
    Compute the per-frame joint-angle features from a PoseFrame (prefers the 3D
    metric world landmarks) or a (33, >=3) landmark array.

    `bowling_arm`: "right" or "left" -- the bowler's own (anatomical) bowling arm.
    `front_leg`:   "left"/"right" if known (used for knee flexion); defaults to
                   the non-bowling-arm leg for backward compatibility.
    """
    landmarks = _array_for(frame)
    if landmarks is None or landmarks.shape[0] < 33:
        raise ValueError("Expected a PoseFrame or a (33, >=3) landmark array")

    three_d = _is_3d(landmarks)
    side = bowling_arm
    other = "left" if side == "right" else "right"
    front = front_leg or other

    shoulder = _pt(landmarks, f"{side}_shoulder")
    elbow = _pt(landmarks, f"{side}_elbow")
    wrist = _pt(landmarks, f"{side}_wrist")
    hip = _pt(landmarks, f"{side}_hip")
    knee = _pt(landmarks, f"{front}_knee")
    ankle = _pt(landmarks, f"{front}_ankle")
    index = _pt(landmarks, f"{side}_index")

    opp_shoulder = _pt(landmarks, f"{other}_shoulder")
    opp_hip = _pt(landmarks, f"{other}_hip")

    shoulder_line = shoulder - opp_shoulder
    hip_line = hip - opp_hip
    trunk = _mid(shoulder, opp_shoulder) - _mid(hip, opp_hip)

    if three_d:
        # Projected onto the ground plane -- the true "X-factor" counter-rotation.
        sh_proj = _horizontal(shoulder_line)
        hip_proj = _horizontal(hip_line)
        shoulder_rotation = _angle_between(sh_proj, hip_proj)
        shoulder_rotation = min(shoulder_rotation, 180 - shoulder_rotation)
        # Pelvic tilt from the horizontal plane (view-independent proxy for hip drive).
        hip_rotation = abs(_angle_to_horizontal(hip_line))
        trunk_lean = _angle_between(trunk, np.array([0.0, 1.0, 0.0]))
    else:
        shoulder_rotation = abs(_angle_to_horizontal(shoulder_line) -
                                _angle_to_horizontal(hip_line))
        shoulder_rotation = min(shoulder_rotation, 360 - shoulder_rotation)
        hip_rotation = abs(_angle_to_horizontal(hip_line))
        trunk_lean = abs(90 - abs(_angle_to_horizontal(trunk)))

    elbow_flexion = _angle_3pt(shoulder, elbow, wrist)
    wrist_angle = _angle_3pt(elbow, wrist, index)
    knee_flexion = _angle_3pt(hip, knee, ankle)

    return FrameFeatures(
        frame_idx=frame_idx, timestamp_sec=timestamp_sec,
        shoulder_rotation_deg=shoulder_rotation,
        elbow_flexion_deg=180.0 - elbow_flexion,  # report as *flexion from straight*
        wrist_angle_deg=wrist_angle,
        hip_rotation_deg=hip_rotation,
        knee_flexion_deg=180.0 - knee_flexion,
        trunk_lean_deg=trunk_lean,
    )


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def compute_sequence_features(pose_sequence: Sequence, bowling_arm: str = "right") -> List[FrameFeatures]:
    """pose_sequence: list of PoseFrame (from pose_estimation.py) OR objects with
    .landmarks (ndarray), .frame_idx, .timestamp_sec attributes."""
    return [
        compute_frame_features(pf, pf.frame_idx, pf.timestamp_sec, bowling_arm)
        for pf in pose_sequence
    ]


# --------------------------------------------------------------------------- #
# Delivery-phase detection
# --------------------------------------------------------------------------- #
def _wrist_elevation(pf, bowling_arm: str) -> float:
    """Bowling wrist height relative to the bowling shoulder (>0 => arm raised).

    World coords (y-up): wrist_y - shoulder_y (meters).
    Normalized coords (y grows down): shoulder_y - wrist_y.
    """
    arr = _array_for(pf)
    if arr is None:
        return 0.0
    wrist = arr[L[f"{bowling_arm}_wrist"]]
    shoulder = arr[L[f"{bowling_arm}_shoulder"]]
    if _is_3d(arr):
        return float(wrist[1] - shoulder[1])
    return float(shoulder[1] - wrist[1])


def _ankle_velocity(prev, cur, side: str, dt: float) -> float:
    """Vertical velocity of one ankle between two frames (meters/s or 1/s)."""
    arr0, arr1 = _array_for(prev), _array_for(cur)
    if arr0 is None or arr1 is None or dt <= 0:
        return float("inf")
    y0 = arr0[L[f"{side}_ankle"]][1]
    y1 = arr1[L[f"{side}_ankle"]][1]
    return abs(y1 - y0) / dt


def _ankle_stationary_windows(pose_sequence: Sequence, side: str,
                               vel_frac: float = 0.02) -> List[tuple]:
    """Runs of frames where the ankle's vertical velocity stays below
    `vel_frac` of the person's body height per second (size-independent)."""
    n = len(pose_sequence)
    if n < 2:
        return []
    windows = []
    start = None
    prev_ts = None
    for i in range(1, n):
        pf = pose_sequence[i]
        dt = pf.timestamp_sec - prev_ts if prev_ts is not None else 0.0
        vel = _ankle_velocity(pose_sequence[i - 1], pf, side, dt)
        height = _body_height(_array_for(pf))
        stationary = vel < vel_frac * max(height, 1e-6)
        if stationary and start is None:
            start = i - 1
        elif not stationary and start is not None:
            windows.append((start, i - 1))
            start = None
        prev_ts = pf.timestamp_sec
    if start is not None:
        windows.append((start, n - 1))
    return windows


def detect_front_leg(pose_sequence: Sequence, release_idx: int,
                     bowling_arm: str = "right") -> str:
    """
    The planted (front) foot is the ankle with the longest stationary stretch
    ending around release -- the front foot lands and stays planted through the
    delivery. Falls back to the non-bowling-arm leg if no signal is available.
    """
    other = "left" if bowling_arm == "right" else "right"
    left_windows = _ankle_stationary_windows(pose_sequence, "left")
    right_windows = _ankle_stationary_windows(pose_sequence, "right")

    def best_span(windows):
        spans = [end - start for start, end in windows]
        return max(spans) if spans else 0

    if best_span(left_windows) > best_span(right_windows):
        return "left"
    if best_span(right_windows) > best_span(left_windows):
        return "right"
    return other  # tie / no signal -> legacy heuristic


def front_foot_contact_frame(pose_sequence: Sequence, front_leg: str,
                             release_idx: int) -> Optional[int]:
    """Start of the front ankle's longest stationary run -- the delivery stance
    (foot planted from landing through release) is typically the longest one."""
    windows = _ankle_stationary_windows(pose_sequence, front_leg)
    if not windows:
        return None
    return int(max(windows, key=lambda w: w[1] - w[0])[0])


def find_release_frame(frame_features: List[FrameFeatures],
                       pose_sequence: Optional[Sequence] = None,
                       bowling_arm: str = "right",
                       lo: int = 0, hi: Optional[int] = None) -> int:
    """
    Heuristic: ball release occurs at peak elbow extension (min elbow_flexion_deg)
    *after* the arm has come down from the top of the backswing, i.e. among frames
    where the bowling wrist is not raised above the shoulder (which filters out
    the backswing top, where the arm is also fully extended). Searches within
    [lo, hi); the caller typically anchors `lo` at front-foot contact so the
    run-up phase can't be mistaken for release. Falls back to the global argmin
    of elbow flexion when no wrist data is available.
    """
    if not frame_features:
        raise ValueError("Empty feature sequence")
    if hi is None:
        hi = len(frame_features)
    lo = max(0, lo)
    hi = min(len(frame_features), hi)
    if lo >= hi:
        return int(np.argmin([f.elbow_flexion_deg for f in frame_features]))

    flexions = np.array([f.elbow_flexion_deg for f in frame_features])

    if pose_sequence is not None and len(pose_sequence) == len(flexions):
        elev = np.array([_wrist_elevation(pf, bowling_arm) for pf in pose_sequence])
        height_ref = max(_body_height(_array_for(pf)) for pf in pose_sequence)
        threshold = 0.10 * max(height_ref, 1e-6) if height_ref > 0 else 0.0
        below_shoulder = [i for i in range(lo, hi) if elev[i] <= threshold]
        if below_shoulder:
            return int(min(below_shoulder, key=lambda i: flexions[i]))
    return lo + int(np.argmin(flexions[lo:hi]))


def analyze_delivery_phases(pose_sequence: Sequence, bowling_arm: str = "right") -> dict:
    """
    Full delivery-phase analysis: detects the front (planted) leg, the front-foot
    contact frame, and the release frame using physical signals (ankle stance +
    elbow extension below shoulder height). Returns diagnostics so callers can
    decide whether the clip contained a clean single delivery.
    """
    if len(pose_sequence) < 3:
        raise ValueError("Need at least 3 pose frames to compute delivery features")

    frame_feats = compute_sequence_features(pose_sequence, bowling_arm)

    # Front leg + stance are detected from motion, independent of release location.
    front_leg = detect_front_leg(pose_sequence, len(pose_sequence) // 2, bowling_arm)
    contact = front_foot_contact_frame(pose_sequence, front_leg, len(pose_sequence) - 1)

    if contact is not None:
        # Release happens shortly after front-foot contact; bound the search window
        # so a straight-arm follow-through isn't mistaken for release.
        release_idx = find_release_frame(frame_feats, pose_sequence, bowling_arm,
                                         lo=contact, hi=min(len(frame_feats), contact + 15))
    else:
        release_idx = find_release_frame(frame_feats, pose_sequence, bowling_arm)

    n = len(frame_feats)
    reliable = True
    reason = None
    if contact is None:
        reliable = False
        reason = "no front-foot stance detected -- clip may not contain a full delivery"
    elif release_idx <= 1 or release_idx >= n - 2:
        reliable = False
        reason = "release frame at clip edge -- clip may not be trimmed to one delivery"
    elif release_idx - contact > 20:
        reliable = False
        reason = "unusually long gap between front-foot contact and release"

    return {
        "frame_features": frame_feats,
        "release_frame_idx": release_idx,
        "front_foot_contact_frame": contact,
        "front_leg": front_leg,
        "bowling_arm": bowling_arm,
        "reliable": reliable,
        "reliability_reason": reason,
        "n_frames": n,
    }


# --------------------------------------------------------------------------- #
# Sequence-level (delivery) features
# --------------------------------------------------------------------------- #
def compute_stride_length(pose_sequence: Sequence, release_idx: int,
                          bowling_arm: str = "right", front_leg: Optional[str] = None) -> float:
    """
    Front-foot-contact stride length normalized by the bowler's own body height
    (vertical landmark extent), so it's comparable across camera distances/bowlers.

    Uses the ground-plane distance between the two ankles at front-foot contact
    (the frame the planted ankle first goes stationary), divided by the person's
    vertical extent in the same frame and the same units.
    """
    other = "left" if bowling_arm == "right" else "right"
    front = front_leg or other
    back = "left" if front == "right" else "right"
    contact = front_foot_contact_frame(pose_sequence, front, release_idx)
    idx = contact if contact is not None else max(0, release_idx - 2)

    pf = pose_sequence[idx]
    landmarks = _array_for(pf)
    if landmarks is None:
        return 0.0

    front_ankle = _pt(landmarks, f"{front}_ankle", horizontal=True)
    back_ankle = _pt(landmarks, f"{back}_ankle", horizontal=True)
    stride = float(np.linalg.norm(front_ankle - back_ankle))

    height = _body_height(landmarks)
    if height < 1e-6:
        return 0.0
    return float(stride / height)


def compute_release_angle(pose_sequence: Sequence, release_idx: int,
                          bowling_arm: str = "right") -> float:
    """Angle of the bowling-arm (shoulder->wrist) relative to the horizontal plane
    at the release frame (0 = horizontal arm, 90 = straight overhead)."""
    pf = pose_sequence[release_idx]
    landmarks = _array_for(pf)
    if landmarks is None:
        return 0.0
    shoulder = _pt(landmarks, f"{bowling_arm}_shoulder")
    wrist = _pt(landmarks, f"{bowling_arm}_wrist")
    arm_vec = wrist - shoulder
    return abs(_angle_to_horizontal(arm_vec))


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
                                 front_leg: Optional[str] = None,
                                 vel_frac: float = 0.02) -> float:
    """
    Front-foot ground contact time: duration (seconds) the front ankle's vertical
    position stays near-stationary around the release frame -- approximates the
    stance phase of the delivery stride.

    The threshold is a *fraction of the bowler's own body height per second*
    (physical units in world space, person-relative in the 2D fallback), so it
    does not depend on how big the bowler appears in the frame.
    """
    other = "left" if bowling_arm == "right" else "right"
    front = front_leg or other

    windows = _ankle_stationary_windows(pose_sequence, front, vel_frac)
    if not windows:
        return 0.0

    lo = max(0, release_idx - 6)
    hi = min(len(pose_sequence), release_idx + 3)
    best = 0.0
    for start, end in windows:
        if start > release_idx:
            continue
        overlap = max(0, min(end, hi) - max(start, lo))
        if overlap > best:
            best = float(overlap)
    if best <= 0:
        return 0.0

    pf0, pf1 = pose_sequence[lo], pose_sequence[hi - 1]
    dt = pf1.timestamp_sec - pf0.timestamp_sec
    if dt <= 0:
        return 0.0
    return float(dt * (best / max(1, hi - lo)))


def build_feature_vector(pose_sequence: Sequence, bowling_arm: str = "right",
                         camera_view: str = "behind") -> dict:
    """Legacy wrapper: return just the feature vector dict."""
    vector, _diag = analyze_delivery(pose_sequence, bowling_arm, camera_view)
    return vector


def analyze_delivery(pose_sequence: Sequence, bowling_arm: str = "right",
                     camera_view: str = "behind") -> tuple:
    """
    Top-level entry point: takes a full delivery's pose sequence and returns
    (feature_vector, diagnostics). The vector matches config.FEATURE_NAMES and is
    ready for the ML module; diagnostics describe the delivery-phase detection so
    the caller can warn when the clip was not a clean single delivery.
    """
    phases = analyze_delivery_phases(pose_sequence, bowling_arm)
    release_idx = phases["release_frame_idx"]
    front_leg = phases["front_leg"]
    frame_feats = phases["frame_features"]

    release_frame = compute_frame_features(
        pose_sequence[release_idx], release_idx,
        pose_sequence[release_idx].timestamp_sec, bowling_arm, front_leg=front_leg)

    vector = {
        "shoulder_rotation_deg": release_frame.shoulder_rotation_deg,
        "elbow_flexion_deg": release_frame.elbow_flexion_deg,
        "wrist_angle_deg": release_frame.wrist_angle_deg,
        "hip_rotation_deg": release_frame.hip_rotation_deg,
        "knee_flexion_deg": release_frame.knee_flexion_deg,
        "trunk_lean_deg": release_frame.trunk_lean_deg,
        "stride_length_norm": compute_stride_length(pose_sequence, release_idx, bowling_arm, front_leg),
        "release_angle_deg": compute_release_angle(pose_sequence, release_idx, bowling_arm),
        "angular_velocity_deg_s": compute_angular_velocity(frame_feats, release_idx),
        "ground_contact_time_s": compute_ground_contact_time(pose_sequence, release_idx, bowling_arm, front_leg),
    }

    phases["feature_vector"] = vector
    phases["camera_view"] = camera_view
    return vector, phases
