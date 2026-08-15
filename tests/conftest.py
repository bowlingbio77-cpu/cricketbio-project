"""
Shared helpers for the cricket_biomech_ai test-suite.

Builds synthetic 33-landmark pose sequences (world-space meters and the 2D
normalized fallback) so the biomechanical feature pipeline can be tested
without a camera, MediaPipe model, or video file.
"""
import numpy as np
import pytest

from src import config
from src.feature_engineering import L


def make_world_pose(wrist_elev=0.0, elbow_deg=170.0, l_ankle_y=-0.9, r_ankle_y=-0.9,
                    l_ankle_x=-0.08, r_ankle_x=0.08, sway=0.0, hip_y=0.0,
                    trunk_lean_deg=0.0):
    """
    A roughly standing right-arm bowler in world coordinates (meters, y-up).
    `wrist_elev`: bowling wrist height relative to the bowling shoulder (m).
    `elbow_deg`:  angle at the bowling elbow (180 = straight).
    """
    lm = np.zeros((33, 3))
    lean = np.radians(trunk_lean_deg)
    sh_y = hip_y + 0.62
    sh_x_off = -0.22 if not trunk_lean_deg else -0.22 * np.cos(lean)
    sh_top = hip_y + 0.62
    lm[L["left_hip"]] = [-0.18 + sway, hip_y, 0]
    lm[L["right_hip"]] = [0.18 + sway, hip_y, 0]
    lm[L["left_shoulder"]] = [-0.22 + sway, sh_top, 0]
    lm[L["right_shoulder"]] = [0.22 + sway, sh_top, 0]
    lm[L["left_knee"]] = [-0.1 + sway, -0.5, 0]
    lm[L["right_knee"]] = [0.1 + sway, -0.5, 0]
    lm[L["left_ankle"]] = [l_ankle_x + sway, l_ankle_y, 0]
    lm[L["right_ankle"]] = [r_ankle_x + sway, r_ankle_y, 0]
    lm[L["left_foot_index"]] = [l_ankle_x + sway, l_ankle_y - 0.02, 0]
    lm[L["right_foot_index"]] = [r_ankle_x + sway, r_ankle_y - 0.02, 0]
    lm[L["nose"]] = [0.0 + sway, 0.80, 0.05]

    # bowling (right) arm: shoulder -> wrist, elbow placed for the given angle.
    s = lm[L["right_shoulder"]].copy()
    w = s + np.array([0.0, wrist_elev, 0.3])
    d = w - s
    ln = np.linalg.norm(d)
    if ln < 1e-9:
        ln = 1e-9
    u = d / ln
    v = np.array([-u[2], 0.0, u[0]])
    theta = np.radians(elbow_deg)
    off = (ln / 2.0) * np.tan((np.pi - theta) / 2.0)
    e = s + 0.5 * d + off * v
    lm[L["right_elbow"]] = e
    lm[L["right_wrist"]] = w
    lm[L["right_index"]] = w + np.array([0.0, -0.02, 0.1])
    lm[L["left_elbow"]] = lm[L["left_shoulder"]] + np.array([0, 0, -0.3])
    lm[L["left_wrist"]] = lm[L["left_elbow"]] + np.array([0, 0, -0.2])
    lm[L["left_index"]] = lm[L["left_wrist"]] + np.array([0, 0, -0.05])
    return lm


def to_normalized(world_pose, scale=1.0):
    """Project a world-space pose into a normalized-2D image (x,y in [0,1], y down)."""
    lm = np.zeros((33, 4))
    # crude orthographic projection: x right, y down
    y_off = 0.5
    for name in config.POSE_LANDMARK_NAMES:
        i = L[name]
        p = world_pose[i]
        lm[i] = [0.5 + p[0] * scale, y_off - p[1] * scale, 0.0, 1.0]
    return lm


class PoseFrame:
    def __init__(self, idx, ts, landmarks, world_landmarks=None):
        self.frame_idx = idx
        self.timestamp_sec = ts
        self.landmarks = landmarks
        self.world_landmarks = world_landmarks


def make_sequence(spec, use_world=True, scale=1.0, fps=20.0):
    """
    `spec`: list of dicts, one per frame, e.g.
        {"wrist_elev": ..., "elbow_deg": ..., "l_ankle_y": ..., "r_ankle_y": ..., "sway": ...}
    Returns a list of PoseFrame with world landmarks (and normalized 2D).
    """
    seq = []
    for i, kw in enumerate(spec):
        world = make_world_pose(**kw)
        norm = to_normalized(world, scale=scale)
        if use_world:
            seq.append(PoseFrame(i, i / fps, norm, world))
        else:
            seq.append(PoseFrame(i, i / fps, norm, None))
    return seq


def delivery_sequence(scale=1.0, use_world=True, fps=20.0, contact_frame=9, release_frame=15):
    """
    A canonical fast-bowling clip: run-up (bent arm) -> bound -> front-foot
    contact (left ankle plants at `contact_frame`) -> backswing overhead ->
    arm sweeps down and extends fully at `release_frame` -> follow-through.
    """
    spec = []
    t = 0.0
    # run-up (arm relaxed/bent, both feet moving)
    for i in range(6):
        spec.append({"wrist_elev": -0.2, "elbow_deg": 140,
                     "l_ankle_y": -0.9 + 0.02 * (i % 3), "r_ankle_y": -0.9 + 0.01 * i,
                     "sway": 0.02 * i})
    # bound (arm rises into the backswing)
    for i in range(3):
        spec.append({"wrist_elev": 0.3 + 0.2 * i, "elbow_deg": 175,
                     "l_ankle_y": -0.8 - 0.03 * i, "r_ankle_y": -0.7 - 0.05 * i,
                     "sway": 0.3 + 0.05 * i})
    # front-foot contact: left ankle planted, arm at top of backswing
    for i in range(3):
        spec.append({"wrist_elev": 0.8 - 0.2 * i, "elbow_deg": 172,
                     "l_ankle_y": -0.85, "r_ankle_y": -0.5 - 0.1 * i,
                     "sway": 0.5 + 0.05 * i})
    # arm sweeps down; full extension (low flexion) at release_frame
    for i in range(4):
        spec.append({"wrist_elev": 0.15 - 0.25 * i, "elbow_deg": 150 + 8 * i,
                     "l_ankle_y": -0.85, "r_ankle_y": -0.6 + 0.1 * i,
                     "sway": 0.7 + 0.05 * i})
    # follow-through
    for i in range(2):
        spec.append({"wrist_elev": -0.6 - 0.1 * i, "elbow_deg": 120,
                     "l_ankle_y": -0.82, "r_ankle_y": -0.5, "sway": 0.9})
    return make_sequence(spec, use_world=use_world, scale=scale, fps=fps)


@pytest.fixture
def delivery_seq():
    return delivery_sequence()
