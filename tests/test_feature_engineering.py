"""
Tests for src/feature_engineering.py -- the 10 biomechanical features.

Covers the critical fixes:
  - joint angles are computed from view-independent world landmarks (3D)
  - stride length is normalized by body height (size/scale invariant)
  - ground-contact time uses physical/person-relative units (size invariant)
  - release frame is anchored to front-foot contact + below-shoulder extension
  - front/back-foot logic is detected from motion, not assumed from the bowling arm
"""
import numpy as np
import pytest

from src import config
from src import feature_engineering as fe
from tests.conftest import (delivery_sequence, make_sequence, make_world_pose,
                            PoseFrame, to_normalized)


# --------------------------------------------------------------------------- #
# Basic angle computation (3D world landmarks = view independent)
# --------------------------------------------------------------------------- #
def test_elbow_angle_world_3d_is_view_independent():
    # Fully straight right arm -> flexion from straight ~ 0
    lm = make_world_pose(wrist_elev=0.2, elbow_deg=180.0)
    f = fe.compute_frame_features(lm, 0, 0.0, "right")
    assert f.elbow_flexion_deg < 1.0

    # Bent arm -> larger flexion
    bent = make_world_pose(wrist_elev=0.2, elbow_deg=130.0)
    f2 = fe.compute_frame_features(bent, 0, 0.0, "right")
    assert f2.elbow_flexion_deg > f.elbow_flexion_deg


def test_world_landmarks_preferred_over_normalized():
    """A PoseFrame with world landmarks must use them (3D), not the 2D fallback."""
    lm = make_world_pose(wrist_elev=0.2, elbow_deg=180.0)
    pf = PoseFrame(0, 0.0, to_normalized(lm), lm)
    f = fe.compute_frame_features(pf, 0, 0.0, "right")
    assert f.elbow_flexion_deg < 1.0


def test_shoulder_rotation_and_trunk_lean_2d_fallback_ranges():
    lm = make_world_pose(wrist_elev=0.2, elbow_deg=170.0)
    norm = to_normalized(lm)
    f = fe.compute_frame_features(norm[:, :2], 0, 0.0, "right")
    assert 0.0 <= f.shoulder_rotation_deg <= 90.0
    assert 0.0 <= f.trunk_lean_deg <= 90.0
    assert 0.0 <= f.knee_flexion_deg <= 90.0


# --------------------------------------------------------------------------- #
# Delivery-phase detection (release anchored to contact, not backswing top)
# --------------------------------------------------------------------------- #
def test_release_frame_after_backswing_peak():
    seq = delivery_sequence()
    fv, diag = fe.analyze_delivery(seq, bowling_arm="right")
    assert diag["reliable"] is True
    # The global argmin of elbow extension would be the backswing top (~frame 8);
    # the corrected detector must land after front-foot contact at release (~15).
    assert diag["front_foot_contact_frame"] == 9
    assert diag["release_frame_idx"] == 15
    assert diag["release_frame_idx"] > diag["front_foot_contact_frame"]


def test_front_leg_is_planted_leg_not_assumed():
    """The planted (stationary) ankle is detected as the front leg, even when it
    is NOT the conventionally assumed non-bowling-arm leg. Here the RIGHT ankle
    is planted through the delivery; the left ankle keeps moving."""
    spec = [
        {"wrist_elev": -0.2, "elbow_deg": 140, "l_ankle_y": -0.9, "r_ankle_y": -0.9, "sway": 0.0},
        {"wrist_elev": 0.6, "elbow_deg": 172, "l_ankle_y": -0.75, "r_ankle_y": -0.85, "sway": 0.0},
        {"wrist_elev": 0.4, "elbow_deg": 172, "l_ankle_y": -0.80, "r_ankle_y": -0.85, "sway": 0.0},
        {"wrist_elev": 0.1, "elbow_deg": 178, "l_ankle_y": -0.85, "r_ankle_y": -0.85, "sway": 0.0},
        {"wrist_elev": -0.4, "elbow_deg": 120, "l_ankle_y": -0.70, "r_ankle_y": -0.85, "sway": 0.0},
    ]
    seq = make_sequence(spec)
    assert fe.detect_front_leg(seq, 2, "right") == "right"


# --------------------------------------------------------------------------- #
# Stride-length normalization (size/scale invariant)
# --------------------------------------------------------------------------- #
def test_stride_length_normalized_by_body_height_is_scale_invariant():
    big = delivery_sequence(scale=2.0)
    small = delivery_sequence(scale=1.0)
    fv_big, diag_big = fe.analyze_delivery(big, bowling_arm="right")
    fv_small, diag_small = fe.analyze_delivery(small, bowling_arm="right")
    assert abs(fv_big["stride_length_norm"] - fv_small["stride_length_norm"]) < 0.01
    assert 0.05 < fv_big["stride_length_norm"] < 0.5  # plausible, not 0.0x-from-3x-overcount


def test_stride_length_world_and_2d_are_both_scale_invariant():
    for use_world in (True, False):
        big = delivery_sequence(scale=2.0, use_world=use_world)
        small = delivery_sequence(scale=1.0, use_world=use_world)
        fv_big, _ = fe.analyze_delivery(big, bowling_arm="right")
        fv_small, _ = fe.analyze_delivery(small, bowling_arm="right")
        assert abs(fv_big["stride_length_norm"] - fv_small["stride_length_norm"]) < 0.01


def test_stride_length_not_degenerate_zero():
    fv, diag = fe.analyze_delivery(delivery_sequence(), bowling_arm="right")
    assert fv["stride_length_norm"] > 0.0


# --------------------------------------------------------------------------- #
# Ground-contact time (person-relative units, not image pixels/sec)
# --------------------------------------------------------------------------- #
def test_ground_contact_time_scale_invariant():
    big = delivery_sequence(scale=2.0)
    small = delivery_sequence(scale=1.0)
    fv_big, diag_big = fe.analyze_delivery(big, bowling_arm="right")
    fv_small, diag_small = fe.analyze_delivery(small, bowling_arm="right")
    assert abs(fv_big["ground_contact_time_s"] - fv_small["ground_contact_time_s"]) < 0.05


def test_ground_contact_time_in_seconds_range():
    fv, _ = fe.analyze_delivery(delivery_sequence(), bowling_arm="right")
    assert 0.0 < fv["ground_contact_time_s"] <= 0.5  # a stance lasts a fraction of a second


# --------------------------------------------------------------------------- #
# build_feature_vector returns the full 10-feature vector in FEATURE_NAMES order
# --------------------------------------------------------------------------- #
def test_feature_vector_has_all_features():
    fv, diag = fe.analyze_delivery(delivery_sequence(), bowling_arm="right")
    assert set(fv.keys()) == set(config.FEATURE_NAMES)
    for name in config.FEATURE_NAMES:
        assert fv[name] is not None


def test_build_feature_vector_wrapper():
    fv = fe.build_feature_vector(delivery_sequence(), bowling_arm="right")
    assert set(fv.keys()) == set(config.FEATURE_NAMES)


def test_left_arm_bowler_uses_left_landmarks():
    """bowling_arm='left' must select the subject's left arm and still work."""
    seq = delivery_sequence()
    fv, diag = fe.analyze_delivery(seq, bowling_arm="left")
    assert diag["bowling_arm"] == "left"
    assert fv["elbow_flexion_deg"] is not None


def test_too_few_frames_raises():
    with pytest.raises(ValueError):
        fe.analyze_delivery(delivery_sequence()[:2], bowling_arm="right")
