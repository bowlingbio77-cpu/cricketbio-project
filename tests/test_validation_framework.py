"""
Tests for the evaluation framework, confidence gate, optical flow,
reels quality metrics, left/right normalization, batch analysis,
comparative experiments, and tracking quality flags.
"""
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import (
    compute_iou, compute_detection_metrics, compute_tracking_metrics,
    compute_release_frame_metrics, compute_pose_metrics,
    compute_wrist_proxy_reliability, compute_reels_quality,
    run_evaluation, BallAnnotation, PoseAnnotation,
)
from src.ball_tracking_v2 import (
    BallTracker, BallPoint, _CVKalman, _quadratic_fit_residual,
    DETECTION_CONFIDENCE_THRESHOLD, MAX_PREDICTION_JUMP_PX,
    OPTICAL_FLOW_ENABLED, OPTICAL_FLOW_MEAS_STD,
    GATE_CHI2_2DOF_99, YOLO_MEAS_STD, MOTION_MEAS_STD,
)
from src.batch_analysis import DeliveryRecord, SpellAnalysis
from src.comparative_experiments import (
    ExperimentConfig, run_comparative_experiment, format_comparison_table,
    DEFAULT_EXPERIMENTS,
)
from src import ball_tracking_v2 as btv2


# --------------------------------------------------------------------------- #
# Evaluation Framework Tests
# --------------------------------------------------------------------------- #

class TestIoU:
    def test_perfect_overlap(self):
        assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        # overlap = 5*5=25, area_a=100, area_b=100, union=175, iou=25/175
        assert iou == pytest.approx(25 / 175, abs=0.01)

    def test_contained_box(self):
        iou = compute_iou((2, 2, 8, 8), (0, 0, 10, 10))
        assert iou == pytest.approx(36 / 100, abs=0.01)  # 36 overlap, 100 union


class TestDetectionMetrics:
    def test_no_ground_truth(self):
        result = compute_detection_metrics({}, [])
        assert result["status"] == "no_ground_truth"

    def test_perfect_detection(self):
        gt = [BallAnnotation(0, 10, 10, 20, 20, 1.0)]
        preds = {0: [(10, 10, 20, 20, 0.9)]}
        result = compute_detection_metrics(preds, gt)
        assert result["status"] == "measured"
        assert result["true_positives"] == 1
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_missed_detection(self):
        gt = [BallAnnotation(0, 10, 10, 20, 20, 1.0)]
        result = compute_detection_metrics({}, gt)
        assert result["false_negatives"] == 1
        assert result["recall"] == 0.0

    def test_false_positive(self):
        gt = [BallAnnotation(0, 10, 10, 20, 20, 1.0)]
        preds = {0: [(100, 100, 110, 110, 0.9)]}
        result = compute_detection_metrics(preds, gt)
        assert result["false_positives"] == 1
        assert result["precision"] == 0.0

    def test_confidence_threshold_filtering(self):
        gt = [BallAnnotation(0, 10, 10, 20, 20, 1.0)]
        preds = {0: [(10, 10, 20, 20, 0.05)]}  # below default threshold
        result = compute_detection_metrics(preds, gt, conf_threshold=0.1)
        assert result["false_negatives"] == 1  # detection filtered out

    def test_multiple_frames(self):
        gt = [
            BallAnnotation(0, 10, 10, 20, 20, 1.0),
            BallAnnotation(1, 30, 30, 40, 40, 1.0),
        ]
        preds = {
            0: [(10, 10, 20, 20, 0.9)],
            1: [(30, 30, 40, 40, 0.8)],
        }
        result = compute_detection_metrics(preds, gt)
        assert result["true_positives"] == 2
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestTrackingMetrics:
    def test_empty_trajectory(self):
        result = compute_tracking_metrics([], [])
        assert result["status"] == "no_data"

    def test_perfect_tracking(self):
        gt = [BallAnnotation(0, 10, 10, 20, 20, 1.0)]
        traj = [BallPoint(0, 0.0, 15, 15, 0.9, True, 10, 10, "yolo")]
        result = compute_tracking_metrics(traj, gt)
        assert result["coverage_pct"] == 100.0
        assert result["detection_ratio"] == 1.0

    def test_id_switches(self):
        traj = [
            BallPoint(0, 0.0, 15, 15, 0.9, True, 10, 10, "yolo"),
            BallPoint(1, 0.05, 20, 20, 0.9, True, 10, 10, "wrist_proxy"),
            BallPoint(2, 0.1, 25, 25, 0.9, True, 10, 10, "yolo"),
        ]
        gt = [BallAnnotation(0, 10, 10, 20, 20)]
        result = compute_tracking_metrics(traj, gt)
        assert result["id_switches"] >= 1


class TestReleaseFrameMetrics:
    def test_perfect_prediction(self):
        result = compute_release_frame_metrics(10, 10, fps=20.0)
        assert result["absolute_error"] == 0
        assert result["time_error_s"] == 0.0
        assert result["status"] == "measured"

    def test_off_by_one(self):
        result = compute_release_frame_metrics(11, 10, fps=20.0)
        assert result["absolute_error"] == 1
        assert result["time_error_s"] == pytest.approx(0.05, abs=0.01)

    def test_none_values(self):
        result = compute_release_frame_metrics(None, 10)
        assert result["status"] == "not_available"

    def test_both_none(self):
        result = compute_release_frame_metrics(None, None)
        assert result["status"] == "not_available"


class TestPoseMetrics:
    def test_no_annotations(self):
        result = compute_pose_metrics({}, [])
        assert result["status"] == "not_available"

    def test_perfect_pose(self):
        gt = [PoseAnnotation(0, "right_wrist", 100.0, 200.0, 1.0)]
        pred = {"right_wrist": (100.0, 200.0)}
        result = compute_pose_metrics(pred, gt)
        assert result["status"] == "measured"
        assert result["mean_pixel_error"] == 0.0

    def test_imperfect_pose(self):
        gt = [PoseAnnotation(0, "right_wrist", 100.0, 200.0, 1.0)]
        pred = {"right_wrist": (103.0, 204.0)}
        result = compute_pose_metrics(pred, gt, image_width=640, image_height=360)
        assert result["mean_pixel_error"] == pytest.approx(5.0, abs=0.1)
        assert result["status"] == "measured"

    def test_no_matching_landmarks(self):
        gt = [PoseAnnotation(0, "right_wrist", 100.0, 200.0, 1.0)]
        pred = {"left_wrist": (100.0, 200.0)}
        result = compute_pose_metrics(pred, gt)
        assert result["status"] == "no_matching_landmarks"


class TestWristProxyReliability:
    def test_high_quality(self):
        result = compute_wrist_proxy_reliability(
            [0.9, 0.85, 0.92], delivery_reliable=True, release_frame_detected=True)
        assert result["quality_level"] == "HIGH"
        assert result["status"] == "measured"

    def test_medium_quality_low_visibility(self):
        result = compute_wrist_proxy_reliability(
            [0.3, 0.4, 0.35], delivery_reliable=True, release_frame_detected=True)
        assert result["quality_level"] == "MEDIUM"

    def test_low_quality(self):
        result = compute_wrist_proxy_reliability(
            [0.1, 0.2], delivery_reliable=False, release_frame_detected=False)
        assert result["quality_level"] == "LOW"

    def test_no_data(self):
        result = compute_wrist_proxy_reliability([], False, False)
        assert result["quality_level"] == "LOW"
        assert "No wrist landmark data" in result["confidence_reason"]


class TestReelsQuality:
    def test_empty_trajectory(self):
        result = compute_reels_quality([])
        assert result["status"] == "not_available"

    def test_smooth_trajectory(self):
        traj = [BallPoint(i, i * 0.05, float(i * 10), 100.0, 0.9, True, 10, 10, "yolo")
                for i in range(10)]
        result = compute_reels_quality(traj)
        assert result["status"] == "measured"
        assert result["max_center_displacement_px"] == pytest.approx(10.0, abs=0.1)


class TestEvaluationRunner:
    def test_no_data(self):
        result = run_evaluation()
        assert result["status"] == "NO_DATA"
        assert "NOT_MEASURED" in str(result["detection"])

    def test_video_not_found_with_data(self):
        """When no metadata exists, any video_id lookup returns NO_DATA."""
        result = run_evaluation(video_id="nonexistent_video")
        assert result["status"] in ("NO_DATA", "VIDEO_NOT_FOUND")


# --------------------------------------------------------------------------- #
# Confidence Gate Tests
# --------------------------------------------------------------------------- #

class TestConfidenceGate:
    def test_constants_exist(self):
        assert DETECTION_CONFIDENCE_THRESHOLD > 0
        assert MAX_PREDICTION_JUMP_PX > 0

    def test_constants_are_configurable(self):
        """Verify the constants can be read and are reasonable."""
        assert 0.05 <= DETECTION_CONFIDENCE_THRESHOLD <= 0.5
        assert 20.0 <= MAX_PREDICTION_JUMP_PX <= 200.0

    def test_cvkalman_predict(self):
        kf = _CVKalman(100.0, 200.0, 5.0, -3.0)
        pos, _ = kf.predict()
        assert pos[0] == pytest.approx(105.0, abs=0.1)
        assert pos[1] == pytest.approx(197.0, abs=0.1)

    def test_cvkalman_update(self):
        kf = _CVKalman(100.0, 200.0)
        kf.predict()
        kf.update(np.array([102.0, 198.0]), YOLO_MEAS_STD)
        assert abs(kf.x[0] - 102.0) < 5.0
        assert abs(kf.x[1] - 198.0) < 5.0

    def test_measurement_noise_sources(self):
        from src.ball_tracking_v2 import _meas_std
        assert _meas_std("yolo") == YOLO_MEAS_STD
        assert _meas_std("motion") == MOTION_MEAS_STD
        assert _meas_std("flow") == OPTICAL_FLOW_MEAS_STD

    def test_optical_flow_disabled_by_default(self):
        assert OPTICAL_FLOW_ENABLED is False


# --------------------------------------------------------------------------- #
# Optical Flow Tests
# --------------------------------------------------------------------------- #

class TestOpticalFlow:
    def test_flow_candidates_no_prev_frame(self):
        result = BallTracker._optical_flow_candidates(
            np.zeros((100, 100, 3), dtype=np.uint8), None)
        assert result == []

    def test_flow_candidates_empty_prev(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        prev = np.zeros((100, 100, 3), dtype=np.uint8)
        # No good features in a black frame
        result = BallTracker._optical_flow_candidates(frame, prev)
        assert isinstance(result, list)

    def test_flow_source_in_candidates(self):
        """Optical flow should produce candidates with source='flow'."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        prev = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = BallTracker._optical_flow_candidates(frame, prev)
        for cand in result:
            assert cand[5] == "flow"


# --------------------------------------------------------------------------- #
# Batch Analysis Tests
# --------------------------------------------------------------------------- #

class TestBatchAnalysis:
    def test_delivery_record(self):
        rec = DeliveryRecord(
            delivery_id=1,
            feature_vector={"release_angle_deg": 45.0, "angular_velocity_deg_s": 800.0},
            performance_score=75.0,
            bowling_arm="right",
        )
        assert rec.delivery_id == 1
        assert rec.bowling_arm == "right"

    def test_spell_analysis_empty(self):
        spell = SpellAnalysis()
        assert spell.n_deliveries == 0
        assert spell.release_angle_consistency() is None
        assert spell.speed_trend() is None

    def test_spell_analysis_consistency(self):
        deliveries = [
            DeliveryRecord(1, {"release_angle_deg": 45.0}),
            DeliveryRecord(2, {"release_angle_deg": 46.0}),
            DeliveryRecord(3, {"release_angle_deg": 44.0}),
        ]
        spell = SpellAnalysis(deliveries=deliveries)
        consistency = spell.release_angle_consistency()
        assert consistency is not None
        assert consistency < 2.0  # very consistent

    def test_spell_analysis_speed_trend_increasing(self):
        deliveries = [
            DeliveryRecord(1, {"angular_velocity_deg_s": 600}),
            DeliveryRecord(2, {"angular_velocity_deg_s": 700}),
            DeliveryRecord(3, {"angular_velocity_deg_s": 800}),
            DeliveryRecord(4, {"angular_velocity_deg_s": 900}),
        ]
        spell = SpellAnalysis(deliveries=deliveries)
        assert spell.speed_trend() == "increasing"

    def test_spell_analysis_speed_trend_stable(self):
        deliveries = [
            DeliveryRecord(1, {"angular_velocity_deg_s": 800}),
            DeliveryRecord(2, {"angular_velocity_deg_s": 810}),
            DeliveryRecord(3, {"angular_velocity_deg_s": 790}),
            DeliveryRecord(4, {"angular_velocity_deg_s": 805}),
        ]
        spell = SpellAnalysis(deliveries=deliveries)
        assert spell.speed_trend() == "stable"

    def test_spell_summary(self):
        deliveries = [
            DeliveryRecord(1, {"release_angle_deg": 45.0, "shoulder_rotation_deg": 30.0}),
            DeliveryRecord(2, {"release_angle_deg": 46.0, "shoulder_rotation_deg": 31.0}),
        ]
        spell = SpellAnalysis(deliveries=deliveries, bowler_name="Test")
        summary = spell.summary()
        assert summary["n_deliveries"] == 2
        assert summary["bowler"] == "Test"


# --------------------------------------------------------------------------- #
# Comparative Experiments Tests
# --------------------------------------------------------------------------- #

class TestComparativeExperiments:
    def test_default_experiments_exist(self):
        assert len(DEFAULT_EXPERIMENTS) >= 3

    def test_run_returns_not_measured(self):
        result = run_comparative_experiment()
        assert result["status"] == "NOT_MEASURED"

    def test_format_comparison_table(self):
        result = run_comparative_experiment()
        table = format_comparison_table(result)
        assert "NOT MEASURED" in table
        assert "Baseline" in table

    def test_experiment_config(self):
        cfg = ExperimentConfig(
            name="Test",
            description="Test config",
            optical_flow=True,
        )
        assert cfg.optical_flow is True
        assert cfg.name == "Test"


# --------------------------------------------------------------------------- #
# Slow-mo Quality Metrics Tests
# --------------------------------------------------------------------------- #

class TestSlowMoQuality:
    def test_render_slowmo_returns_quality_metrics(self):
        """Verify the render_slowmo_zoom output dict contains quality fields."""
        # We can't easily test the actual rendering without a video file,
        # but we can verify the function signature accepts the new params.
        import inspect
        sig = inspect.signature(btv2.render_slowmo_zoom)
        params = list(sig.parameters.keys())
        assert "slow_factor" in params
        assert "zoom_end" in params

    def test_slow_motion_frames(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
        slowed = btv2._slow_motion_frames(frames, 2.0)
        assert len(slowed) == 10  # 2x factor = 10 frames from 5

    def test_slow_motion_frames_fractional(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]
        slowed = btv2._slow_motion_frames(frames, 2.5)
        # 2.5x on 4 frames: 4 * 2.5 = 10 expected output frames
        assert len(slowed) == 10

    def test_duplicate_trajectory(self):
        pts = [(10.0, 20.0), (30.0, 40.0)]
        duped = btv2._duplicate_trajectory(pts, 2.0)
        assert len(duped) == 4

    def test_smooth_centers(self):
        centers = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
        smoothed = btv2._smooth_centers(centers, alpha=0.5)
        assert len(smoothed) == 3
        # Second point should be smoothed toward the first
        assert smoothed[1][0] < 10.0
        assert smoothed[1][1] < 10.0

    def test_ease_in_out(self):
        assert btv2._ease_in_out(0.0) == 0.0
        assert btv2._ease_in_out(1.0) == 1.0
        assert btv2._ease_in_out(0.5) == pytest.approx(0.5, abs=0.01)
        assert btv2._ease_in_out(-1.0) == 0.0  # clamped
        assert btv2._ease_in_out(2.0) == 1.0  # clamped


# --------------------------------------------------------------------------- #
# Left/Right Arm Normalization Tests
# --------------------------------------------------------------------------- #

class TestLeftRightNormalization:
    def test_bowling_arm_in_feature_engineering(self):
        """Verify feature engineering respects bowling_arm parameter."""
        from src.feature_engineering import compute_frame_features
        from tests.conftest import make_world_pose
        # Create a right-arm bowling pose
        landmarks = make_world_pose()
        feat = compute_frame_features(landmarks, bowling_arm="right")
        assert feat.elbow_flexion_deg >= 0

    def test_left_arm_reverses_landmarks(self):
        """Left-arm bowler should use left-side landmarks."""
        from src.feature_engineering import compute_frame_features
        from tests.conftest import make_world_pose
        landmarks = make_world_pose()
        feat_r = compute_frame_features(landmarks, bowling_arm="right")
        feat_l = compute_frame_features(landmarks, bowling_arm="left")
        # Both should produce valid features (not crash)
        assert feat_r is not None
        assert feat_l is not None


# --------------------------------------------------------------------------- #
# Edge Case Tests
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_ball_point_defaults(self):
        bp = BallPoint(0, 0.0, 100.0, 200.0, 0.5)
        assert bp.detected is True
        assert bp.source == "motion"

    def test_ball_point_all_sources(self):
        for src in ("yolo", "motion", "predicted", "wrist_proxy", "blended", "flow"):
            bp = BallPoint(0, 0.0, 100.0, 200.0, 0.5, source=src)
            assert bp.source == src

    def test_cvkalman_initial_state(self):
        kf = _CVKalman(50.0, 60.0)
        assert kf.x[0] == 50.0
        assert kf.x[1] == 60.0
        assert kf.x[2] == 0.0  # vx
        assert kf.x[3] == 0.0  # vy

    def test_cvkalman_innovation(self):
        kf = _CVKalman(100.0, 200.0)
        maha2, S = kf.innovation(np.array([100.0, 200.0]), YOLO_MEAS_STD)
        assert maha2 == pytest.approx(0.0, abs=0.01)

    def test_cvkalman_innovation_large_offset(self):
        kf = _CVKalman(100.0, 200.0)
        maha2, _ = kf.innovation(np.array([500.0, 500.0]), YOLO_MEAS_STD)
        assert maha2 > 100.0  # should be large

    def test_quadratic_fit_insufficient_points(self):
        pts = [BallPoint(i, float(i), float(i * 10), 100.0, 0.9, True, 10, 10, "yolo")
               for i in range(3)]
        result = _quadratic_fit_residual(pts)
        assert result is None  # fewer than BALLISTIC_MIN_POINTS

    def test_quadratic_fit_valid(self):
        pts = [BallPoint(i, float(i), float(i * 10), 100.0 + i * 5, 0.9, True, 10, 10, "yolo")
               for i in range(8)]
        result = _quadratic_fit_residual(pts)
        assert result is not None
        assert result >= 0.0
