"""
Tests for the P1 independent ground-truth validation pipeline:
annotation-CSV parsing, missing/invalid values, agreement statistics,
insufficient-data handling, Bland-Altman, release-frame error + FPS
conversion, and ground-truth/prediction separation.
"""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import annotator as annot
from evaluation.ground_truth_validation import (
    GT_COLUMNS, CONTINUOUS_FEATURES, FRAME_EVENTS, EXCLUDED_FEATURES,
    MIN_SAMPLES_DESCRIPTIVE, MIN_SAMPLES_LOA, MIN_SAMPLES_CORRELATION,
    MIN_SAMPLES_ICC,
    parse_ground_truth_csv, load_prediction, prediction_frame,
    prediction_feature, reference_rows, paired_samples,
    icc_two_way_random_single, bland_altman, compute_feature_agreement,
    compute_frame_event_metrics, run_ground_truth_validation,
    build_predictions_from_runs,
)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")


# --------------------------------------------------------------------------- #
# Annotation CSV parsing
# --------------------------------------------------------------------------- #

class TestGTParse:
    def test_empty_file_returns_no_records(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
        records, errors = parse_ground_truth_csv(path)
        assert records == []
        assert errors == []

    def test_valid_row(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        _write_csv(path, GT_COLUMNS, [
            ["clip1", 1, "A", "1.1", "2026-09-11", 15, 11, 30.0, 165.0, 25.0,
             120.0, 5.0, "", "good", "good", "good", "good", 5, ""],
        ])
        records, errors = parse_ground_truth_csv(path)
        assert len(records) == 1
        r = records[0]
        assert r["video_id"] == "clip1"
        assert r["release_frame"] == 15
        assert r["front_contact_frame"] == 11
        assert r["elbow_flexion_deg"] == pytest.approx(165.0)
        assert r["annotation_confidence"] == "5"
        assert errors == []

    def test_missing_values_are_none(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        _write_csv(path, GT_COLUMNS, [
            ["clip1", 1, "A", "1.1", "", "", "", "", "", "", "", "",
             "", "", "", "", 3, "occluded elbow"],
        ])
        records, errors = parse_ground_truth_csv(path)
        assert len(records) == 1
        r = records[0]
        assert r["release_frame"] is None
        assert r["elbow_flexion_deg"] is None
        assert r["trunk_lean_deg"] is None
        # missing is never coerced to 0 -> would be a fabricated value
        assert r["elbow_flexion_deg"] != 0.0

    def test_invalid_frame_number_flagged_as_missing(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        _write_csv(path, GT_COLUMNS, [
            ["clip1", 1, "A", "1.1", "", "-3", "abc", 30.0, 165.0, 25.0,
             120.0, 5.0, "", "good", "good", "good", "good", 5, "bad frames"],
        ])
        records, errors = parse_ground_truth_csv(path)
        assert len(records) == 1
        r = records[0]
        # invalid frame numbers are dropped, not clamped to 0
        assert r["release_frame"] is None
        assert r["front_contact_frame"] is None

    def test_row_without_video_id_skipped(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        _write_csv(path, GT_COLUMNS, [
            ["", 1, "A", "1.1", "", 15, 11, 30.0, 165.0, 25.0, 120.0, 5.0, "",
             "good", "good", "good", "good", 5, ""],
        ])
        records, errors = parse_ground_truth_csv(path)
        assert records == []
        assert any("video_id" in e for e in errors)


# --------------------------------------------------------------------------- #
# Ground-truth / prediction separation
# --------------------------------------------------------------------------- #

class TestSeparation:
    def test_paired_samples_empty_when_no_gt(self):
        assert paired_samples([], {}, "elbow_flexion_deg") == []

    def test_paired_samples_skips_missing_prediction(self):
        records = [{
            "video_id": "clip1", "delivery_id": "1", "annotator_id": "A",
            "elbow_flexion_deg": 165.0, "release_frame": 15,
            "front_contact_frame": 11,
        }]
        samples = paired_samples(records, {"clip1": {"biomechanics": {}}},
                                 "elbow_flexion_deg")
        assert samples == []  # prediction missing -> nothing paired, no imputation

    def test_paired_samples_uses_both_sides(self):
        records = [{
            "video_id": "clip1", "delivery_id": "1", "annotator_id": "A",
            "elbow_flexion_deg": 165.0,
        }]
        pred = {"biomechanics": {"elbow_flexion_deg": 170.0}}
        samples = paired_samples(records, {"clip1": pred}, "elbow_flexion_deg")
        assert len(samples) == 1
        assert samples[0][1] == pytest.approx(165.0)
        assert samples[0][2] == pytest.approx(170.0)

    def test_release_frame_alias_accepts_release_idx(self):
        pred = {"release_idx": 12}
        assert prediction_frame(pred, "release_frame") == 12

    def test_release_frame_accepts_biomechanics_alias(self):
        pred = {"biomechanics": {"front_foot_contact_frame": 11}}
        assert prediction_frame(pred, "front_contact_frame") == 11

    def test_adjudicator_wins_over_annotators(self):
        rows = [
            {"video_id": "v", "delivery_id": "1", "annotator_id": "A",
             "elbow_flexion_deg": 160.0},
            {"video_id": "v", "delivery_id": "1", "annotator_id": "B",
             "elbow_flexion_deg": 170.0},
            {"video_id": "v", "delivery_id": "1", "annotator_id": "adjudicator",
             "elbow_flexion_deg": 165.0},
        ]
        ref = reference_rows(rows)
        assert len(ref) == 1
        assert ref[0]["annotator_id"] == "adjudicator"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def _samples(*pairs):
    return [(f"v{i}", float(gt), float(pred))
            for i, (gt, pred) in enumerate(pairs)]


class TestAgreementStats:
    def test_insufficient_data_when_no_samples(self):
        m = compute_feature_agreement("elbow_flexion_deg", [])
        assert m["status"] == "no_paired_samples"
        assert m["bias"] is None and m["mae"] is None and m["rmse"] is None
        assert m["icc21"] is None and m["pearson_r"] is None

    def test_perfect_agreement(self):
        m = compute_feature_agreement("elbow_flexion_deg",
                                      _samples((1, 1), (2, 2), (3, 3), (4, 4),
                                               (5, 5), (6, 6)))
        assert m["status"] == "measured"
        assert m["bias"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["pearson_r"] == pytest.approx(1.0)
        assert m["r2"] == pytest.approx(1.0)
        assert m["icc21"] == pytest.approx(1.0, abs=1e-4)

    def test_constant_offset_bias(self):
        m = compute_feature_agreement("elbow_flexion_deg",
                                      _samples((10, 12), (20, 22), (30, 32),
                                               (40, 42)))
        assert m["bias"] == pytest.approx(2.0)
        assert m["mae"] == pytest.approx(2.0)
        assert m["rmse"] == pytest.approx(2.0)

    def test_few_samples_suppress_correlation(self):
        m = compute_feature_agreement("elbow_flexion_deg", _samples((10, 12), (20, 22)))
        # descriptive metrics present
        assert m["mae"] == pytest.approx(2.0)
        # correlation suppressed (needs >= 4)
        assert m["pearson_r"] is None
        assert m["r2"] is None

    def test_single_sample_only_descriptive(self):
        m = compute_feature_agreement("elbow_flexion_deg", _samples((10, 13)))
        assert m["mae"] == pytest.approx(3.0)
        assert m["sd_error"] is None
        assert m["loa_lower"] is None
        assert m["icc21"] is None

    def test_constant_prediction_returns_no_correlation(self):
        m = compute_feature_agreement("trunk_lean_deg",
                                      _samples((10, 5), (20, 5), (30, 5), (40, 5)))
        assert m["pearson_r"] is None  # zero-variance prediction -> None, not NaN


class TestBlandAltman:
    def test_known_values(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = np.array([12.0, 22.0, 32.0, 42.0, 52.0])
        ba = bland_altman(gt, pred)
        assert ba["bias"] == pytest.approx(2.0)
        assert ba["sd_diff"] == pytest.approx(0.0)
        assert ba["loa_lower"] == pytest.approx(2.0)
        assert ba["loa_upper"] == pytest.approx(2.0)

    def test_loa_spread(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = gt + np.array([1.0, -1.0, 2.0, -2.0, 0.5])
        ba = bland_altman(gt, pred)
        assert ba["loa_lower"] < ba["bias"] < ba["loa_upper"]

    def test_insufficient_for_loa(self):
        m = compute_feature_agreement("elbow_flexion_deg",
                                      _samples((10, 12), (20, 22)))
        assert m["loa_lower"] is None and m["loa_upper"] is None


class TestICC:
    def test_perfect(self):
        r = icc_two_way_random_single(np.array([1., 2., 3., 4., 5.]),
                                      np.array([1., 2., 3., 4., 5.]))
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_collinear_offset_lowers_absolute_agreement_icc(self):
        r = icc_two_way_random_single(np.array([1., 2., 3., 4., 5.]),
                                      np.array([2., 3., 4., 5., 6.]))
        # ICC(2,1) is an ABSOLUTE-agreement index: a constant +1 rater offset
        # is penalized (5/6 by hand via Shrout & Fleiss ANOVA terms).
        assert r == pytest.approx(5.0 / 6.0, abs=1e-4)

    def test_degenerate_constant_returns_none(self):
        r = icc_two_way_random_single(np.ones(5), np.ones(5))
        assert r is None

    def test_less_degenerate(self):
        gt = np.array([1., 2., 3., 4., 5., 6.])
        pred = np.array([1.2, 1.9, 3.3, 3.8, 5.4, 5.7])
        r = icc_two_way_random_single(gt, pred)
        assert r is None or 0.0 < r <= 1.0


class TestFrameEventMetrics:
    def test_perfect(self):
        m = compute_frame_event_metrics(
            "release_frame", _samples((15, 15), (20, 20), (30, 30)), fps=20.0)
        assert m["mean_abs_frame_error"] == pytest.approx(0.0)
        assert m["mean_abs_error_ms"] == pytest.approx(0.0)

    def test_frame_and_ms_conversion(self):
        m = compute_frame_event_metrics(
            "release_frame", _samples((15, 15), (20, 21), (30, 31)), fps=20.0)
        assert m["mean_abs_frame_error"] == pytest.approx(2 / 3, abs=0.01)
        # 1 frame @ 20fps = 50 ms
        assert m["median_abs_frame_error"] == pytest.approx(1.0)
        assert m["median_abs_error_ms"] == pytest.approx(50.0)

    def test_signed_error_sign_gt_minus_pred(self):
        m = compute_frame_event_metrics(
            "release_frame", _samples((20, 21)), fps=20.0)
        # gt 20, pred 21 -> signed error = -1 (model late)
        assert m["mean_signed_frame_error"] == pytest.approx(-1.0)

    def test_no_samples(self):
        m = compute_frame_event_metrics("release_frame", [], fps=20.0)
        assert m["status"] == "no_paired_samples"
        assert m["absolute_errors"] is None

    def test_fps_drives_ms(self):
        m_slow = compute_frame_event_metrics("release_frame", _samples((20, 21)),
                                             fps=10.0)
        m_fast = compute_frame_event_metrics("release_frame", _samples((20, 21)),
                                             fps=30.0)
        assert m_slow["mean_abs_error_ms"] == pytest.approx(100.0)
        assert m_fast["mean_abs_error_ms"] == pytest.approx(1000.0 / 30.0, abs=0.1)


class TestMinSamplesConstants:
    def test_thresholds_sane(self):
        assert MIN_SAMPLES_DESCRIPTIVE == 1
        assert MIN_SAMPLES_LOA == 3
        assert MIN_SAMPLES_CORRELATION == 4
        assert MIN_SAMPLES_ICC == 3


# --------------------------------------------------------------------------- #
# Schema integrity
# --------------------------------------------------------------------------- #

class TestSchema:
    def test_excluded_features_documented(self):
        assert "stride_length_px" in EXCLUDED_FEATURES
        assert "release_speed_mps" in EXCLUDED_FEATURES

    def test_continuous_mapping_keys_are_gt_columns(self):
        for key in CONTINUOUS_FEATURES:
            assert key in GT_COLUMNS

    def test_frame_events_are_gt_columns(self):
        for key in FRAME_EVENTS:
            assert key in GT_COLUMNS


# --------------------------------------------------------------------------- #
# Annotator geometry + CSV persistence
# --------------------------------------------------------------------------- #

class TestAnnotatorGeometry:
    def test_angle_90(self):
        assert annot.angle_at_vertex((0, 0), (0, 1), (2, 1)) == pytest.approx(90.0)

    def test_angle_straight(self):
        assert annot.angle_at_vertex((0, 0), (1, 1), (2, 2)) == pytest.approx(180.0)

    def test_angle_zero(self):
        assert annot.angle_at_vertex((0, 0), (1, 1), (0, 2)) == pytest.approx(90.0)

    def test_trunk_lean_upright(self):
        assert annot.trunk_lean_from_points((0, 10), (0, 0)) == pytest.approx(0.0)

    def test_trunk_lean_forward(self):
        # hip lower (y=10), shoulder above and 5 px forward of hip: lean is
        # measured from the vertical -> atan(5/10)
        lean = annot.trunk_lean_from_points((0, 10), (5, 0))
        assert lean == pytest.approx(math.degrees(math.atan(0.5)), abs=0.1)
        assert 0 < lean < 90

    def test_stride_distance(self):
        assert annot.stride_distance((0, 0), (30, 40)) == pytest.approx(50.0)

    def test_release_angle_horizontal(self):
        assert annot.release_angle_from_ball((0, 0), (100, 0)) == pytest.approx(0.0)


class TestAnnotatorCSV:
    def test_upsert_creates_header_only_file(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
        row = {c: "" for c in GT_COLUMNS}
        row.update({"video_id": "clip1", "annotator_id": "A", "delivery_id": "1",
                    "release_frame": "15", "elbow_flexion_deg": "165.0"})
        assert annot.upsert_ground_truth_row(path, row) is True
        records, _ = annot.read_ground_truth_records(path)
        assert ("clip1", "A", "1") in records

    def test_upsert_replaces_same_annotator(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
        base = {c: "" for c in GT_COLUMNS}
        row1 = dict(base, video_id="clip1", annotator_id="A", delivery_id="1",
                    release_frame="15")
        row2 = dict(base, video_id="clip1", annotator_id="A", delivery_id="1",
                    release_frame="20")
        annot.upsert_ground_truth_row(path, row1)
        annot.upsert_ground_truth_row(path, row2)
        records, _ = annot.read_ground_truth_records(path)
        assert records[("clip1", "A", "1")]["release_frame"] == "20"
        assert len(records) == 1

    def test_upsert_keeps_second_annotator(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
        base = {c: "" for c in GT_COLUMNS}
        annot.upsert_ground_truth_row(
            path, dict(base, video_id="clip1", annotator_id="A", delivery_id="1"))
        annot.upsert_ground_truth_row(
            path, dict(base, video_id="clip1", annotator_id="B", delivery_id="1"))
        records, _ = annot.read_ground_truth_records(path)
        assert ("clip1", "A", "1") in records
        assert ("clip1", "B", "1") in records
        assert len(records) == 2

    def test_session_rejects_empty_annotator(self):
        tmp = tempfile.mkdtemp()
        session = annot.AnnotationSession(
            video_path="nonexistent.avi", annotator_id="",
            delivery_id="1", out_path=os.path.join(tmp, "gt.csv"))
        row = session.row()
        assert annot.upsert_ground_truth_row(os.path.join(tmp, "gt2.csv"), row) is False


# --------------------------------------------------------------------------- #
# Predicted-cache builder (pure re-format, not ground truth)
# --------------------------------------------------------------------------- #

class TestPredictionBuilder:
    def test_build_from_runs_keeps_prediction_keys(self):
        tmp = tempfile.mkdtemp()
        runs_dir = os.path.join(tmp, "runs")
        pred_dir = os.path.join(tmp, "preds")
        os.makedirs(runs_dir)
        import json
        run = {
            "video": "x.avi",
            "features": {"elbow_flexion_deg": 22.8, "trunk_lean_deg": 25.4},
            "diagnostics": {"release_frame_idx": 15,
                            "front_foot_contact_frame": 11,
                            "reliable": True},
        }
        with open(os.path.join(runs_dir, "clip1.json"), "w") as f:
            json.dump(run, f)
        written = build_predictions_from_runs(runs_dir, pred_dir)
        assert len(written) == 1
        pred = load_prediction("clip1", pred_dir)
        assert pred is not None
        assert pred["release_frame"] == 15
        assert pred["front_contact_frame"] == 11
        assert prediction_frame(pred, "release_frame") == 15
        assert prediction_feature(pred, "elbow_flexion_deg") == pytest.approx(22.8)

    def test_build_from_missing_dir_returns_empty(self):
        tmp = tempfile.mkdtemp()
        assert build_predictions_from_runs(os.path.join(tmp, "nope"),
                                           os.path.join(tmp, "p")) == []


# --------------------------------------------------------------------------- #
# End-to-end runner (no annotations -> honest insufficient data)
# --------------------------------------------------------------------------- #

class TestRunner:
    def test_no_annotations_is_honest(self):
        tmp = tempfile.mkdtemp()
        gt_path = os.path.join(tmp, "gt.csv")
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
        res = run_ground_truth_validation(
            gt_path=gt_path,
            predictions_dir=os.path.join(tmp, "preds"),
            results_dir=os.path.join(tmp, "res"),
            write_reports=False)
        assert res["status"] == "PILOT_VALIDATION"
        assert res["ground_truth_rows"] == 0
        assert res["paired_samples_available"] is False
        for m in res["feature_metrics"].values():
            assert m["status"] == "no_paired_samples"
        assert res["excluded_features"]["stride_length_px"]["reason"]

    def test_missing_gt_file_reports_error(self):
        tmp = tempfile.mkdtemp()
        res = run_ground_truth_validation(
            gt_path=os.path.join(tmp, "missing.csv"),
            predictions_dir=os.path.join(tmp, "preds"),
            results_dir=os.path.join(tmp, "res"),
            write_reports=False)
        assert res["parse_errors"]