"""
P1.1 tests: annotation workflow + quality control.

Covers:
  * annotation data validation (duplicates, invalid frames, angle bounds,
    annotator_id, visibility/confidence vocabulary, fps metadata)
  * ground-truth immutability when the evaluator runs
  * ground-truth / prediction separation (builders never touch the other side)
  * reproducibility fields (protocol version, annotation date, prediction
    artifact version)
  * honest annotation-status reporting when nothing is annotated yet
  * the "NO REAL GROUND TRUTH -- INSUFFICIENT DATA" evaluator phrase
"""
import datetime
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import annotator as annot
from evaluation.ground_truth_validation import (
    GT_COLUMNS, ANNOTATION_PROTOCOL_VERSION,
    parse_ground_truth_csv, load_clip_metadata, validate_annotations,
    build_predictions_from_runs, run_ground_truth_validation,
    build_validation_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from build_annotation_status import build_status_markdown  # noqa: E402


def _gt_record(video_id="clip1", delivery_id="1", annotator_id="ANNOTATOR_A",
               release_frame=15, front_contact_frame=11, **overrides):
    record = {
        "video_id": video_id,
        "delivery_id": delivery_id,
        "annotator_id": annotator_id,
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "annotation_date": "2026-09-11",
        "release_frame": release_frame,
        "front_contact_frame": front_contact_frame,
        "front_knee_angle_deg": 130.0,
        "elbow_flexion_deg": 165.0,
        "trunk_lean_deg": 25.0,
        "stride_length_px": 120.0,
        "release_angle_deg": 5.0,
        "release_speed_mps": "",
        "visibility_release": "good",
        "visibility_knee": "good",
        "visibility_elbow": "partial",
        "visibility_trunk": "good",
        "annotation_confidence": "5",
        "notes": "",
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Annotation data validation
# --------------------------------------------------------------------------- #

class TestValidateAnnotations:
    def test_valid_row_has_no_issues(self):
        meta = {"clip1": {"frame_count": 60, "fps": 20.0}}
        assert validate_annotations([_gt_record()], meta) == []

    def test_duplicate_row_detected(self):
        rows = [_gt_record(), _gt_record()]
        issues = validate_annotations(rows, {})
        codes = [i["code"] for i in issues]
        assert "duplicate_row" in codes

    def test_missing_annotator_id(self):
        rows = [_gt_record(annotator_id="")]
        issues = validate_annotations(rows, {})
        assert any(i["code"] == "missing_annotator_id" for i in issues)

    def test_negative_frame_reported(self):
        issues = validate_annotations([_gt_record(release_frame=-3,
                                                  front_contact_frame=-1)], {})
        codes = [i["code"] for i in issues]
        assert codes.count("negative_frame") == 2

    def test_frame_beyond_clip_length_reported(self):
        meta = {"clip1": {"frame_count": 60, "fps": 20.0}}
        issues = validate_annotations([_gt_record(release_frame=61)], meta)
        assert any(i["code"] == "frame_out_of_range" for i in issues)

    def test_no_frame_bound_check_without_metadata(self):
        # without metadata we must not invent the clip length
        issues = validate_annotations([_gt_record(release_frame=9999)], {})
        assert all(i["code"] != "frame_out_of_range" for i in issues)

    def test_angle_out_of_bounds_reported(self):
        issues = validate_annotations(
            [_gt_record(elbow_flexion_deg=275.0, trunk_lean_deg=91.0,
                        release_angle_deg=-10.0)], {})
        codes = [i["code"] for i in issues]
        assert codes.count("out_of_bounds") == 3

    def test_invalid_visibility_reported(self):
        issues = validate_annotations(
            [_gt_record(visibility_knee="guessed")], {})
        assert any(i["code"] == "invalid_visibility" for i in issues)

    def test_invalid_confidence_reported(self):
        issues = validate_annotations([_gt_record(annotation_confidence="9")], {})
        assert any(i["code"] == "invalid_confidence" for i in issues)

    def test_invalid_visibility_is_not_accepted(self):
        # occluded is a documented value (not a validation error)
        assert validate_annotations([_gt_record(visibility_elbow="occluded")],
                                    {}) == []


class TestClipMetadata:
    def test_loads_fps_and_frame_count(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "inventory.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("video_id,fps,frame_count,width,height\n")
            f.write("clip1,25.0,61,1280,720\n")
            f.write("bad,UNKNOWN,UNKNOWN,0,0\n")
        meta = load_clip_metadata(path)
        assert meta["clip1"] == {"fps": 25.0, "frame_count": 61}
        # invalid metadata is dropped, never invented
        assert meta["bad"] == {"fps": None, "frame_count": None}

    def test_missing_inventory_returns_empty(self):
        assert load_clip_metadata(os.path.join(tempfile.mkdtemp(), "nope.csv")) == {}


# --------------------------------------------------------------------------- #
# Ground-truth immutability + GT/prediction separation
# --------------------------------------------------------------------------- #

class TestSeparation:
    def test_validation_never_modifies_ground_truth_csv(self):
        tmp = tempfile.mkdtemp()
        gt_path = os.path.join(tmp, "gt.csv")
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
            f.write("clip1,1,ANNOTATOR_A,1.1,2026-09-11,15,11,130,165,25,"
                    "120,5,,good,good,partial,good,5,\n")
        before = open(gt_path, "rb").read()
        res_path = os.path.join(tmp, "res")

        run_ground_truth_validation(
            gt_path=gt_path,
            predictions_dir=os.path.join(tmp, "preds"),
            results_dir=res_path,
            inventory_path=os.path.join(tmp, "no_inventory.csv"),
            write_reports=True)

        assert open(gt_path, "rb").read() == before
        # reports go to results_dir only
        assert os.listdir(res_path)
        assert sorted(os.listdir(res_path)) == sorted([
            "validation_summary.csv", "feature_errors.csv",
            "bland_altman_data.csv", "agreement_metrics.json",
            "validation_report.md"])

    def test_gt_validation_note_never_fabricates_pairs(self):
        # a GT row with a frame beyond the (known) clip length is REPORTED,
        # not clamped, and produces no paired samples
        tmp = tempfile.mkdtemp()
        gt_path = os.path.join(tmp, "gt.csv")
        inv_path = os.path.join(tmp, "inventory.csv")
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
            f.write("clip1,1,ANNOTATOR_A,1.1,2026-09-11,9999,9999,,,,\n")
        with open(inv_path, "w", newline="", encoding="utf-8") as f:
            f.write("video_id,fps,frame_count\nclip1,20.0,60\n")
        res = run_ground_truth_validation(
            gt_path=gt_path, predictions_dir=os.path.join(tmp, "preds"),
            results_dir=os.path.join(tmp, "res"), inventory_path=inv_path,
            write_reports=False)
        assert res["paired_samples_available"] is False
        codes = [i["code"] for i in res["validation_issues"]]
        assert codes.count("frame_out_of_range") == 2

    def test_prediction_builder_writes_only_predictions_dir(self):
        tmp = tempfile.mkdtemp()
        runs_dir = os.path.join(tmp, "runs")
        pred_dir = os.path.join(tmp, "preds")
        gt_dir = os.path.join(tmp, "ground_truth")
        os.makedirs(runs_dir)
        os.makedirs(gt_dir)
        with open(os.path.join(runs_dir, "clip1.json"), "w") as f:
            json.dump({"features": {"elbow_flexion_deg": 160.0},
                       "diagnostics": {"release_frame_idx": 15}}, f)
        snapshot = sorted(os.listdir(gt_dir))
        build_predictions_from_runs(runs_dir, pred_dir)
        # ground_truth dir untouched; predictions written to predictions dir
        assert sorted(os.listdir(gt_dir)) == snapshot
        assert os.path.exists(os.path.join(pred_dir, "clip1_pred.json"))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

class TestReproducibility:
    def test_annotator_session_stamps_protocol_and_date(self):
        tmp = tempfile.mkdtemp()
        session = annot.AnnotationSession(
            video_path="nonexistent.avi", annotator_id="ANNOTATOR_A",
            delivery_id="1", out_path=os.path.join(tmp, "gt.csv"))
        row = session.row()
        assert row["annotation_protocol_version"] == ANNOTATION_PROTOCOL_VERSION
        assert row["annotation_date"].startswith((str(datetime.date.today().year),))

    def test_prediction_cache_carries_artifact_metadata(self):
        tmp = tempfile.mkdtemp()
        runs_dir = os.path.join(tmp, "runs")
        pred_dir = os.path.join(tmp, "preds")
        os.makedirs(runs_dir)
        with open(os.path.join(runs_dir, "clip1.json"), "w") as f:
            json.dump({"features": {}, "diagnostics": {}}, f)
        build_predictions_from_runs(runs_dir, pred_dir)
        with open(os.path.join(pred_dir, "clip1_pred.json")) as f:
            pred = json.load(f)
        assert pred["frame_space"] == "preprocessed_20fps_640x360"
        assert "NOT ground truth" in pred["note"]


# --------------------------------------------------------------------------- #
# Honest no-data reporting
# --------------------------------------------------------------------------- #

class TestHonestNoData:
    def test_annotation_status_empty_gt(self):
        inventory = [
            {"video_id": "clip1", "view_type": "UNKNOWN",
             "usable_for_annotation": "UNKNOWN", "notes": "x"},
            {"video_id": "clip2", "view_type": "side_view",
             "usable_for_annotation": "yes", "notes": "x"},
        ]
        text, counts = build_status_markdown(inventory, [])
        assert counts == {"total": 2, "reviewed": 1, "usable": 1,
                          "annotated": 0, "missing": 2}
        assert "Human annotation required." in text
        assert "not_annotated" in text

    def test_annotation_status_with_rows(self):
        inventory = [
            {"video_id": "clip1", "view_type": "side_view",
             "usable_for_annotation": "yes", "notes": "x"},
        ]
        gt = [{c: "" for c in GT_COLUMNS}]
        gt[0]["video_id"] = "clip1"
        gt[0]["annotator_id"] = "ANNOTATOR_A"
        gt[0]["release_frame"] = "15"
        gt[0]["front_contact_frame"] = "11"
        text, counts = build_status_markdown(inventory, gt)
        assert counts["annotated"] == 1
        assert "partially_annotated" in text   # most measurements still blank
        assert "elbow_flexion_deg" in text     # listed as missing

    def test_report_contains_no_real_ground_truth_phrase(self):
        results = {
            "ground_truth_rows": 0,
            "annotated_video_ids": [],
            "prediction_video_ids": ["clip1"],
            "paired_samples_available": False,
            "validation_issues": [],
            "excluded_features": {"stride_length_px": {"reason": "unit mismatch"}},
        }
        report = build_validation_report(
            results, [], "gt.csv", "preds")
        assert "NO REAL GROUND TRUTH -- INSUFFICIENT DATA." in report


# --------------------------------------------------------------------------- #
# Parse reports invalid values loudly
# --------------------------------------------------------------------------- #

class TestParseReportsInvalid:
    def test_invalid_frame_value_reported_not_fixed(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
            f.write("clip1,1,ANNOTATOR_A,1.1,2026-09-11,-3,nonsense,,,,\n")
        records, errors = parse_ground_truth_csv(path)
        assert records[0]["release_frame"] is None
        assert records[0]["front_contact_frame"] is None
        assert any("invalid release_frame" in e for e in errors)
        assert any("invalid front_contact_frame" in e for e in errors)

    def test_annotation_row_includes_protocol_fields(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gt.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(GT_COLUMNS) + "\n")
            f.write("clip1,1,ANNOTATOR_A,1.1,2026-09-11,15,11,130,165,25,"
                    "120,5,,good,good,partial,good,5,\n")
        records, _ = parse_ground_truth_csv(path)
        assert records[0]["annotation_protocol_version"] == "1.1"
        assert records[0]["annotation_date"] == "2026-09-11"
        assert records[0]["elbow_flexion_deg"] == pytest.approx(165.0)