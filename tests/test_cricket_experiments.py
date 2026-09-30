"""Tests for the Phase 9-11 training/metrics/experiment layer.

The point of most of these tests is not that the code computes a number. It is
that the code *refuses* to compute one when it has no honest basis. A metric that
silently reports 0.0 because no human labelled a clip is the single most
dangerous failure mode in this project, because 0.0 looks like a real result.

So: `NOT_MEASURED` must survive an empty ground truth, a refused model must stay
refused, the test split must stay out of selection, and a declared (untuned)
prior must stay declared.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.cricket_understanding import metrics as M          # noqa: E402
from src.cricket_understanding import schema, tasks         # noqa: E402

experiments = importlib.import_module("scripts.cricket.experiments")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _gt(video_id="v1", bowler_track_id=3, annotator="human_1", **kw):
    g = M.ClipGroundTruth(video_id=video_id)
    g.annotator_id = annotator
    g.bowler_track_id = bowler_track_id
    g.bowler_frames = kw.get("bowler_frames", [10, 11, 12])
    return g


def _pred(video_id="v1", bowler_track_id=3, confirmed=True, switches=0, **kw):
    p = M.ClipPrediction(video_id=video_id)
    p.predicted_bowler_track_id = bowler_track_id
    p.bowler_confirmed = confirmed
    p.identity_switch_count = switches
    return p


# --------------------------------------------------------------------------- #
# The core safety property: no ground truth -> NOT MEASURED, never 0.0
# --------------------------------------------------------------------------- #

class TestNoGroundTruthStaysNotMeasured:
    def test_every_metric_is_not_measured_with_empty_gt(self):
        res = M.compute_all([_pred()], [])
        assert res, "compute_all must return something"
        for name, metric in res.items():
            assert metric.status == M.NOT_MEASURED, f"{name} was measured without labels"

    def test_no_metric_reports_a_zero_value_when_unmeasured(self):
        res = M.compute_all([_pred(), _pred("v2")], [])
        for name, metric in res.items():
            assert metric.value is None, f"{name} leaked a value: {metric.value}"

    def test_empty_everything_is_not_measured_not_zero(self):
        for name, metric in M.compute_all([], []).items():
            assert metric.status == M.NOT_MEASURED
            assert metric.value is None

    def test_gt_without_human_annotator_is_rejected(self):
        g = _gt()
        g.annotator_id = ""
        res = M.compute_all([_pred()], [g])
        assert res["bowler_top1_accuracy"].status == M.NOT_MEASURED

    def test_model_derived_annotator_is_rejected(self):
        """An annotation produced by a model is not ground truth, however
        confident the annotator field claims to be."""
        g = _gt(annotator="auto_labeler_v3")
        res = M.compute_all([_pred()], [g])
        assert res["bowler_top1_accuracy"].status == M.NOT_MEASURED

    def test_gt_with_ambiguous_bowler_is_not_scorable(self):
        g = _gt()
        g.bowler_track_id = None      # two or more bowlers, or none declared
        res = M.compute_all([_pred()], [g])
        assert res["bowler_top1_accuracy"].status == M.NOT_MEASURED

    def test_not_measured_is_the_sentinel_string(self):
        assert M.NOT_MEASURED == "NOT MEASURED"
        assert M.MEASURED == "MEASURED"

    def test_metric_serialises_as_not_measured(self):
        m = M.compute_all([], [])["bowler_f1"].to_dict()
        assert m["status"] == M.NOT_MEASURED
        assert m["value"] is None
        assert m.get("interval") is None

    def test_rejection_reason_names_the_offending_annotator(self):
        g = _gt(annotator="auto_labeler_v3")
        reason = M.compute_all([_pred()], [g])["bowler_f1"].reason
        assert "auto_labeler_v3" in reason
        assert "not attributable to a human" in reason


# --------------------------------------------------------------------------- #
# With real human ground truth, the metrics must actually compute
# --------------------------------------------------------------------------- #

class TestMetricsWithGroundTruth:
    def test_perfect_prediction_scores_one(self):
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        preds = [_pred(f"v{i}", 3) for i in range(6)]
        res = M.compute_all(preds, gts)
        assert res["bowler_top1_accuracy"].status == M.MEASURED
        assert res["bowler_top1_accuracy"].value == 1.0
        assert res["bowler_recall"].value == 1.0
        assert res["bowler_precision"].value == 1.0
        assert res["bowler_f1"].value == 1.0

    def test_wrong_track_scores_zero_and_is_measured(self):
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        preds = [_pred(f"v{i}", 9) for i in range(6)]
        res = M.compute_all(preds, gts)
        assert res["bowler_top1_accuracy"].status == M.MEASURED
        assert res["bowler_top1_accuracy"].value == 0.0

    def test_abstention_is_measured_not_ignored(self):
        """Refusing to answer is a decision with consequences, so it must land
        in the denominators rather than disappearing."""
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        preds = [_pred(f"v{i}", 3) for i in range(3)]
        preds += [_pred(f"v{i}", None, confirmed=False) for i in (3, 4, 5)]
        res = M.compute_all(preds, gts)
        assert res["bowler_abstention_rate"].status == M.MEASURED
        assert res["bowler_abstention_rate"].value == pytest.approx(0.5)
        assert res["bowler_recall"].value == pytest.approx(0.5)

    def test_false_selection_rate_is_separate_from_error(self):
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        preds = [_pred(f"v{i}", 9 if i % 2 == 0 else 3) for i in range(6)]
        res = M.compute_all(preds, gts)
        assert res["false_selection_rate"].status == M.MEASURED
        assert res["false_selection_rate"].value == pytest.approx(0.5)

    def test_small_sample_gets_no_interval_but_still_a_value(self):
        gts = [_gt("v1", 3), _gt("v2", 9)]
        preds = [_pred("v1", 3), _pred("v2", 3)]
        res = M.compute_all(preds, gts)
        acc = res["bowler_top1_accuracy"]
        assert acc.status == M.MEASURED
        assert acc.value == pytest.approx(0.5)
        assert acc.interval is None, "no interval may be claimed from too few clips"

    def test_six_clip_interval_brackets_the_point_estimate(self):
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        preds = [_pred(f"v{i}", 3 if i < 3 else 9) for i in range(6)]
        acc = M.compute_all(preds, gts)["bowler_top1_accuracy"]
        assert acc.interval is not None
        assert acc.interval.lo <= acc.value <= acc.interval.hi

    def test_a_clip_without_an_annotator_is_excluded_not_scored(self):
        """One bad clip must drop out of the denominators, not be counted as a
        wrong answer -- but the metric stays measured on the rest."""
        gts = [_gt(f"v{i}", 3) for i in range(6)]
        gts[2].annotator_id = ""
        res = M.compute_all([_pred(f"v{i}", 3) for i in range(6)], gts)
        acc = res["bowler_top1_accuracy"]
        assert acc.status == M.MEASURED
        assert acc.n == 5
        assert acc.extra["unscored_clips"] == 1
        assert acc.extra["rejected_ground_truth"] == ["v2:<empty>"]

    def test_wilson_interval_is_bounded_and_centred(self):
        w = M.wilson_interval(5, 10)
        assert w is not None
        assert 0.0 <= w.lo <= 0.5 <= w.hi <= 1.0

    def test_wilson_on_zero_trials_is_none(self):
        assert M.wilson_interval(0, 0) is None

    def test_wilson_on_tiny_sample_is_none(self):
        assert M.wilson_interval(1, 2) is None

    def test_measured_metrics_echo_their_annotators(self):
        gts = [_gt(f"v{i}", 3, annotator=f"human_{i % 2}") for i in range(6)]
        res = M.compute_all([_pred(f"v{i}", 3) for i in range(6)], gts)
        assert res["bowler_top1_accuracy"].extra["annotators"] == ["human_0", "human_1"]


class TestHumanAnnotatorDetection:
    @pytest.mark.parametrize("bad", [
        "", "   ", "auto", "auto_labeler", "model_v2", "yolo11n", "PaceAI_auto",
        "mediapipe", "pseudo_label", "synthetic_gt", "heuristic_v1", "tracking_bot",
    ])
    def test_rejects_non_human_ids(self, bad):
        assert M.is_human_annotator(bad) is False

    @pytest.mark.parametrize("good", [
        "human_1", "annotator_7", "SS", "m.khan", "Dr Patel", "PaceAI_human_review",
    ])
    def test_accepts_people(self, good):
        assert M.is_human_annotator(good) is True

    def test_delivery_metrics_also_reject_non_human_truth(self):
        g = _gt(annotator="auto_v1")
        g.deliveries = {"d1": {"start_frame": 0, "end_frame": 20, "release_frame": 10}}
        p = _pred()
        p.predicted_deliveries = [(0, 20)]
        p.predicted_release_frames = {"d1": 10}
        res = M.compute_all([p], [g])
        assert res["delivery_detection_accuracy"].status == M.NOT_MEASURED

    def test_phase_metrics_also_reject_non_human_truth(self):
        g = _gt(annotator="model_1")
        g.phases = {(10, 3): "RELEASE"}
        p = _pred()
        p.predicted_phases = {(10, 3): "RELEASE"}
        res = M.compute_all([p], [g])
        assert res["phase_accuracy"].status == M.NOT_MEASURED


# --------------------------------------------------------------------------- #
# Test-split protection
# --------------------------------------------------------------------------- #

class TestNoTestSelection:
    def test_plain_selection_passes(self):
        M.assert_no_test_selection({"A": "val", "B": "train"}, ["v_test"])

    def test_selecting_on_test_raises(self):
        with pytest.raises(M.ProtocolViolationError):
            M.assert_no_test_selection({"A": "test"}, ["v_test"])

    def test_test_video_id_anywhere_in_selection_raises(self):
        with pytest.raises(M.ProtocolViolationError):
            M.assert_no_test_selection({"B": "v_test_tuned"}, ["v_test"])

    def test_the_error_explains_why(self):
        with pytest.raises(M.ProtocolViolationError, match="held-out"):
            M.assert_no_test_selection({"A": "test"}, ["v_test"])


# --------------------------------------------------------------------------- #
# Arm comparison
# --------------------------------------------------------------------------- #

class TestCompareArms:
    @staticmethod
    def _metric(name, value, hib=True, interval=(0.0, 1.0)):
        return M.Metric(name, value, M.MEASURED, n=20,
                        interval=M.Interval(interval[0], interval[1], "wilson"),
                        higher_is_better=hib)

    def _arms(self, a_value, b_value, hib=True, ia=(0.0, 1.0), ib=(0.0, 1.0)):
        return M.compare_arms({
            "A": {"bowler_f1": self._metric("bowler_f1", a_value, hib, ia)},
            "B": {"bowler_f1": self._metric("bowler_f1", b_value, hib, ib)},
        })

    def test_two_measured_arms_are_comparable(self):
        out = self._arms(0.6, 0.8)
        row = out["metrics"]["bowler_f1"]
        assert row["comparable_arms"] == ["A", "B"]
        assert row["comparable"] is True

    def test_comparable_row_always_has_a_verdict(self):
        """Regression: the report generator reads row['verdict'] unconditionally.
        It used to be missing on exactly the path that matters -- the first time
        a real holdout makes two arms comparable."""
        out = self._arms(0.6, 0.8)
        assert out["metrics"]["bowler_f1"]["verdict"]

    def test_overlapping_intervals_are_inconclusive(self):
        out = self._arms(0.6, 0.8, ia=(0.4, 0.7), ib=(0.5, 0.9))
        v = out["metrics"]["bowler_f1"]["verdict"]
        assert v.startswith("INCONCLUSIVE")
        assert "cannot be told apart" in v

    def test_separated_intervals_name_the_better_arm(self):
        out = self._arms(0.6, 0.9, ia=(0.50, 0.70), ib=(0.85, 0.95))
        v = out["metrics"]["bowler_f1"]["verdict"]
        assert v.startswith("SEPARATED")
        assert out["metrics"]["bowler_f1"]["best_point_estimate"] == "B"

    def test_lower_is_better_metrics_are_not_ranked_high_first(self):
        out = self._arms(3.0, 1.0, hib=False, ia=(2.5, 3.5), ib=(0.5, 1.5))
        row = out["metrics"]["bowler_f1"]
        assert row["best_point_estimate"] == "B"
        assert "lowest" in row["verdict"]

    def test_missing_interval_cannot_claim_separation(self):
        out = M.compare_arms({
            "A": {"bowler_f1": M.Metric("bowler_f1", 0.6, M.MEASURED, n=3)},
            "B": {"bowler_f1": M.Metric("bowler_f1", 0.9, M.MEASURED, n=3)},
        })
        assert out["metrics"]["bowler_f1"]["verdict"].startswith("INCONCLUSIVE")

    def test_one_measured_arm_is_declared_not_comparable(self):
        out = M.compare_arms({
            "A": {"bowler_f1": M.Metric("bowler_f1", 0.6, M.MEASURED, n=20)},
            "B": {"bowler_f1": M.Metric("bowler_f1", None, M.NOT_MEASURED, n=0,
                                        reason="no human annotation")},
        })
        row = out["metrics"]["bowler_f1"]
        assert row["comparable_arms"] == ["A"]
        assert "NOT COMPARABLE" in row["verdict"]
        assert "best_point_estimate" not in row

    def test_no_measured_arms_is_not_comparable(self):
        out = M.compare_arms({
            "A": {"bowler_f1": M.Metric("bowler_f1", None, M.NOT_MEASURED, n=0)},
            "B": {"bowler_f1": M.Metric("bowler_f1", None, M.NOT_MEASURED, n=0)},
        })
        row = out["metrics"]["bowler_f1"]
        assert row["comparable_arms"] == []
        assert "NOT COMPARABLE" in row["verdict"]

    def test_never_declares_a_single_winner_field(self):
        out = self._arms(0.9, 0.1, ia=(0.85, 0.95), ib=(0.05, 0.15))
        assert "winner" not in out
        assert "best" not in out
        assert "score" not in out

    def test_blocked_arms_appear_explicitly(self):
        out = M.compare_arms({
            "A": {"bowler_f1": M.Metric("bowler_f1", None, M.NOT_MEASURED, n=0)},
            "C": {"bowler_f1": M.Metric("bowler_f1", None, M.NOT_MEASURED, n=0)},
        })
        assert set(out["metrics"]["bowler_f1"]) >= {"A", "C"}
        assert out["metrics"]["bowler_f1"]["A"]["status"] == M.NOT_MEASURED

    def test_identical_arms_overlap(self):
        out = self._arms(0.5, 0.5, ia=(0.4, 0.6), ib=(0.4, 0.6))
        assert out["metrics"]["bowler_f1"]["verdict"].startswith("INCONCLUSIVE")


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #

class TestPairedBootstrap:
    def test_too_few_pairs_is_not_measured(self):
        r = experiments.paired_bootstrap_delta([1.0, 0.0], [0.0, 0.0])
        assert r["interval"] is None
        assert M.NOT_MEASURED in r["verdict"]

    def test_consistent_win_excludes_zero(self):
        a = [1.0] * 10
        b = [0.0] * 10
        r = experiments.paired_bootstrap_delta(a, b)
        assert r["delta"] == pytest.approx(1.0)
        assert r["interval"]["lo"] > 0
        assert "A scores higher" in r["verdict"]

    def test_identical_arms_span_zero(self):
        v = [1.0, 0.0] * 5
        r = experiments.paired_bootstrap_delta(v, list(v))
        assert r["delta"] == pytest.approx(0.0)
        assert r["interval"]["lo"] <= 0 <= r["interval"]["hi"]
        assert "INDISTINGUISHABLE" in r["verdict"]

    def test_missing_pairs_are_dropped(self):
        r = experiments.paired_bootstrap_delta([1.0, None, 0.0], [0.0, 1.0, 0.0])
        assert r["n_pairs"] == 2

    def test_is_deterministic(self):
        a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        assert experiments.paired_bootstrap_delta(a, a) == \
            experiments.paired_bootstrap_delta(a, a)


# --------------------------------------------------------------------------- #
# The spatial prior
# --------------------------------------------------------------------------- #

class _FakeTrack:
    def __init__(self, tid, boxes):
        self.track_id = tid
        self.frames = list(range(len(boxes)))
        self.bboxes = boxes

    def __len__(self):
        return len(self.frames)


class TestSpatialPrior:
    def test_track_low_in_frame_gets_a_bonus(self):
        t = _FakeTrack(1, [(10, 60, 40, 98)] * 30)     # bottom near the frame floor
        high = _FakeTrack(2, [(10, 5, 40, 30)] * 30)    # high in frame
        p = experiments.LowerHalfPrior()
        assert p.bonus(t, 100.0) > p.bonus(high, 100.0)

    def test_bonus_is_bounded_by_the_weight(self):
        t = _FakeTrack(1, [(10, 60, 40, 100)] * 30)
        p = experiments.LowerHalfPrior(weight=0.5)
        assert 0.0 <= p.bonus(t, 100.0) <= 0.5

    def test_empty_track_gets_no_bonus(self):
        assert experiments.LowerHalfPrior().bonus(_FakeTrack(1, []), 100.0) == 0.0

    def test_zero_height_frame_is_not_a_division_error(self):
        t = _FakeTrack(1, [(10, 60, 40, 98)] * 30)
        assert experiments.LowerHalfPrior().bonus(t, 0.0) == 0.0

    def test_prior_is_declared_untuned(self):
        assert experiments.LowerHalfPrior().to_dict()["kind"] == \
            "declared_constant_not_tuned"

    def test_prior_only_reorders_and_never_confirms_alone(self, monkeypatch):
        """With the prior active, arm A's own gates must still run. If the prior
        could confirm a bowler on its own, arm B would be measuring the prior's
        threshold rather than the prior's information."""
        from src import tracking

        tracks = {1: _FakeTrack(1, [(10, 60, 40, 98)] * 30),
                  2: _FakeTrack(2, [(10, 5, 40, 30)] * 30)}
        base = tracking.score_bowler_tracks(tracks, (100, 100), 30)
        without = [r["track_id"] for r in base]

        with experiments.spatial_prior_applied(tracks, (100, 100),
                                               experiments.LowerHalfPrior()):
            with_prior = tracking.score_bowler_tracks(tracks, (100, 100), 30)
        after = [r["track_id"] for r in with_prior]

        assert sorted(without) == sorted(after) == [1, 2], "candidate set must not change"
        assert all("score_without_prior" in r for r in with_prior)

    def test_patch_is_removed_even_if_selection_raises(self):
        from src import tracking
        original = tracking.score_bowler_tracks
        with pytest.raises(RuntimeError):
            with experiments.spatial_prior_applied({}, (100, 100),
                                                   experiments.LowerHalfPrior()):
                raise RuntimeError("boom")
        assert tracking.score_bowler_tracks is original

    def test_selection_still_uses_the_margin_gate(self, monkeypatch):
        """A near-tie must still come back unconfirmed under arm B."""
        from src import tracking
        tracks = {1: _FakeTrack(1, [(10, 60, 40, 98)] * 30),
                  2: _FakeTrack(2, [(10, 59, 40, 97)] * 30)}

        def fake_score(trk, frame_dims=None, total_frames=None):
            base = dict(confidence=0.5, n_frames=30, motion=0.5, active=0.5,
                        start_frame=0, end_frame=29, relative=0.5, growth=0.0,
                        directed=0.5, delivery=0.0)
            return [dict(base, track_id=1, score=0.50),
                    dict(base, track_id=2, score=0.495)]

        monkeypatch.setattr(tracking, "score_bowler_tracks", fake_score)
        monkeypatch.setattr(tracking, "_stitch_continuations",
                            lambda tr, tracks, *a, **k: tr)
        _track, meta = experiments.select_with_prior(
            tracks, (100, 100), experiments.LowerHalfPrior(weight=0.01))
        assert meta["confirmed"] is False
        assert meta["confirm_reason"] == "ambiguous_margin"


# --------------------------------------------------------------------------- #
# Experiment plan
# --------------------------------------------------------------------------- #

class TestExperimentPlan:
    def test_all_four_arms_present(self):
        plan = experiments.build_plan()
        assert [a.key for a in plan] == ["A", "B", "C", "D"]

    def test_A_and_B_are_always_runnable(self):
        plan = {a.key: a for a in experiments.build_plan()}
        assert plan["A"].blocked_by is None
        assert plan["B"].blocked_by is None

    def test_C_is_blocked_without_a_trained_model_B(self):
        rep = tasks.readiness_report()
        plan = {a.key: a for a in experiments.build_plan()}
        expected = None if rep["tasks"]["B"]["ready"] else "model_B"
        assert plan["C"].blocked_by == expected

    def test_D_is_blocked_without_a_trained_model_C(self):
        rep = tasks.readiness_report()
        plan = {a.key: a for a in experiments.build_plan()}
        expected = None if rep["tasks"]["C"]["ready"] else "model_C"
        assert plan["D"].blocked_by == expected

    def test_runnable_flag_matches_blocked_by(self):
        for a in experiments.build_plan():
            assert a.to_dict()["runnable"] is (a.blocked_by is None)


# --------------------------------------------------------------------------- #
# Training readiness guards
# --------------------------------------------------------------------------- #

class TestTrainingReadiness:
    @staticmethod
    def _failed(r):
        return [g for g in r.guards if not g.passed]

    def test_three_tasks_exist(self):
        assert set(tasks.TASKS) == {"A", "B", "C"}

    def test_no_rows_means_not_ready(self):
        for key, spec in tasks.TASKS.items():
            r = tasks.evaluate_readiness(spec, [])
            assert r.ready is False, f"task {key} was ready with no training rows"
            assert self._failed(r)

    def test_rows_without_a_human_annotator_are_refused(self):
        spec = tasks.TASKS["B"]
        rows = [{"image": "a.jpg", "label": spec.labels[0], "split": "train",
                 "annotator_id": "auto_labeler"} for _ in range(5)]
        r = tasks.evaluate_readiness(spec, rows)
        assert r.ready is False
        names = {g.name for g in self._failed(r)}
        assert "human_labels_present" in names

    def test_human_rows_alone_do_not_satisfy_the_test_split_guard(self):
        spec = tasks.TASKS["B"]
        rows = [{"image": f"{i}.jpg", "label": lab, "split": "train",
                 "annotator_id": "human_1"}
                for i in range(4) for lab in spec.labels]
        r = tasks.evaluate_readiness(spec, rows)
        assert r.ready is False
        names = {g.name for g in self._failed(r)}
        assert "held_out_split_nonempty" in names

    def test_test_split_alone_does_not_satisfy_the_human_guard(self):
        spec = tasks.TASKS["B"]
        rows = [{"image": f"{i}.jpg", "label": lab, "split": "test",
                 "annotator_id": "human_1"}
                for i in range(4) for lab in spec.labels]
        r = tasks.evaluate_readiness(spec, rows)
        assert r.ready is False
        names = {g.name for g in self._failed(r)}
        assert "human_labels_present" in names

    def test_too_few_examples_per_class_blocks_readiness(self):
        spec = tasks.TASKS["B"]
        rows = [{"image": f"{i}.jpg", "label": spec.labels[0], "split": "train",
                 "annotator_id": "human_1"} for i in range(1)]
        r = tasks.evaluate_readiness(spec, rows)
        assert r.ready is False
        assert r.unlearnable_classes

    def test_the_three_guards_are_all_reported_by_name(self):
        r = tasks.evaluate_readiness(tasks.TASKS["B"], [])
        assert {g.name for g in r.guards} == {
            "human_labels_present", "held_out_split_nonempty",
            "label_support_per_class"}

    def test_counts_are_reported_not_invented(self):
        r = tasks.evaluate_readiness(tasks.TASKS["B"], [])
        assert (r.n_train, r.n_val, r.n_test) == (0, 0, 0)
        assert r.per_class_train == {}

    def test_status_says_no_human_labels_when_there_are_none(self):
        r = tasks.evaluate_readiness(tasks.TASKS["B"], [])
        assert "no human-labelled samples" in r.status

    def test_require_ready_raises_and_names_the_failed_guards(self):
        with pytest.raises(tasks.TaskNotReadyError, match="human_labels_present"):
            tasks.require_ready(tasks.evaluate_readiness(tasks.TASKS["B"], []))

    def test_readiness_report_covers_every_task(self):
        rep = tasks.readiness_report()
        assert set(rep["tasks"]) == {"A", "B", "C"}

    def test_report_says_nothing_is_ready_without_data(self):
        rep = tasks.readiness_report()
        for key in ("A", "B", "C"):
            assert rep["tasks"][key]["ready"] is False, f"{key} claimed ready"
        assert rep["n_ready"] == 0

    def test_generated_markdown_states_zero_of_three_are_trainable(self):
        md = tasks.tasks_markdown()
        assert "0 of 3 tasks are trainable" in md


# --------------------------------------------------------------------------- #
# The generated artefacts must not overclaim
# --------------------------------------------------------------------------- #

class TestGeneratedArtefacts:
    def _read(self, rel):
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            pytest.skip(f"{rel} not generated yet")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_baseline_json_reports_not_measured(self):
        data = json.loads(self._read("evaluation/cricket_understanding_baseline.json"))
        text = json.dumps(data)
        if '"n_human_annotated"' in text:
            assert "NOT MEASURED" in text

    def test_baseline_report_never_claims_accuracy(self):
        text = self._read("evaluation/cricket_understanding_baseline.md").lower()
        for banned in ("accuracy of", "achieves", "improves accuracy",
                       "outperforms", "state of the art"):
            assert banned not in text, f"baseline report overclaims: {banned!r}"

    def test_experiments_report_declares_a_blocked_arm(self):
        text = self._read("evaluation/cricket_experiments.md")
        assert "model_B" in text or "model_C" in text

    def test_experiments_report_has_no_composite_winner(self):
        text = self._read("evaluation/cricket_experiments.md").lower()
        assert "overall winner" not in text
        assert "best arm" not in text


class TestDocsDoNotOverclaim:
    """The hand-written docs are the one place a number can drift in without a
    test noticing, because they are not generated. These checks are deliberately
    blunt: if a doc says a metric is measured, that is a bug."""

    DOCS = ("docs/cricket_understanding_training.md",
            "evaluation/cricket_understanding_report.md")

    def _doc(self, rel):
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            pytest.skip(f"{rel} not written yet")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_no_doc_claims_a_measured_accuracy(self):
        for rel in self.DOCS:
            low = self._doc(rel).lower()
            for banned in ("achieves ", "we achieve ", "outperforms",
                           "beats the baseline", "state of the art", "sota"):
                assert banned not in low, f"{rel} overclaims: {banned!r}"

    def test_no_line_attaches_a_value_to_accuracy(self):
        """Precise version of the anti-overclaim check: a line is only suspect if
        it mentions accuracy *and* puts a number on it. Metric-name listings and
        definitions are legitimate; 'accuracy of 0.87' is not."""
        negations = ("not measured", "not a claim", "not zero", "cannot",
                     "needs a human", "no human", "unmeasured", "not yet",
                     "0 of 12", "not comparable", "refus", "will report",
                     "never", "self-agreement", "not accuracy", "no accuracy")
        value = re.compile(r"\d+\.\d|\d+\s*%")
        for rel in self.DOCS:
            for i, line in enumerate(self._doc(rel).splitlines(), 1):
                low = line.lower()
                if "accuracy" not in low or not value.search(line):
                    continue
                assert any(n in low for n in negations), \
                    f"{rel}:{i} attaches a value to accuracy: {line.strip()!r}"

    def test_no_doc_reports_an_accuracy_percentage(self):
        """A 'NN% accuracy' figure would mean a fabricated result. Behavioural
        rates (confirmation rate) are allowed because they are labelled as
        behaviour; a percentage attached to the word accuracy is not."""
        for rel in self.DOCS:
            for m in re.finditer(r"accuracy[^.\n]{0,40}?(\d+\.?\d*)\s*%",
                                 self._doc(rel), re.IGNORECASE):
                pytest.fail(f"{rel} reports accuracy as a percentage: {m.group(0)!r}")

    def test_docs_agree_that_zero_videos_are_annotated(self):
        for rel in self.DOCS:
            assert "0" in self._doc(rel)
            assert "72" in self._doc(rel), f"{rel} should state the corpus size"

    def test_report_records_the_wrong_torch_claim_was_corrected(self):
        """The Phase 0 audit asserted torch was unusable. If that correction is
        ever quietly reverted, the docs must not still claim Model A cannot run."""
        audit = os.path.join(REPO, "evaluation/cricket_training_current_state.md")
        if not os.path.exists(audit):
            pytest.skip("audit not present")
        with open(audit, encoding="utf-8") as fh:
            text = fh.read()
        assert "SUPERSEDED" in text or "CORRECTED" in text
        assert "venv\\Scripts\\python.exe" in text or "venv/Scripts/python.exe" in text
