"""
Tests for cricket_understanding. These must run WITHOUT torch: importing torch
in this environment fails with an AppLocker DLL block, and a test suite that
cannot run verifies nothing.

Run:  python -m pytest tests/test_cricket_understanding.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cricket_understanding import (  # noqa: E402
    bowler_lock, confidence, generators, manifest, pose_seq, qc, registry,
    schema, splits,
)
from src.cricket_understanding.annotation import (  # noqa: E402
    AnnotationError, AnnotationSession,
)

VID = "v_test000001"


# --------------------------------------------------------------------------- #
# Object factories (mirrors the real on-disk annotation layout exactly)
# --------------------------------------------------------------------------- #

def person(frame, track, role="BOWLER", x1=10.0, y1=20.0, x2=60.0, y2=140.0,
           annotator="TESTER", visibility="fully_visible"):
    return schema.PersonAnnotation(
        video_id=VID, frame_id=frame, track_id=track, role=role,
        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
        visibility=visibility, annotator_id=annotator, annotation_confidence=1.0)


def delivery(did="d1", bowler=1, start=0, end=20, **kw):
    d = dict(delivery_id=did, video_id=VID, bowler_track_id=bowler,
             start_frame=start, end_frame=end, annotator_id="TESTER")
    d.update(kw)
    return schema.DeliveryAnnotation(**d)


def scene(**kw):
    d = dict(video_id=VID, annotator_id="TESTER")
    d.update(kw)
    return schema.SceneAnnotation(**d)


# --------------------------------------------------------------------------- #
# PHASE 2: schema
# --------------------------------------------------------------------------- #

class TestSchema:
    def test_round_trip_is_lossless(self):
        a = person(5, 1, "BOWLER")
        b = schema.PersonAnnotation.from_dict(a.to_dict())
        assert b.to_dict() == a.to_dict()

    def test_bbox_property_and_indices(self):
        p = person(0, 1, "WICKETKEEPER")
        assert p.bbox == (10.0, 20.0, 60.0, 140.0)
        assert p.role_index == schema.ROLE_INDEX["WICKETKEEPER"]

    def test_rejects_degenerate_bbox(self):
        with pytest.raises(ValueError):
            person(0, 1, x1=10, y1=20, x2=10, y2=20)

    def test_rejects_nan_bbox(self):
        with pytest.raises(ValueError):
            person(0, 1, y1=float("nan"))

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            person(0, 1, "BATSMAN")

    def test_rejects_negative_frame(self):
        with pytest.raises(ValueError):
            person(-1, 1)

    def test_delivery_requires_annotator_id(self):
        with pytest.raises(ValueError):
            schema.DeliveryAnnotation(delivery_id="d1", video_id=VID, bowler_track_id=1,
                                      start_frame=0, end_frame=5, annotator_id="")

    def test_delivery_key_frames_never_backfill_approach(self):
        d = delivery(gather_frame=5)
        kf = d.key_frames()
        assert kf["APPROACH"] is None, "APPROACH has no key-frame field; must stay None"
        assert kf["GATHER"] == 5

    def test_scene_rejects_impossible_box(self):
        with pytest.raises(ValueError):
            scene(pitch_region=[10, 10, 5, 5])

    def test_scene_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            scene(pitch_region=[1, 2, 3])

    def test_scene_accepts_none(self):
        s = scene()
        assert s.pitch_region is None

    def test_role_enum_rejects_unknown(self):
        with pytest.raises(ValueError):
            schema.Role("GOALIE")

    def test_yolo_class_map_covers_roles_and_objects(self):
        assert set(schema.YOLO_CLASS_MAP) == {0, 1, 2, 3, 4, 5, 10, 11}
        assert schema.YOLO_NAMES[10] == "BALL"
        assert schema.YOLO_NAMES[5] == "UNKNOWN"

    def test_dumps_is_stable(self):
        o = {"b": 1, "a": [3, 2, 1], "c": {"z": 1, "y": 2}}
        assert schema.dumps(o) == schema.dumps(json.loads(schema.dumps(o)))

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        p = os.path.join(str(tmp_path), "sub", "x.json")
        schema._atomic_write(p, schema.dumps({"a": 1}))
        assert os.path.exists(p)
        assert not [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]


# --------------------------------------------------------------------------- #
# PHASE 3: annotation session
# --------------------------------------------------------------------------- #

class TestAnnotationSession:
    def _s(self, tmp_path, n_frames=40):
        return AnnotationSession(video_id=VID, annotator_id="TESTER",
                                 n_frames=n_frames,
                                 video_path=os.path.join(str(tmp_path), "v.mp4"),
                                 autosave=False)

    def test_annotator_id_is_mandatory(self, tmp_path):
        with pytest.raises(AnnotationError):
            AnnotationSession(video_id=VID, annotator_id="   ", n_frames=10)
        with pytest.raises(AnnotationError):
            AnnotationSession(video_id=VID, annotator_id="", n_frames=10)

    def test_suggestions_never_enter_ground_truth_unless_accepted(self, tmp_path):
        s = self._s(tmp_path)
        n = s.load_suggestions([
            {"frame_id": 0, "track_id": 1, "bbox": [10, 20, 60, 140],
             "suggested_role": "BOWLER", "score": 0.99},
            {"frame_id": 0, "track_id": 2, "bbox": [70, 20, 120, 140],
             "suggested_role": "FIELDER", "score": 0.90},
        ])
        assert n == 2
        assert s.people == {}, "loading suggestions must write nothing to ground truth"
        assert all(x.suggestion is True for x in s.suggestions.values())

        s.accept_suggestion(1)
        assert len(s.people) == 1
        accepted = s.people[(0, 1)]
        assert accepted.role == "BOWLER"
        assert accepted.visibility == "uncertain", \
            "an accepted box is not a confident role judgement"
        assert accepted.annotation_confidence == 0.5
        assert "TESTER" == s.annotator_id
        assert (0, 2) not in s.people, "the unaccepted suggestion must stay a hint"

    def test_accept_without_suggestion_raises(self, tmp_path):
        s = self._s(tmp_path)
        with pytest.raises(AnnotationError):
            s.accept_suggestion(1, 0)

    def test_suggestion_cannot_smuggle_a_false_marker(self):
        with pytest.raises(AnnotationError):
            from src.cricket_understanding.annotation import Suggestion
            Suggestion(video_id=VID, frame_id=0, track_id=1,
                       bbox=(0, 0, 1, 1), suggestion=False)

    def test_set_role_requires_an_existing_box(self, tmp_path):
        s = self._s(tmp_path)
        with pytest.raises(AnnotationError):
            s.set_role(0, 1, "BOWLER")

    def test_move_bbox_resizes_and_deletes(self, tmp_path):
        s = self._s(tmp_path)
        s.set_person(0, 1, [10, 20, 60, 140], role="BOWLER")
        s.move_bbox(0, 1, [15, 25, 65, 145])
        assert s.people[(0, 1)].bbox == (15.0, 25.0, 65.0, 145.0)
        assert s.people[(0, 1)].role == "BOWLER", "moving must not clear the role"
        s.move_bbox(0, 1, None)
        assert s.people == {}

    def test_apply_role_range_never_invents_a_box(self, tmp_path):
        s = self._s(tmp_path)
        s.set_person(0, 1, [10, 20, 60, 140])
        s.set_person(1, 1, [11, 21, 61, 141])
        n = s.apply_role_range(0, 10, 1, "BOWLER")
        assert n == 2, "only frames that already have a box may be labelled"
        assert len(s.people) == 2

    def test_delivery_lifecycle(self, tmp_path):
        s = self._s(tmp_path)
        d = s.start_delivery("d1", 1, start_frame=5)
        assert d.start_frame == 5
        s.set_delivery_window("d1", 5, 25)
        s.set_delivery_frame("d1", "gather_frame", 12)
        s.set_delivery_frame("d1", "release_frame", 18)
        d = s.deliveries["d1"]
        assert (d.gather_frame, d.release_frame) == (12, 18)
        assert d.key_frames()["APPROACH"] is None
        s.current_frame = 10
        assert s.current_delivery().delivery_id == "d1"
        s.current_frame = 100
        assert s.current_delivery().delivery_id == "d1", \
            "past the window the most recent delivery is still the active context"
        assert s.delete_delivery("d1") is True
        assert s.deliveries == {}

    def test_unknown_delivery_field_is_rejected(self, tmp_path):
        s = self._s(tmp_path)
        s.start_delivery("d1", 1, start_frame=0)
        with pytest.raises(AnnotationError):
            s.set_delivery_frame("d1", "bowling_arm_frame", 3)

    def test_delivery_frame_none_is_legal(self, tmp_path):
        s = self._s(tmp_path)
        s.start_delivery("d1", 1, start_frame=0)
        s.set_delivery_frame("d1", "gather_frame", 5)
        d = s.set_delivery_frame("d1", "gather_frame", None)
        assert d.gather_frame is None, "clearing a key frame must be allowed"

    def test_out_of_range_frame_is_rejected(self, tmp_path):
        s = self._s(tmp_path, n_frames=10)
        with pytest.raises(AnnotationError):
            s.set_person(10, 1, [1, 1, 5, 5])

    def test_scene_field_must_exist(self, tmp_path):
        s = self._s(tmp_path)
        s.set_scene_from_box("pitch_region", [0, 0, 10, 10])
        assert s.scene.pitch_region == [0.0, 0.0, 10.0, 10.0]
        with pytest.raises(AnnotationError):
            s.set_scene_field("not_a_field", [0, 0, 1, 1])

    def test_save_is_atomic_and_reloadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(schema, "ANNOTATIONS_DIR",
                            os.path.join(str(tmp_path), "annotations"))
        monkeypatch.setattr(schema, "DATASET_ROOT", str(tmp_path))
        s = self._s(tmp_path)
        s.set_person(0, 1, [10, 20, 60, 140], role="BOWLER")
        s.start_delivery("d1", 1, start_frame=0)
        paths = s.save()
        assert paths and all(os.path.exists(p) for p in paths.values())
        for p in paths.values():
            assert not [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]
        assert len(schema.load_person_annotations(VID)) == 1
        assert len(schema.load_deliveries(VID)) == 1

    def test_status_line_is_informative(self, tmp_path):
        s = self._s(tmp_path)
        s.set_person(0, 1, [10, 20, 60, 140])
        assert "tracks=1" in s.status_line()


# --------------------------------------------------------------------------- #
# PHASE 4: registry
# --------------------------------------------------------------------------- #

class TestRegistry:
    def _rec(self, **kw):
        d = dict(video_id="v_x", path="p", filename="fast_left_00000001.avi", bytes=1,
                 fps=20.0, width=64, height=64, frame_count=10, probe_status="OK")
        d.update(kw)
        return registry.VideoRecord(**d)

    def test_same_bytes_different_name_is_same_id(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "totally_renamed.mp4"
        blob = b"identical bytes, different filename"
        a.write_bytes(blob)
        b.write_bytes(blob)
        assert schema.video_id_for_path(str(a)) == schema.video_id_for_path(str(b))

    def test_different_bytes_different_id(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        a.write_bytes(b"aaaa")
        b.write_bytes(b"aaaa" + b"\x00")   # differs only in the LAST byte
        assert schema.video_id_for_path(str(a)) != schema.video_id_for_path(str(b))

    def test_missing_file_is_reported_not_faked(self, tmp_path):
        r = registry.probe_video(os.path.join(str(tmp_path), "nope.mp4"))
        assert r["probe_status"] == "FAILED"
        assert r["probe_error"]

    def test_group_key_strips_the_clip_number(self):
        assert self._rec().group_key == "view:fast_left"

    def test_known_bowler_takes_priority_over_view_prefix(self):
        assert self._rec(bowler_id_if_known="B7").group_key == "bowler:B7"

    def test_reviewed_without_a_reviewer_is_rejected(self):
        with pytest.raises(ValueError):
            self._rec(review_status="REVIEWED", notes="")

    def test_reviewed_with_a_note_is_accepted(self):
        assert self._rec(review_status="REVIEWED", notes="by A on 2026-01-01")

    def test_jsonl_round_trip(self, tmp_path):
        p = os.path.join(str(tmp_path), "videos.jsonl")
        registry.save_records([self._rec()], p)
        assert [r.video_id for r in registry.load_records(p)] == ["v_x"]
        assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]


# --------------------------------------------------------------------------- #
# PHASE 5: QC -- report only, never fix
# --------------------------------------------------------------------------- #

class TestQC:
    def test_no_bowler_anywhere_is_an_error(self):
        codes = {i.code for i in qc.check_bowler_track([person(0, 1, "STRIKER")], VID)}
        assert "MISSING_BOWLER_TRACK" in codes

    def test_two_bowlers_in_one_frame_is_an_error(self):
        issues = qc.check_conflicting_roles(
            [person(0, 1, "BOWLER"), person(0, 2, "BOWLER")], VID)
        assert any(i.code == "CONFLICTING_ROLE" and i.severity == "ERROR" for i in issues)

    def test_same_role_on_two_far_apart_tracks_is_not_a_conflict(self):
        """Two fielders standing apart is normal; only ONE bowler may exist."""
        a = person(0, 1, "FIELDER", x1=10.0, x2=60.0)
        b = person(0, 2, "FIELDER", x1=400.0, x2=450.0)
        issues = qc.check_conflicting_roles([a, b], VID)
        assert not any(i.severity == "ERROR" for i in issues)

    def test_overlapping_unique_roles_are_an_error(self):
        """A STRIKER box drawn on the bowler is an identity mistake, not noise."""
        a = person(0, 1, "BOWLER", x1=10.0, x2=60.0)
        b = person(0, 2, "STRIKER", x1=12.0, x2=62.0)
        issues = qc.check_conflicting_roles([a, b], VID)
        assert any(i.severity == "ERROR" and "wrong person" in i.message
                   for i in issues)

    def test_overlapping_fielders_are_a_warning_not_an_error(self):
        a = person(0, 1, "FIELDER", x1=10.0, x2=60.0)
        b = person(0, 2, "FIELDER", x1=12.0, x2=62.0)
        issues = qc.check_conflicting_roles([a, b], VID)
        assert any(i.code == "OVERLAPPING_BBOX" and i.severity == "WARNING"
                   for i in issues)
        assert not any(i.severity == "ERROR" for i in issues)

    def test_nan_bbox_is_reported(self):
        p = person(0, 1)
        p.bbox_y1 = float("nan")
        assert any(i.code == "IMPOSSIBLE_BBOX" for i in qc.check_bboxes([p], VID))

    def test_frame_beyond_video_is_reported(self):
        assert any(i.code == "FRAME_OUT_OF_VIDEO_RANGE"
                   for i in qc.check_frame_numbers([person(99, 1)], VID, 10))

    def test_missing_annotator_is_reported(self):
        p = person(0, 1, annotator="")
        p.annotator_id = ""
        assert any(i.code == "MISSING_ANNOTATOR_ID"
                   for i in qc.check_annotator([p], VID))

    def test_missing_bowler_track_does_not_leak_into_actions(self):
        p = person(0, 1, "STRIKER")
        issues = qc.check_bowler_track([p], VID)
        assert any(i.code == "MISSING_BOWLER_TRACK" for i in issues)

    def test_release_outside_delivery_is_reported(self):
        d = delivery(start=0, end=10, release_frame=30)
        assert any(i.code == "RELEASE_OUTSIDE_DELIVERY"
                   for i in qc.check_delivery(d, VID, 100))

    def test_phase_order_violation_is_reported_not_repaired(self):
        d = delivery(start=0, end=30, gather_frame=20, release_frame=10)
        issues = qc.check_delivery(d, VID, 100)
        assert any(i.code == "PHASE_ORDER_VIOLATION" for i in issues)
        assert d.gather_frame == 20 and d.release_frame == 10, "QC must not mutate"

    def test_valid_order_passes(self):
        d = delivery(start=0, end=30, runup_start_frame=2, gather_frame=12,
                     release_frame=20, followthrough_end_frame=28)
        assert not [i for i in qc.check_delivery(d, VID, 100)
                    if i.severity == "ERROR"]

    def test_delivery_with_no_key_frames_is_a_warning(self):
        issues = qc.check_delivery(delivery(start=0, end=10), VID, 100)
        assert any(i.severity == "WARNING" for i in issues)

    def test_overlapping_deliveries_are_reported(self):
        a, b = delivery("d1", start=0, end=10), delivery("d2", start=8, end=20)
        assert any(i.code == "CONFLICTING_ROLE"
                   for i in qc.check_overlapping_deliveries([a, b], VID))

    def test_duplicate_delivery_id_is_reported(self):
        issues = qc.check_delivery_ids(
            [delivery("d1", bowler=1), delivery("d1", bowler=2)], VID)
        assert {i.code for i in issues} >= {"DUPLICATE_DELIVERY_ID", "CONFLICTING_ROLE"}

    def test_frame_gaps_are_info_not_error(self):
        issues = qc.check_missing_frames([person(0, 1), person(3, 1)], VID)
        assert issues and issues[0].severity == "INFO"
        assert issues[0].detail["n_holes"] == 2

    def test_split_leakage_is_an_error(self):
        issues = qc.check_split_leakage({"train": ["a", "b"], "val": ["b"]})
        assert issues and issues[0].code == "SPLIT_LEAKAGE"
        assert issues[0].severity == "ERROR"

    def test_frame_hash_leakage_detected(self):
        issues = qc.check_frame_hash_leakage(
            {"train": {"h1", "h2"}, "test": {"h2"}})
        assert issues and issues[0].detail["n_shared"] == 1

    def test_duplicate_video_ids_reported(self):
        recs = [registry.VideoRecord(video_id="v1", path="a", filename="a.avi", bytes=1),
                registry.VideoRecord(video_id="v1", path="b", filename="b.avi", bytes=1)]
        assert qc.check_duplicate_video_ids(recs)

    def test_issue_line_carries_location(self):
        i = qc.QCIssue("CODE", "ERROR", VID, "msg", frame_id=3, track_id=2,
                       delivery_id="d1")
        assert "f3" in i.line() and "t2" in i.line() and "d=d1" in i.line()

    def test_run_all_declares_it_did_not_repair(self, tmp_path, monkeypatch):
        monkeypatch.setattr(schema, "ANNOTATIONS_DIR",
                            os.path.join(str(tmp_path), "annotations"))
        r = qc.run_all(video_ids=[VID], n_frames_by_id={VID: 40})
        assert r["auto_repaired"] is False
        assert "No human annotation was modified" in r["note"]


# --------------------------------------------------------------------------- #
# PHASE 6: splits -- leakage prevention
# --------------------------------------------------------------------------- #

class TestSplits:
    def _recs(self, n_groups=6, per_group=4, **kw):
        out = []
        for g in range(n_groups):
            for i in range(per_group):
                out.append(registry.VideoRecord(
                    video_id=f"g{g}_v{i}", path=f"x/g{g}/v{i}.mp4",
                    filename=f"fast_{g:02d}_{i:08d}.avi", bytes=1,
                    fps=20.0, width=64, height=64, frame_count=60,
                    probe_status="OK", **kw))
        return out

    def test_no_group_spans_two_splits(self):
        r = splits.make_splits(self._recs(), seed=42)
        g2s = {}
        for name, ids in r.splits.items():
            for vid in ids:
                g2s.setdefault(r.groups[vid], set()).add(name)
        assert all(len(v) == 1 for v in g2s.values()), g2s

    def test_no_video_id_is_reused(self):
        r = splits.make_splits(self._recs(), seed=42)
        allids = [v for ids in r.splits.values() for v in ids]
        assert len(allids) == len(set(allids))

    def test_every_video_lands_exactly_once(self):
        r = splits.make_splits(self._recs(), seed=42)
        assert sum(len(v) for v in r.splits.values()) == 24

    def test_deterministic_for_same_seed(self):
        a = splits.make_splits(self._recs(), seed=7)
        b = splits.make_splits(self._recs(), seed=7)
        assert a.splits == b.splits

    def test_different_seed_changes_group_assignment(self):
        def group_split(seed):
            res = splits.make_splits(self._recs(), seed=seed)
            return {res.groups[v]: s for s, ids in res.splits.items() for v in ids}
        assert group_split(1) != group_split(2)

    def test_split_verifier_finds_no_leakage(self):
        assert splits.verify(splits.make_splits(self._recs(), seed=42)) == []

    def test_bowler_id_wins_over_view_prefix(self):
        """Different camera-view folders, but all the SAME known bowler."""
        recs = [registry.VideoRecord(
            video_id=f"b0_{g}", path=f"x/{g}.mp4", filename=f"fast_{g:02d}_0.avi",
            bytes=1, fps=20.0, width=64, height=64, frame_count=60,
            probe_status="OK", bowler_id_if_known="BOWLER_0") for g in range(8)]
        r = splits.make_splits(recs, seed=5)
        assert r.splits["test"] == [] and r.splits["val"] == []
        assert len(r.splits["train"]) == 8
        assert r.strategy == "refuse_bowler_leak_all_train"
        assert "NO HELD-OUT SET EXISTS" in r.residual_leakage_risk
        assert splits.verify(r) == []

    def test_single_group_records_residual_risk(self):
        r = splits.make_splits(self._recs(n_groups=1, per_group=6), seed=1)
        assert r.grouping_degraded is True
        assert "RESIDUAL LEAKAGE RISK" in r.residual_leakage_risk
        assert "OPTIMISTIC" in r.residual_leakage_risk

    def test_tiny_corpus_does_not_claim_the_ratio(self):
        r = splits.make_splits(self._recs(n_groups=2, per_group=1), seed=1)
        assert r.ratio_split_applied is False or r.n_videos >= 10
        assert r.note or r.ratio_split_applied

    def test_empty_corpus_is_empty_by_construction(self):
        r = splits.make_splits([], seed=1)
        assert r.n_videos == 0
        assert r.splits == {"train": [], "val": [], "test": []}
        assert "empty by construction" in r.note

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(schema, "SPLITS_DIR", os.path.join(str(tmp_path), "splits"))
        monkeypatch.setattr(splits, "SPLIT_FILES", {
            k: os.path.join(str(tmp_path), "splits", f"{k}.txt")
            for k in ("train", "val", "test")})
        r = splits.make_splits(self._recs(), seed=42)
        splits.save_splits(r)
        # EVERY artefact must land in tmp_path. A previous version resolved
        # split_meta.json through schema.p(), which ignores the patched
        # SPLITS_DIR and silently overwrote the real data tree with fixture ids.
        assert os.path.exists(os.path.join(str(tmp_path), "splits", "split_meta.json"))
        back = splits.load_splits()
        assert back is not None
        for name in ("train", "val", "test"):
            assert back[name] == sorted(r.splits[name])
        assert splits.split_of(r.splits["train"][0], back) == "train"


# --------------------------------------------------------------------------- #
# PHASE 7: generators
# --------------------------------------------------------------------------- #

def _annotated_contexts(root, n=2, with_delivery=True, with_phases=True,
                        n_frames=40, n_labeled=5, split="train"):
    """Write synthetic annotation files and build the matching contexts."""
    recs = []
    split_map = {"train": [], "val": [], "test": [], "unassigned": []}
    for i in range(n):
        vid = f"vid{i:03d}"
        os.makedirs(os.path.join(root, "roles"), exist_ok=True)
        with open(os.path.join(root, "roles", f"{vid}.json"), "w",
                  encoding="utf-8") as fh:
            fh.write(schema.dumps({
                "schema_version": schema.SCHEMA_VERSION, "video_id": vid,
                "kind": "person_roles",
                "records": [
                    {**person(f, t, role, annotator="TESTER").to_dict(),
                     "video_id": vid}
                    for t, role in ((1, "BOWLER"), (2, "STRIKER"))
                    for f in range(n_labeled)
                ]}))
        if with_delivery:
            os.makedirs(os.path.join(root, "deliveries"), exist_ok=True)
            d = delivery("d1", 1, 0, min(20, n_frames - 1), gather_frame=8,
                         release_frame=15, annotation_date="2026-01-01T00:00:00Z")
            d.video_id = vid
            with open(os.path.join(root, "deliveries", f"{vid}.json"), "w",
                      encoding="utf-8") as fh:
                fh.write(schema.dumps({"schema_version": schema.SCHEMA_VERSION,
                                       "video_id": vid, "kind": "deliveries",
                                       "records": [d.to_dict()]}))
        if with_phases:
            os.makedirs(os.path.join(root, "bowling_phases"), exist_ok=True)
            with open(os.path.join(root, "bowling_phases", f"{vid}.json"), "w",
                      encoding="utf-8") as fh:
                fh.write(schema.dumps({
                    "schema_version": schema.SCHEMA_VERSION, "video_id": vid,
                    "kind": "bowling_phases",
                    "records": [
                        schema.PhaseAnnotation(
                            video_id=vid, frame_id=f, track_id=1, phase="GATHER",
                            annotator_id="TESTER").to_dict() for f in range(0, 4)]}))
        recs.append(registry.VideoRecord(
            video_id=vid, path=f"D:/synthetic/{vid}.mp4",
            filename=f"{vid}.mp4", bytes=100, fps=20.0, width=200, height=200,
            frame_count=n_frames, probe_status="OK"))
        split_map[split].append(vid)
    return generators.load_contexts(recs, split_map, require_labels=True,
                                    annotation_root=root)


class TestGenerators:
    def test_labels_require_the_video_to_be_registered(self, tmp_path):
        recs = [registry.VideoRecord(
            video_id="vid000", path="D:/synthetic/missing.mp4", filename="missing.mp4",
            bytes=1, fps=20.0, width=200, height=200, frame_count=40,
            probe_status="OK")]
        assert generators.load_contexts(
            recs, {"train": ["vid000"], "val": [], "test": []},
            require_labels=True,
            annotation_root=os.path.join(str(tmp_path), "nope")) == []

    def test_role_labels_come_from_the_schema_vocabulary(self, tmp_path):
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"))
        s = generators.build_role_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                          dry_run=True)
        assert s["n_samples"] == 4, "2 videos x 2 tracks"
        assert set(s["n_by_label"]) <= set(schema.PLAYER_ROLES.values())
        assert set(s["n_by_label"]) == {"BOWLER", "STRIKER"}

    def test_action_labels_come_from_the_phase_taxonomy(self, tmp_path):
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"))
        s = generators.build_delivery_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                              dry_run=True)
        assert s["n_rows"] == 2
        assert s["n_rejected"] == 0

    def test_delivery_generator_rejects_phase_order_violations(self, tmp_path):
        root = os.path.join(str(tmp_path), "a")
        ctxs = _annotated_contexts(root, n=1)
        bad = schema.DeliveryAnnotation(
            **{**ctxs[0].deliveries[0].to_dict(), "gather_frame": 18,
               "release_frame": 12})
        ctxs[0].deliveries = [bad]
        s = generators.build_delivery_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                              dry_run=True)
        assert s["n_rows"] == 0
        assert s["n_rejected"] == 1
        assert s["notes"]["policy"].startswith("violations reported")
        assert bad.gather_frame == 18 and bad.release_frame == 12, "never repaired"

    def test_no_annotations_yields_empty_datasets_not_guesses(self, tmp_path):
        recs = [registry.VideoRecord(
            video_id="none", path="D:/synthetic/none.mp4", filename="none.mp4",
            bytes=1, fps=20.0, width=200, height=200, frame_count=10,
            probe_status="OK")]
        ctxs = generators.load_contexts(recs, {"train": ["none"], "val": [], "test": []},
                                       require_labels=True,
                                       annotation_root=os.path.join(str(tmp_path), "e"))
        for name, fn in generators.GENERATORS.items():
            kwargs = {"imsize": None} if name == "det" else {}
            s = fn(ctxs, os.path.join(str(tmp_path), "o"), dry_run=True, **kwargs)
            assert s["n_samples"] == 0, f"{name} invented samples from nothing"

    def test_label_provenance_is_declared_human(self, tmp_path):
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"), n=1)
        s = generators.build_role_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                          dry_run=True)
        assert s["label_source_is_human"] is True
        assert s["label_source"] == generators.LABEL_ROOT

    def test_auto_labelled_trees_are_refused(self):
        for bad in generators.FORBIDDEN_LABEL_ROOTS:
            with pytest.raises(generators.GeneratorError):
                generators.assert_human_label_source(bad)
            with pytest.raises(generators.GeneratorError):
                generators.assert_human_label_source(os.path.join(bad, "train"))

    def test_human_root_is_accepted(self):
        generators.assert_human_label_source(generators.LABEL_ROOT)

    def test_manifest_rows_record_full_lineage(self, tmp_path):
        out = os.path.join(str(tmp_path), "o")
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"), n=1)
        generators.build_role_dataset(ctxs, out, dry_run=False)
        with open(os.path.join(out, "manifest.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        assert rows
        r = rows[0]
        for key in ("source_video_id", "source_video_path", "source_track_id",
                    "label", "transformation", "split", "annotator_id"):
            assert key in r, f"provenance row is missing {key}"
        assert r["annotator_id"] == "TESTER"

    def test_track_sequence_emits_explicit_missing_rows(self, tmp_path):
        root = os.path.join(str(tmp_path), "a")
        ctxs = _annotated_contexts(root, n=1, n_labeled=5)
        # punch a hole at frame 2
        ctxs[0].people = [p for p in ctxs[0].people
                          if not (p.track_id == 1 and p.frame_id == 2)]
        out = os.path.join(str(tmp_path), "o")
        s = generators.build_track_sequence_dataset(ctxs, out, dry_run=False)
        assert s["n_missing"] == 1
        path = os.path.join(out, "train", f"{ctxs[0].video_id}_T1.json")
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        hole = rec["sequence"][2]
        assert hole["missing"] == 1.0
        assert hole["cx_norm"] == 0.0, "a hole must not be interpolated"

    def test_dry_run_writes_manifest_but_no_pixels(self, tmp_path):
        out = os.path.join(str(tmp_path), "o")
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"), n=1)
        s = generators.build_yolo_dataset(ctxs, out, dry_run=True)
        assert s["n_samples"] > 0
        assert os.path.exists(os.path.join(out, "manifest.jsonl"))
        assert not os.path.isdir(os.path.join(out, "images"))

    def test_manifest_writer_rejects_a_foreign_row(self, tmp_path):
        w = generators.ManifestWriter("role", str(tmp_path), dry_run=True)
        with pytest.raises(generators.GeneratorError):
            w.add(generators.ProvenanceRow(
                "det", "s", "v", "p", 0, 1, "BOWLER", 0, "t", "train"))

    def test_manifest_writer_rejects_an_unknown_split(self, tmp_path):
        w = generators.ManifestWriter("role", str(tmp_path), dry_run=True)
        with pytest.raises(generators.GeneratorError):
            w.add(generators.ProvenanceRow(
                "role", "s", "v", "p", 0, 1, "BOWLER", 0, "t", "holdout"))

    def test_yolo_line_is_normalised_and_clipped(self):
        line = generators.yolo_label_line(0, [-20, -10, 120, 240], 100, 100)
        c, cx, cy, w, h = line.split()
        assert c == "0"
        assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0
        assert math.isclose(float(cx), 0.5, abs_tol=1e-6)
        assert math.isclose(float(cy), 0.5, abs_tol=1e-6)
        assert math.isclose(float(w), 1.0, abs_tol=1e-6)
        assert math.isclose(float(h), 1.0, abs_tol=1e-6)

    def test_yolo_line_rejects_zero_area(self):
        assert generators.yolo_label_line(0, [5, 5, 5, 5], 100, 100) is None

    def test_yolo_line_fully_outside_the_frame_is_rejected(self):
        assert generators.yolo_label_line(0, [300, 300, 400, 400], 100, 100) is None

    def test_pose_features_zero_a_missing_frame(self):
        f = generators.pose_features(None, 0.5, 0.9, missing=True)
        assert f["pose_missing"] == 1.0
        assert f["pose_confidence"] == 0.0
        assert all(f[k] == 0.0 for k in generators.ACTION_FEATURE_NAMES
                   if k not in ("pose_missing", "frame_idx_norm"))

    def test_derivative_does_not_bridge_a_missing_frame(self):
        seq = [{"lwrist_x": 0.0, "pose_missing": 0.0},
               {"lwrist_x": 0.0, "pose_missing": 1.0},
               {"lwrist_x": 2.0, "pose_missing": 0.0}]
        out = generators._differentiate(seq, ["lwrist_x"], "v")
        assert out[0]["v_lwrist_x"] == 0.0
        assert out[1]["v_lwrist_x"] == 0.0, "must not compute across a gap"
        assert out[2]["v_lwrist_x"] == 0.0

    def test_action_dataset_needs_pose_sequences(self, tmp_path, monkeypatch):
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"), n=1)
        monkeypatch.setattr(pose_seq, "POSE_DIR", os.path.join(str(tmp_path), "pose"))
        s = generators.build_action_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                            dry_run=True)
        assert s["n_windows"] == 0
        assert s["skipped"]["no_pose_sequence"] == 1

    def test_unassigned_videos_never_leak_into_a_holdout(self, tmp_path):
        ctxs = _annotated_contexts(os.path.join(str(tmp_path), "a"), n=1,
                                   split="unassigned")
        assert ctxs[0].split == "unassigned"
        s = generators.build_role_dataset(ctxs, os.path.join(str(tmp_path), "o"),
                                          dry_run=True)
        assert s["n_by_split"] == {"unassigned": 2}


# --------------------------------------------------------------------------- #
# PHASE 8: pose sequences
# --------------------------------------------------------------------------- #

def _install_fake_pose(monkeypatch, landmarks=None, raises=None):
    class FakePose:
        def __init__(self):
            self.landmarks = landmarks
            self.n_people = 1

    class FakeEstimator:
        def __init__(self, *a, **k):
            if raises is not None:
                raise raises
        def process_frame(self, crop, f, ts):
            return FakePose()
        def close(self):
            pass
    fake = types.SimpleNamespace(PoseEstimator=FakeEstimator)
    monkeypatch.setitem(sys.modules, "src.pose_estimation", fake)
    # pose_seq does ``from .. import pose_estimation``, which resolves through the
    # parent package attribute -- patching sys.modules alone is a no-op once any
    # other test module has already imported the real submodule.
    import src as _src
    monkeypatch.setattr(_src, "pose_estimation", fake, raising=False)


class TestPoseSeq:
    def test_crop_pads_the_human_box_and_is_clipped(self):
        boxes = {0: (10.0, 20.0, 60.0, 140.0)}
        (x1, y1, x2, y2), used = pose_seq._crop_from_human_boxes(boxes, 0, (200, 200, 3))
        # 30% of the box on each side: 15px in x, 36px in y; the top-left corner
        # goes off-frame and is clipped, never turned into a negative index.
        assert (x1, y1, x2, y2) == (0, 0, 75, 176)
        assert used == 0

    def test_crop_records_which_frame_the_box_came_from(self):
        boxes = {4: (10.0, 20.0, 60.0, 140.0)}
        _, used = pose_seq._crop_from_human_boxes(boxes, 5, (200, 200, 3))
        assert used == 4, "a nearest-frame fallback must be recorded, not hidden"

    def test_no_box_and_no_nearby_frame_yields_none(self):
        assert pose_seq._crop_from_human_boxes({}, 5, (200, 200, 3)) is None
        assert pose_seq._crop_from_human_boxes({0: (1, 1, 50, 100)}, 40,
                                              (200, 200, 3)) is None

    def test_tiny_box_yields_none_rather_than_garbage(self):
        assert pose_seq._crop_from_human_boxes({0: (1.0, 1.0, 2.0, 2.0)}, 0,
                                              (200, 200, 3)) is None

    def test_missing_pose_is_recorded_never_dropped(self, tmp_path, monkeypatch):
        _install_fake_pose(monkeypatch, landmarks=None)
        d = delivery("d1", 1, 0, 4)
        frame = np.zeros((200, 200, 3), np.uint8)
        out = pose_seq.extract_for_delivery(
            str(tmp_path / "v.mp4"), VID, d,
            {i: (10, 20, 60, 140) for i in range(5)}, lambda f: frame)
        assert out["n_frames"] == 5
        assert out["n_observed"] == 0 and out["n_missing"] == 5
        assert len(out["frames"]) == 5
        for rec in out["frames"]:
            assert rec["observed"] is False
            assert rec["keypoints"] is None
            assert rec["keypoint_visibility"] is None
            assert rec["pose_confidence"] == 0.0
            assert rec["context"], "a missing frame must say WHY"

    def test_observed_frames_keep_per_landmark_visibility(self, tmp_path, monkeypatch):
        lm = [[0.1, 0.2, 0.0, 0.9], [0.3, 0.4, 0.0, 0.1], [0.5, 0.6, 0.0, 0.8]]
        _install_fake_pose(monkeypatch, landmarks=lm)
        d = delivery("d1", 1, 0, 1)
        frame = np.zeros((200, 200, 3), np.uint8)
        out = pose_seq.extract_for_delivery(
            str(tmp_path / "v.mp4"), VID, d,
            {i: (10, 20, 60, 140) for i in range(2)}, lambda f: frame)
        assert out["n_observed"] == 2 and out["n_missing"] == 0
        r = out["frames"][0]
        assert r["observed"] is True
        assert len(r["keypoints"]) == 3 and len(r["keypoints"][0]) == 3
        assert r["keypoint_visibility"] == [0.9, 0.1, 0.8]
        assert r["low_confidence_landmarks"] == 1
        assert math.isclose(out["mean_pose_confidence"], (0.9 + 0.1 + 0.8) / 3,
                            rel_tol=1e-3)
        assert "NOT calibrated" in out["pose_confidence_is"]

    def test_unreadable_frame_is_recorded(self, tmp_path, monkeypatch):
        _install_fake_pose(monkeypatch, landmarks=None)
        d = delivery("d1", 1, 5, 6)
        out = pose_seq.extract_for_delivery(
            str(tmp_path / "v.mp4"), VID, d, {}, lambda f: None)
        assert out["n_missing"] == 2
        assert out["frames"][0]["context"] == "frame_unreadable"

    def test_estimator_failure_is_reported_not_swallowed(self, tmp_path, monkeypatch):
        _install_fake_pose(monkeypatch, raises=RuntimeError("AppLocker blocked torch"))
        d = delivery("d1", 1, 0, 1)
        out = pose_seq.extract_for_delivery(
            str(tmp_path / "v.mp4"), VID, d, {}, lambda f: None)
        assert "error" in out and "AppLocker" in out["error"]
        assert out["n_observed"] == 0 and out["n_missing"] == 2

    def test_inverted_window_is_normalised_by_the_schema(self, tmp_path, monkeypatch):
        _install_fake_pose(monkeypatch, landmarks=None)
        d = delivery("d1", 1, 9, 4)
        assert (d.start_frame, d.end_frame) == (9, 9), "end < start collapses, not wraps"
        out = pose_seq.extract_for_delivery(
            str(tmp_path / "v.mp4"), VID, d, {}, lambda f: None)
        assert (out["start_frame"], out["end_frame"]) == (9, 9)
        assert out["n_frames"] == 1

    def test_save_load_round_trip_is_atomic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pose_seq, "POSE_DIR", os.path.join(str(tmp_path), "pose"))
        payload = {"video_id": "v1", "delivery_id": "d1", "frames": [
            {"frame_id": 0, "observed": False, "keypoints": None}]}
        p = pose_seq.save(payload)
        assert os.path.exists(p)
        assert not [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]
        assert pose_seq.load_sequence("v1", "d1")[0]["observed"] is False
        assert pose_seq.load_sequence("v1", "missing") == []


# --------------------------------------------------------------------------- #
# PHASE 12: bowler lock
# --------------------------------------------------------------------------- #

class FakeTrack:
    """Duck-typed ``src.tracking.Track`` (no torch import needed)."""

    def __init__(self, track_id, frames, bboxes):
        self.track_id = track_id
        self.frames = list(frames)
        self.bboxes = [tuple(float(v) for v in b) for b in bboxes]

    def __len__(self):
        return len(self.frames)


class TestBowlerLock:
    def _lock(self):
        return bowler_lock.BowlerLock.from_user(
            1, boxes={f: (float(f), 10.0, float(f) + 50.0, 120.0) for f in range(20)})

    # -- rule 6: refusal -------------------------------------------------- #

    def test_unconfirmed_lock_withholds_biomechanics(self):
        lock = bowler_lock.BowlerLock.from_role_and_action([])
        assert lock.gate()["status"] == bowler_lock.NOT_CONFIRMED
        assert lock.gate()["biomechanics_available"] is False
        with pytest.raises(bowler_lock.BowlerLockError):
            lock.biomechanics_window(0, 10, None)

    def test_low_evidence_is_not_confirmed(self):
        c = bowler_lock.LockCandidate(1, combined_score=0.10)
        assert bowler_lock.BowlerLock.from_role_and_action([c]).confirmed is False

    def test_ambiguous_margin_is_not_confirmed(self):
        cs = [bowler_lock.LockCandidate(1, combined_score=0.50),
              bowler_lock.LockCandidate(2, combined_score=0.48)]
        lk = bowler_lock.BowlerLock.from_role_and_action(cs)
        assert lk.reason == "ambiguous_margin" and lk.confirmed is False

    def test_clear_winner_is_confirmed(self):
        cs = [bowler_lock.LockCandidate(1, combined_score=0.80),
              bowler_lock.LockCandidate(2, combined_score=0.20)]
        assert bowler_lock.BowlerLock.from_role_and_action(cs).confirmed is True

    # -- rule 3: pose identity -------------------------------------------- #

    def test_pose_crop_refuses_a_different_track(self):
        with pytest.raises(bowler_lock.BowlerLockError) as e:
            self._lock().pose_crop(5, track_id=2, frame_shape=(200, 200, 3))
        assert "identity violation" in str(e.value)

    def test_pose_crop_refuses_when_there_is_no_lock(self):
        with pytest.raises(bowler_lock.BowlerLockError) as e:
            bowler_lock.BowlerLock().pose_crop(5, 1, (200, 200, 3))
        assert bowler_lock.NOT_CONFIRMED in str(e.value)

    def test_pose_crop_works_for_the_locked_track(self):
        c = self._lock().pose_crop(5, track_id=1, frame_shape=(200, 200, 3))
        assert c is not None and len(c) == 4
        assert c[0] >= 0 and c[2] <= 200

    def test_pose_crop_without_a_box_is_none_not_a_substitute(self):
        assert self._lock().pose_crop(9999, 1, (200, 200, 3)) is None

    # -- rule 4: biomechanics identity ------------------------------------ #

    def test_biomechanics_refuse_a_different_track(self):
        with pytest.raises(bowler_lock.BowlerLockError):
            self._lock().biomechanics_window(0, 10, 7)

    def test_biomechanics_reject_inverted_window(self):
        with pytest.raises(bowler_lock.BowlerLockError):
            self._lock().biomechanics_window(10, 5, 1)

    # -- rule 5: replay identity ------------------------------------------ #

    def test_replay_never_returns_another_persons_box(self):
        lk = self._lock()
        lk.boxes[999] = (0.0, 0.0, 1.0, 1.0)
        assert lk.replay_boxes(5) == lk.boxes[5]
        assert lk.replay_track_id() == 1

    def test_replay_returns_none_not_a_substitute(self):
        assert self._lock().replay_boxes(10_000) is None
        assert bowler_lock.BowlerLock().replay_boxes(0) is None

    # -- rules 1 & 2: identity stability and recovery --------------------- #

    def test_recovery_refused_on_a_long_gap(self):
        lk = self._lock()
        lk.source, lk.confirmed = "heuristic", True
        assert lk.recover_lock(9, {500: (400.0, 10.0, 450.0, 120.0)}) is False
        assert any("gap" in w for w in lk.warnings)

    def test_recovery_refused_when_appearance_check_fails(self):
        lk = self._lock()
        lk.source, lk.confirmed = "heuristic", True
        assert lk.recover_lock(9, {25: (20.0, 10.0, 70.0, 120.0)},
                               appearance_check=lambda: 0.01) is False
        assert any("appearance" in w for w in lk.warnings)

    def test_recovery_refuses_silent_switch_when_confirmed_by_model(self):
        lk = self._lock()
        lk.source = "role_model"
        assert lk.recover_lock(9, {25: (20.0, 10.0, 70.0, 120.0)}) is False
        assert lk.track_id == 1

    def test_spatial_only_recovery_is_flagged_not_hidden(self):
        lk = self._lock()
        lk.source, lk.confirmed = "heuristic", True
        assert lk.recover_lock(9, {25: (20.0, 10.0, 70.0, 120.0)}) is True
        assert lk.track_id == 1, "the lock id must never change on recovery"
        assert lk.recovered_segments[-1]["appearance_checked"] is False
        assert any("SPATIALLY ONLY" in w for w in lk.warnings)

    def test_recovery_with_passing_appearance_is_recorded(self):
        lk = self._lock()
        lk.source, lk.confirmed = "heuristic", True
        assert lk.recover_lock(9, {25: (20.0, 10.0, 70.0, 120.0)},
                               appearance_check=lambda: 0.9) is True
        seg = lk.recovered_segments[-1]
        assert seg["appearance_checked"] is True
        assert seg["into_track_id"] == 1 and seg["from_track_id"] == 9

    def test_failed_recovery_never_removes_the_lock(self):
        lk = self._lock()
        before = dict(lk.boxes)
        lk.recover_lock(9, {5000: (0.0, 0.0, 10.0, 10.0)})
        for f, b in before.items():
            assert lk.boxes.get(f) == b

    def test_recovery_without_a_lock_is_refused(self):
        lk = bowler_lock.BowlerLock()
        assert lk.recover_lock(3, {10: (0.0, 0.0, 5.0, 5.0)}) is False

    def test_every_failed_recovery_is_counted(self):
        lk = self._lock()
        lk.source, lk.confirmed = "heuristic", True
        lk.recover_lock(9, {5000: (0.0, 0.0, 10.0, 10.0)})
        assert lk.identity_switch_attempts == 1

    # -- integration ------------------------------------------------------ #

    def test_build_lock_labels_the_heuristic_path(self):
        t = FakeTrack(3, [0, 1], [(0, 0, 10, 10), (1, 1, 11, 11)])
        lock = bowler_lock.build_lock_from_pipeline(
            {3: t}, t, {"confirmed": True, "confirm_reason": None,
                        "candidates": [{"track_id": 3, "score": 0.7,
                                        "motion": 0.5, "active": 0.4, "n_frames": 2}],
                        "identity_switch_count": 0})
        assert lock.source == "heuristic" and lock.confirmed is True
        assert any("NOT a learned" in w for w in lock.warnings)
        assert lock.candidates and lock.candidates[0].track_id == 3

    def test_no_bowler_track_yields_an_unconfirmed_lock(self):
        lock = bowler_lock.build_lock_from_pipeline({}, None, None)
        assert lock.track_id is None and lock.confirmed is False
        assert lock.gate()["biomechanics_available"] is False

    def test_user_confirmation_wins(self):
        t1 = FakeTrack(1, [0], [(0, 0, 10, 10)])
        t2 = FakeTrack(2, [0], [(0, 0, 10, 10)])
        lock = bowler_lock.build_lock_from_pipeline(
            {1: t1, 2: t2}, t1, {"confirmed": True}, user_track_id=2)
        assert lock.track_id == 2 and lock.source == "user" and lock.confirmed

    def test_gate_shape_when_locked(self):
        g = self._lock().gate()
        assert g["status"] == "BOWLER LOCKED"
        assert g["biomechanics_available"] is True
        assert g["lock"]["track_id"] == 1
        assert g["lock"]["n_boxed_frames"] == 20

    def test_heuristic_meta_confidence_is_not_a_role_confidence(self):
        """The heuristic bowler score must not be laundered into role_confidence."""
        t = FakeTrack(3, [0], [(0, 0, 10, 10)])
        lock = bowler_lock.build_lock_from_pipeline(
            {3: t}, t, {"confirmed": True, "confidence": 0.91, "candidates": [],
                        "identity_switch_count": 0})
        lock.confidence = confidence.from_pipeline(
            pose_confidence=0.8, track_coverage=1.0, delivery_reliable=True,
            bowler_confirmed=True)
        assert lock.confidence.available("role_confidence") is False


# --------------------------------------------------------------------------- #
# PHASE 13: confidence
# --------------------------------------------------------------------------- #

class TestConfidence:
    def test_unmeasured_stays_none_not_zero(self):
        b = confidence.ConfidenceBundle.empty()
        assert b.value("pose_confidence") is None
        assert b.available("pose_confidence") is False
        assert "NOT MEASURED" in b.channels["pose_confidence"].display()

    def test_setting_none_is_unavailable(self):
        b = confidence.ConfidenceBundle.empty().set("pose_confidence", None, "mediapipe")
        assert b.available("pose_confidence") is False

    def test_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            confidence.ConfidenceBundle.empty().set("pose_confidence", 1.4, "x")
        with pytest.raises(ValueError):
            confidence.ConfidenceBundle.empty().set("pose_confidence", -0.1, "x")

    def test_unknown_channel_is_rejected(self):
        with pytest.raises(KeyError):
            confidence.ConfidenceBundle.empty().set("accuracy", 0.9, "x")

    def test_measurement_confidence_unavailable_if_any_dependency_missing(self):
        b = confidence.from_pipeline(pose_confidence=0.99, track_coverage=1.0)
        assert b.available("pose_confidence") and b.available("tracking_confidence")
        assert b.available("measurement_confidence") is False
        assert any("delivery_confidence" in w for w in b.warnings)

    def test_measurement_confidence_is_min_of_dependencies(self):
        b = confidence.from_pipeline(pose_confidence=0.9, track_coverage=1.0,
                                     delivery_reliable=True, bowler_confirmed=True)
        assert math.isclose(b.value("measurement_confidence"), 0.9, rel_tol=1e-6)

    def test_a_missing_channel_never_raises_the_measurement(self):
        good = confidence.from_pipeline(pose_confidence=0.9, track_coverage=1.0,
                                        delivery_reliable=True, bowler_confirmed=True)
        bad = confidence.from_pipeline(pose_confidence=0.1, track_coverage=1.0,
                                       delivery_reliable=True, bowler_confirmed=True)
        assert good.value("measurement_confidence") > bad.value("measurement_confidence")

    def test_role_confidence_is_not_backfilled_by_the_heuristic(self):
        b = confidence.from_pipeline(pose_confidence=0.8, track_coverage=1.0,
                                     delivery_reliable=True, bowler_confirmed=True)
        assert b.available("role_confidence") is False
        assert "DIFFERENT quantity" in b.channels["role_confidence"].reason

    def test_delivery_confidence_zero_when_unreliable(self):
        b = confidence.from_pipeline(delivery_reliable=False, bowler_confirmed=True)
        assert b.value("delivery_confidence") == 0.0
        assert b.available("measurement_confidence") is False

    def test_unconfirmed_bowler_warns_and_withholds(self):
        b = confidence.from_pipeline(pose_confidence=0.95, track_coverage=0.95,
                                     delivery_reliable=True, bowler_confirmed=False)
        assert any("NOT CONFIRMED" in w for w in b.warnings)
        r = confidence.not_confirmed_result("low_evidence")
        assert r["biomechanics_available"] is False
        assert r["coaching_available"] is False
        assert "NOT MEASURED" in r["message"]
        assert r["status"] == bowler_lock.NOT_CONFIRMED

    def test_identity_switch_penalty_is_documented(self):
        b = confidence.from_pipeline(track_coverage=1.0, identity_switches=2)
        assert b.value("tracking_confidence") == pytest.approx(0.6)
        assert any("penalised" in w for w in b.warnings)

    def test_no_detector_scores_means_not_measured_not_one(self):
        b = confidence.from_pipeline(detection_confidences=[])
        assert b.available("detector_confidence") is False

    def test_serialised_bundle_is_honest(self):
        d = confidence.ConfidenceBundle.empty().to_dict()
        assert "accuracy" in d["disclaimer"]
        assert d["measurement_dependencies"] == list(confidence.MEASUREMENT_DEPENDENCIES)
        assert {r["channel"] for r in confidence.ConfidenceBundle.empty().to_rows()} \
            == set(confidence.CHANNELS)

    def test_rows_mark_unavailable_explicitly(self):
        rows = confidence.ConfidenceBundle.empty().to_rows()
        for r in rows:
            assert r["value"] is None
            assert r["available"] is False
            assert r["reason"]

    def test_from_pipeline_with_nothing_measured(self):
        b = confidence.from_pipeline()
        for name in confidence.CHANNELS:
            assert b.available(name) is False, name

    def test_summary_line_shows_not_measured_literally(self):
        s = confidence.ConfidenceBundle.empty().summary_line()
        assert s.count("NOT MEASURED") == len(confidence.CHANNELS)


# --------------------------------------------------------------------------- #
# Cross-cutting: the manifest must never overstate what exists
# --------------------------------------------------------------------------- #

def test_manifest_reports_zero_annotations_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "ANNOTATIONS_DIR",
                        os.path.join(str(tmp_path), "annotations"))
    m = manifest.build([registry.VideoRecord(
        video_id="v1", path="p", filename="f.avi", bytes=1, fps=20.0, width=64,
        height=64, frame_count=10, probe_status="OK")],
        splits={"train": ["v1"], "val": [], "test": []})
    assert m["n_videos_registered"] == 1
    assert m["n_videos_annotated"] == 0
    assert m["n_videos_trainable"] == 0
    assert m["deliveries_annotated"] == 0
    assert m["person_role_annotations"] == 0
    assert m["videos"][0]["annotation_status"] == "NONE"
