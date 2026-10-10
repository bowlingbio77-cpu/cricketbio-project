"""Tests for the bowler-identity ground-truth builder.

The yardstick that every reported bowler-identification number depends on must
not regress silently: the previous policy was a ``"bowler" in raw_lower``
substring test (``evaluation/single_video_benchmark/run_validation.py:155-156``)
that could not distinguish ``bowler`` from ``non_striker``-style labels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GT_SCRIPT = REPO / "evaluation" / "build_bowler_gt.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_bowler_gt", GT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _box(x1, y1, x2, y2):
    return [x1, y1, x2, y2, 0]


def _write_annotation(tmp_path, person, event=None, name="vid_2overs.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"person": person, "event": event or {}}, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


class TestCleanBox:
    def test_accepts_five_value_box(self):
        assert mod._clean_box([10, 20, 30, 60, 0]) == [10.0, 20.0, 30.0, 60.0]

    def test_normalises_reversed_corners(self):
        assert mod._clean_box([30, 60, 10, 20, 0]) == [10.0, 20.0, 30.0, 60.0]

    @pytest.mark.parametrize("raw", [None, [], [1, 2], "nope", [1, 2, 3, "x", 0]])
    def test_rejects_malformed(self, raw):
        assert mod._clean_box(raw) is None

    def test_rejects_degenerate_boxes(self):
        assert mod._clean_box([10, 20, 10, 20, 0]) is None  # zero area
        assert mod._clean_box([10, 20, 10.5, 80, 0]) is None  # too thin
        assert mod._clean_box([10, 20, 90, 20.4, 0]) is None  # too short


class TestGroupDeliveries:
    def test_groups_contiguous_frames(self):
        assert mod._group_deliveries([1, 2, 3, 7, 8], gap=2) == [[1, 2, 3], [7, 8]]

    def test_splits_on_large_gap(self):
        assert mod._group_deliveries([1, 2, 90], gap=2) == [[1, 2], [90]]

    def test_empty_input(self):
        assert mod._group_deliveries([], gap=2) == []


class TestPerDeliveryPolicy:
    def test_resolves_two_bowlers_across_two_overs(self, tmp_path):
        """Two bowlers in one video must NOT collapse into a single dominant id.

        This is the core defect the policy fixes: a whole-video dominant id
        scored only ~35% of bowler-role frames.
        """
        person, event = {}, {}
        # Over 1: bowler "A" delivers on frames 0-4.
        for f in range(0, 5):
            person[str(f)] = [
                {"id": "A", "role": "bowler", "box": _box(10, 10, 40, 90)},
                {"id": "B", "role": "batsman", "box": _box(200, 10, 260, 95)},
            ]
            event[str(f)] = [{"event": "is bowling"}]
        # Over 2: bowler "Z" delivers on frames 20-24.
        for f in range(20, 25):
            person[str(f)] = [
                {"id": "Z", "role": "bowler", "box": _box(300, 10, 330, 90)},
                {"id": "B", "role": "batsman", "box": _box(200, 10, 260, 95)},
            ]
            event[str(f)] = [{"event": "is bowling"}]

        built = mod.build_for_video(_write_annotation(tmp_path, person, event))

        assert "error" not in built
        assert built["resolved_bowler_ids"] == 2
        assert built["detected_deliveries"] == 2
        # Both bowlers are scored, not just the globally dominant one.
        assert built["action_window_frames"] == 10
        assert set(built["bowler_bboxes"]) == {str(f) for f in list(range(0, 5)) + list(range(20, 25))}
        # Frame 0 belongs to bowler A, frame 20 belongs to bowler Z.
        assert built["bowler_bboxes"]["0"][0] == 10.0
        assert built["bowler_bboxes"]["20"][0] == 300.0

    def test_excludes_fragmented_frames_rather_than_scoring_them_as_misses(self, tmp_path):
        person = {
            "0": [{"id": "A", "role": "bowler", "box": _box(10, 10, 40, 90)}],
            "1": [{"id": "Q", "role": "bowler", "box": _box(500, 10, 530, 90)}],
        }
        event = {"0": [{"event": "is bowling"}], "1": [{"event": "is bowling"}]}

        built = mod.build_for_video(_write_annotation(tmp_path, person, event))

        assert built["action_window_frames"] == 1
        assert built["fragmented_frames_excluded"] == 1
        assert "1" not in built["bowler_bboxes"]

    def test_ignores_non_bowler_roles(self, tmp_path):
        """`non_striker` must never satisfy a bowler query.

        Guards the exact defect of the old `"bowler" in raw_lower` substring test.
        """
        person = {
            str(f): [
                {"id": "N", "role": "non_striker", "box": _box(10, 10, 40, 90)},
                {"id": "W", "role": "wicketkeeper", "box": _box(100, 10, 130, 90)},
            ]
            for f in range(3)
        }
        event = {str(f): [{"event": "is bowling"}] for f in range(3)}

        built = mod.build_for_video(_write_annotation(tmp_path, person, event))
        assert "error" in built

    def test_is_case_and_whitespace_insensitive_on_role(self, tmp_path):
        person = {
            "0": [{"id": "A", "role": "  Bowler ", "box": _box(10, 10, 40, 90)}],
        }
        built = mod.build_for_video(
            _write_annotation(tmp_path, person, {"0": [{"event": "is bowling"}]})
        )
        assert built["action_window_frames"] == 1


class TestDominantIdFallback:
    def test_whole_video_policy_uses_single_id(self, tmp_path):
        person = {
            str(f): [{"id": "A", "role": "bowler", "box": _box(10, 10, 40, 90)}]
            for f in range(4)
        }
        for f in (10, 11):
            person[str(f)] = [{"id": "Z", "role": "bowler", "box": _box(300, 10, 330, 90)}]
        event = {str(f): [{"event": "is bowling"}] for f in range(6)}

        built = mod.build_for_video(
            _write_annotation(tmp_path, person, event), policy="DOMINANT_ID"
        )
        assert built["identity_policy"] == "DOMINANT_ID"
        # A wins on total frames (4 > 2), so the two Z frames are excluded.
        assert built["action_window_frames"] == 4
        assert built["fragmented_frames_excluded"] == 2


class TestManifestIntegrity:
    def test_shipped_manifest_is_consistent(self):
        """If a committed manifest exists, its totals must add up.

        Guards against a report quoting a number the artifacts do not support.
        """
        manifest_path = (
            REPO / "evaluation" / "bowler_identification" / "ground_truth" / "manifest.json"
        )
        if not manifest_path.exists():
            pytest.skip("ground truth not built; run evaluation/build_bowler_gt.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["identity_policy"] in ("PER_DELIVERY", "DOMINANT_ID")
        assert "not_biomechanics_ground_truth" in manifest
        per_video = manifest["per_video"]
        assert len(per_video) == manifest["videos"]
        assert sum(v["action_window_frames"] for v in per_video) == manifest["total_scored_frames"]
        assert (
            sum(v["fragmented_frames_excluded"] for v in per_video)
            == manifest["total_fragmented_excluded"]
        )

    def test_every_scored_frame_has_a_valid_box(self):
        gt_dir = REPO / "evaluation" / "bowler_identification" / "ground_truth"
        files = sorted(gt_dir.glob("*_2overs.json")) if gt_dir.exists() else []
        if not files:
            pytest.skip("ground truth not built; run evaluation/build_bowler_gt.py")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["bowler_bboxes"], f"{path.name} has no scored frames"
            for frame, box in data["bowler_bboxes"].items():
                assert len(box) == 4, f"{path.name} frame {frame} malformed"
                x1, y1, x2, y2 = box
                assert x2 > x1 and y2 > y1, f"{path.name} frame {frame} inverted box"
