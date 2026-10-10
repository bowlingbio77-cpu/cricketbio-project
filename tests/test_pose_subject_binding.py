"""Tests for pose subject binding.

Regression coverage for the second identity decision described in
``bowler_identity_audit.md``: :func:`src.pose_estimation._primary_person` chose
the analysed subject by largest landmark box, ignoring the locked bowler track.
On side-on broadcast footage the nearest, largest person is the batsman, so the
pipeline could report a locked bowler while measuring a different human.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation import (
    _iou,
    _landmark_bbox,
    _largest_area_index,
    _primary_person,
    _select_subject,
)


def _person(cx, cy, w, h):
    """Synthetic 33-landmark person filling an exact rectangle.

    Points are laid on the rectangle perimeter *including the corners*, so the
    landmark bounding box is exactly ``(cx-w/2, cy-h/2, cx+w/2, cy+h/2)`` and
    overlap assertions can be tight.
    """
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    lm = []
    for i in range(33):
        t = i / 32.0 * 4.0  # walk the perimeter
        edge = int(t)
        frac = t - edge
        ax, ay = corners[edge % 4]
        bx, by = corners[(edge + 1) % 4]
        lm.append(type("L", (), {"x": float(ax + (bx - ax) * frac),
                                 "y": float(ay + (by - ay) * frac)})())
    return lm


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

class TestGeometryHelpers:
    def test_landmark_bbox(self):
        lm = _person(0.5, 0.5, 0.2, 0.4)
        x1, y1, x2, y2 = _landmark_bbox(lm)
        assert (x1, y1, x2, y2) == pytest.approx((0.4, 0.3, 0.6, 0.7))

    def test_iou_of_identical_boxes_is_one(self):
        box = (0.0, 0.0, 0.2, 0.4)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert _iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_iou_half_overlap(self):
        a = (0.0, 0.0, 0.2, 0.2)
        b = (0.1, 0.0, 0.3, 0.2)
        # intersection 0.1*0.2 = 0.02 ; union 0.04 + 0.04 - 0.02 = 0.06
        assert _iou(a, b) == pytest.approx(0.02 / 0.06)


# --------------------------------------------------------------------------- #
# Legacy behaviour preserved
# --------------------------------------------------------------------------- #

class TestLegacyPrimaryPerson:
    def test_picks_largest_bbox(self):
        small = _person(0.2, 0.2, 0.05, 0.12)
        mid = _person(0.5, 0.5, 0.10, 0.25)
        big = _person(0.7, 0.6, 0.25, 0.60)
        assert _primary_person([small, mid, big]) == 2

    def test_single_person(self):
        assert _primary_person([_person(0.5, 0.5, 0.2, 0.5)]) == 0

    def test_empty_is_safe(self):
        assert _largest_area_index([]) == 0
        assert _select_subject([]) == (0, "none")


# --------------------------------------------------------------------------- #
# Subject binding: the actual fix
# --------------------------------------------------------------------------- #

class TestSubjectBinding:
    def _broadcast_scene(self):
        """A small distant bowler plus a huge near-camera batsman.

        This is the measured failure: the batsman dominates by area, so the
        legacy heuristic picks the wrong human every time.
        """
        bowler = _person(cx=0.72, cy=0.50, w=0.03, h=0.10)
        batsman = _person(cx=0.10, cy=0.50, w=0.25, h=0.62)
        return [batsman, bowler]  # batsman listed first, as MediaPipe may return

    def test_legacy_picks_the_wrong_person(self):
        """Confirms the bug this change fixes still existed without a lock."""
        people = self._broadcast_scene()
        assert _primary_person(people) == 0  # the batsman

    def test_locked_region_selects_the_bowler(self):
        people = self._broadcast_scene()
        # Region in pixel coords of a 1280x720 frame around the bowler.
        subject = (1280 * 0.705, 720 * 0.44, 1280 * 0.735, 720 * 0.56)
        idx, method = _select_subject(
            people, subject_bbox=subject, frame_size=(720, 1280))
        assert idx == 1
        assert method == "subject_overlap"

    def test_locked_region_beats_a_much_larger_person(self):
        """The whole point: identity outranks size."""
        small_bowler = _person(0.80, 0.50, 0.02, 0.06)
        huge_batsman = _person(0.15, 0.50, 0.40, 0.90)
        subject = (1280 * 0.79, 720 * 0.46, 1280 * 0.81, 720 * 0.54)
        idx, method = _select_subject(
            [huge_batsman, small_bowler],
            subject_bbox=subject, frame_size=(720, 1280),
        )
        assert idx == 1
        assert method == "subject_overlap"

    def test_uses_padded_region_to_tolerate_bbox_drift(self):
        """A slightly-off region should still bind, not fall back."""
        bowler = _person(0.72, 0.50, 0.03, 0.10)
        batsman = _person(0.10, 0.50, 0.25, 0.62)
        # Region offset from the bowler but still overlapping it.
        subject = (1280 * 0.68, 720 * 0.40, 1280 * 0.78, 720 * 0.60)
        idx, method = _select_subject(
            [batsman, bowler], subject_bbox=subject, frame_size=(720, 1280))
        assert idx == 1
        assert method == "subject_overlap"

    def test_reports_fallback_when_nobody_overlaps(self):
        """An empty region must be visible, not silently scored by size."""
        bowler = _person(0.72, 0.50, 0.03, 0.10)
        batsman = _person(0.10, 0.50, 0.25, 0.62)
        # Region on empty grass.
        subject = (1280 * 0.45, 720 * 0.05, 1280 * 0.55, 720 * 0.15)
        idx, method = _select_subject(
            [bowler, batsman], subject_bbox=subject, frame_size=(720, 1280))
        assert method == "largest_area_fallback"
        # The fallback is the old behaviour, and it is the WRONG person here.
        assert idx == 1  # batsman, the large one

    def test_no_region_reports_unconstrained(self):
        people = self._broadcast_scene()
        idx, method = _select_subject(people, subject_bbox=None, frame_size=(720, 1280))
        assert method == "largest_area"
        assert idx == 0

    def test_degenerate_frame_size_falls_back_safely(self):
        people = self._broadcast_scene()
        idx, method = _select_subject(
            people, subject_bbox=(0, 0, 10, 10), frame_size=(0, 0))
        assert method == "largest_area"
        assert idx == 0

    def test_wrong_coordinate_space_is_detectable(self):
        """Passing full-frame coords to a cropped frame yields a fallback.

        This is why the pipeline maps the region per frame rather than reusing one
        global box: a mismatch must surface as "largest_area_fallback", never as
        a confident wrong pick.
        """
        bowler = _person(0.50, 0.50, 0.10, 0.30)
        batsman = _person(0.10, 0.50, 0.25, 0.62)
        # Full-frame coords (bowler at x~900) applied to a 200px-wide crop.
        subject = (900.0, 300.0, 960.0, 430.0)
        idx, method = _select_subject(
            [batsman, bowler], subject_bbox=subject, frame_size=(720, 200))
        assert method == "largest_area_fallback"


# --------------------------------------------------------------------------- #
# PoseEstimator plumbing
# --------------------------------------------------------------------------- #

class TestPoseFrameCarriesSelection:
    def test_poseframe_defaults_to_unconstrained(self):
        from src.pose_estimation import PoseFrame

        pf = PoseFrame(0, 0.0, np.zeros((33, 4)))
        assert pf.subject_selection == "largest_area"
        assert pf.n_people == 1

    def test_poseframe_accepts_selection(self):
        from src.pose_estimation import PoseFrame

        pf = PoseFrame(0, 0.0, np.zeros((33, 4)), None, 2, "subject_overlap")
        assert pf.subject_selection == "subject_overlap"
        assert pf.n_people == 2

    def test_process_video_frames_threads_regions(self, monkeypatch):
        """The region must be looked up per frame and passed through."""
        from src import pose_estimation

        estimator = pose_estimation.PoseEstimator.__new__(
            pose_estimation.PoseEstimator
        )
        seen = []

        def fake_process_frame(frame, idx, ts, subject_bbox=None):
            seen.append((idx, subject_bbox))
            return None

        monkeypatch.setattr(estimator, "process_frame", fake_process_frame)

        frames = [(0, 0.0, np.zeros((10, 10, 3), dtype=np.uint8)),
                  (1, 0.1, np.zeros((10, 10, 3), dtype=np.uint8)),
                  (2, 0.2, np.zeros((10, 10, 3), dtype=np.uint8))]
        boxes = {0: (1.0, 1.0, 5.0, 5.0), 2: (2.0, 2.0, 6.0, 6.0)}
        estimator.process_video_frames(iter(frames), subject_box_by_frame=boxes)

        assert seen[0] == (0, (1.0, 1.0, 5.0, 5.0))
        assert seen[1] == (1, None)  # no region for this frame
        assert seen[2] == (2, (2.0, 2.0, 6.0, 6.0))

    def test_process_video_frames_without_regions(self, monkeypatch):
        from src import pose_estimation

        estimator = pose_estimation.PoseEstimator.__new__(
            pose_estimation.PoseEstimator
        )
        seen = []
        monkeypatch.setattr(
            estimator, "process_frame",
            lambda f, i, t, subject_bbox=None: seen.append(subject_bbox),
        )
        frames = [(0, 0.0, np.zeros((10, 10, 3), dtype=np.uint8))]
        estimator.process_video_frames(iter(frames))
        assert seen == [None]
