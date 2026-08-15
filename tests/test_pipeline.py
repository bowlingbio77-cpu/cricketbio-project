"""Tests for the orchestration layer: bowler-crop helpers and the in-memory
tracking path (IoU fallback), which run without ultralytics/MediaPipe."""
import numpy as np
import pytest

from src import pipeline, tracking
from src.tracking import Track


def _make_frame(h=90, w=160, value=0):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_crop_to_bbox_expands_and_clamps():
    frame = _make_frame()
    # bbox covering a person in the middle; crop must be inside frame bounds
    cut = pipeline._crop_to_bbox(frame, (60, 30, 120, 80))
    assert cut is not None
    # pad_frac 0.3: 60px-wide bbox gets 18px each side -> 96px wide
    assert cut.shape[1] == 96
    # vertical pad (15px each side) clamps to the 90px-tall frame
    assert cut.shape[0] == 75


def test_crop_to_bbox_clamps_to_frame_edges():
    frame = _make_frame()
    cut = pipeline._crop_to_bbox(frame, (0, 0, 40, 40))
    assert cut is not None
    assert cut.shape[1] <= 160 and cut.shape[0] <= 90


def test_crop_to_bbox_rejects_tiny_bbox():
    frame = _make_frame()
    assert pipeline._crop_to_bbox(frame, (10, 10, 12, 12)) is None


def test_crop_frames_to_bowler_carries_bbox_forward():
    frames = [(0, 0.0, _make_frame()), (1, 0.05, _make_frame()), (2, 0.1, _make_frame())]
    track = Track(track_id=1, frames=[0], bboxes=[(50, 20, 110, 80)])
    cropped = pipeline._crop_frames_to_bowler(frames, {1: track}, track)
    assert len(cropped) == 3
    for _, _, cut in cropped:
        assert cut is not None
        assert cut.shape != _make_frame().shape


def test_crop_frames_to_bowler_no_track_keeps_frames():
    frames = [(0, 0.0, _make_frame()), (1, 0.05, _make_frame())]
    track = Track(track_id=1, frames=[], bboxes=[])
    cropped = pipeline._crop_frames_to_bowler(frames, {1: track}, track)
    assert len(cropped) == 2
    assert cropped[0][2].shape == _make_frame().shape


def test_select_bowler_track_prefers_longest_largest():
    # Longer AND larger bbox wins over a short-but-big or long-but-small track.
    t1 = Track(track_id=1, frames=list(range(10)), bboxes=[(0, 0, 200, 200)] * 10)
    t2 = Track(track_id=2, frames=list(range(3)), bboxes=[(0, 0, 150, 150)] * 3)
    t3 = Track(track_id=3, frames=list(range(6)), bboxes=[(0, 0, 40, 40)] * 6)
    assert tracking.select_bowler_track({1: t1, 2: t2, 3: t3}).track_id == 1


def test_select_bowler_track_empty():
    assert tracking.select_bowler_track({}) is None


class _FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, frame, idx):
        from src.detection import Detection
        return [Detection(idx, b, 0.9, 0) for b in self._boxes]


def test_iou_fallback_tracks_frames_in_memory(monkeypatch):
    from src import detection
    monkeypatch.setattr(detection, "_HAS_ULTRALYTICS", False)
    # rebuild the module-level decision the same way tracking.py does
    monkeypatch.setattr(tracking, "_HAS_ULTRALYTICS", False)
    t = tracking.BowlerTracker(detector=_FakeDetector([(10, 10, 60, 90)]))
    frames = [(0, _make_frame()), (1, _make_frame()), (2, _make_frame())]
    tracks = t.track_frames(frames)
    assert len(tracks) == 1
    tr = next(iter(tracks.values()))
    assert tr.frames == [0, 1, 2]
    assert len(tr.bboxes) == 3
