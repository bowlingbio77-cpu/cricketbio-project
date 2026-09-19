"""Tests for G2 (P0): a YOLO/ByteTrack constructor failure (missing weights,
no network, corrupt file) must degrade to the flagged HOG/IoU fallback instead
of throwing and silently disabling the whole detection stage."""
import pytest

from src import detection, tracking


class _RaisingYOLO:
    def __init__(self, *a, **k):
        raise RuntimeError("no network to fetch yolo11n.pt")


def test_detector_degrades_to_hog_when_yolo_fails(monkeypatch):
    monkeypatch.setattr(detection, "_HAS_ULTRALYTICS", True)
    monkeypatch.setattr(detection, "YOLO", _RaisingYOLO, raising=False)

    det = detection.BowlerDetector(weights="does_not_exist.pt")

    assert det.backend == "hog_fallback"
    assert det.fallback_reason is not None
    assert "YOLO" in det.fallback_reason and "HOG" in det.fallback_reason
    # HOG must actually be usable (produce detections on a blank frame)
    dets = det.detect(_blank_frame(160, 240), 0)
    assert isinstance(dets, list)


def test_tracker_degrades_to_iou_when_bytetrack_fails(monkeypatch):
    monkeypatch.setattr(tracking, "_HAS_ULTRALYTICS", True)
    monkeypatch.setattr(tracking, "YOLO", _RaisingYOLO, raising=False)
    # keep the inner detector off YOLO too so the test stays hermetic
    monkeypatch.setattr(detection, "_HAS_ULTRALYTICS", False)

    tr = tracking.BowlerTracker(weights="does_not_exist.pt")

    assert tr.backend == "iou_fallback"
    assert tr.fallback_reason is not None
    assert "ByteTrack" in tr.fallback_reason and "IoU" in tr.fallback_reason


def test_tracker_survives_and_still_tracks_after_degradation(monkeypatch):
    monkeypatch.setattr(tracking, "_HAS_ULTRALYTICS", True)
    monkeypatch.setattr(tracking, "YOLO", _RaisingYOLO, raising=False)

    class _FakeDetector:
        def __init__(self, weights=None, conf_threshold=None):
            self.backend = "fake_detector"
            self.fallback_reason = None
        def detect(self, frame, idx):
            from src.detection import Detection
            return [Detection(idx, (5, 5, 40, 70), 0.9, 0)]

    tr = tracking.BowlerTracker(weights="does_not_exist.pt", detector=_FakeDetector())
    assert tr.backend == "iou_fallback"
    tracks = tr.track_frames([(0, _blank_frame()), (1, _blank_frame()), (2, _blank_frame())])
    assert len(tracks) == 1
    assert list(tracks.values())[0].frames == [0, 1, 2]


def _blank_frame(w=160, h=90):
    import numpy as np
    return np.zeros((h, w, 3), dtype=np.uint8)