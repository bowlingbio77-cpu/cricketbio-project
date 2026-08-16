"""Ball-tracking tests: release->impact windowing and impact-clipped annotation."""
import numpy as np

from src.ball_tracking import (BallTracker, BallPoint, MOTION_CONF,
                               annotate_frames, summarize)


def _ball(points):
    return [BallPoint(i, i / 30.0, float(x), float(y), 0.8, detected=True, w=16, h=16)
            for i, (x, y) in enumerate(points)]


def _blank_frames(n, w=300, h=200):
    return [(i, i / 30.0, np.zeros((h, w, 3), dtype=np.uint8)) for i in range(n)]


def _tracker(model_present: bool) -> BallTracker:
    """A BallTracker without loading a real YOLO model."""
    t = BallTracker.__new__(BallTracker)
    t._model = object() if model_present else None
    t.conf_threshold = 0.2
    t.seed_min_conf = 0.3
    return t


def _state():
    return {"last": None, "velocity": np.zeros(2), "gaps": 0,
            "box_size": None, "pending": None}


def _delivery_track():
    """Slow pre-release jitter -> fast straight release -> sharp deflection (impact)."""
    pts = [
        (100.0, 100.0), (101.0, 101.0), (102.0, 100.0), (101.5, 102.0), (103.0, 101.0),
        (105.0, 100.0), (125.0, 100.0), (145.0, 100.0), (165.0, 100.0), (185.0, 100.0),
        (200.0, 120.0), (215.0, 140.0), (230.0, 160.0), (245.0, 180.0),
    ]
    return _ball(pts)


def test_release_impact_window_detected():
    track = _delivery_track()
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    assert info["release_idx"] == 5
    assert info["impact_idx"] == 10
    assert info["trimmed"] is True


def test_annotate_clips_at_impact():
    track = _delivery_track()
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    clipped = [p for p in track if p.frame_idx <= info["impact_idx"]]
    assert clipped[-1].frame_idx == 10

    annotated = annotate_frames(_blank_frames(14), clipped)
    boxed = [i for i, (_idx, _ts, img) in enumerate(annotated) if img.any()]
    assert boxed == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # stops at impact (frame 10)

    full = annotate_frames(_blank_frames(14), track)
    boxed_full = [i for i, (_idx, _ts, img) in enumerate(full) if img.any()]
    assert boxed_full == list(range(14))  # without clipping the box follows the whole track


def test_no_impact_keeps_full_track():
    track = _ball([(100.0, 100.0 + i) for i in range(8)])
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    assert info["impact_idx"] is None
    clipped = track if info["impact_idx"] is None else [p for p in track
                                                        if p.frame_idx <= info["impact_idx"]]
    assert clipped == track


def test_motion_blob_cannot_hijack_yolo_track():
    """A closer motion blob must never outrank a YOLO detection in radius."""
    tracker = _tracker(model_present=True)
    st = _state()
    st["last"] = np.array([100.0, 100.0])
    st["velocity"] = np.zeros(2)
    st["box_size"] = (16, 16)
    cands = [
        (102.0, 100.0, 0.2, 10, 10, "motion"),   # closer, but a limb
        (110.0, 100.0, 0.9, 16, 16, "yolo"),     # correct source
    ]
    pt = tracker._step(1, 1 / 30.0, cands, st)
    assert pt is not None and pt.detected
    assert abs(pt.x - 110.0) < 1e-6


def test_confirmed_motion_seeds_when_yolo_misses():
    """A small blob persisting two frames can still start a track."""
    tracker = _tracker(model_present=True)
    st = _state()
    blob = (120.0, 80.0, MOTION_CONF, 12, 12, "motion")
    assert tracker._step(1, 1 / 30.0, [blob], st) is None   # frame 1: just stashed
    pt = tracker._step(2, 2 / 30.0, [blob], st)             # frame 2: confirmed
    assert pt is not None and pt.detected
    assert abs(pt.x - 120.0) < 1e-6 and abs(pt.y - 80.0) < 1e-6


def test_no_model_seeds_from_motion_directly():
    """Without a detector, the first motion blob may start the track."""
    tracker = _tracker(model_present=False)
    st = _state()
    pt = tracker._step(1, 1 / 30.0, [(100.0, 100.0, MOTION_CONF, 10, 10, "motion")], st)
    assert pt is not None and pt.detected


def test_summarize_fps_reports_px_per_second():
    track = _ball([(100.0, 100.0), (130.0, 100.0), (160.0, 100.0)])  # 30 px/frame @ 30 fps
    s = summarize(track, fps=30.0)
    assert s["max_speed_px_s"] == 900.0
    assert s["avg_speed_px_s"] == 900.0
