"""Ball-tracking tests: release->impact windowing and impact-clipped annotation."""
import numpy as np

from src.ball_tracking import BallTracker, BallPoint, annotate_frames


def _ball(points):
    return [BallPoint(i, i / 30.0, float(x), float(y), 0.8, detected=True, w=16, h=16)
            for i, (x, y) in enumerate(points)]


def _blank_frames(n, w=300, h=200):
    return [(i, i / 30.0, np.zeros((h, w, 3), dtype=np.uint8)) for i in range(n)]


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
