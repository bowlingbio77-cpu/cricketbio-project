"""Ball-tracking v2 contract tests: same windowing/validity guarantees as the
v1 suite, exercised through v2's multi-hypothesis Kalman pipeline."""
import numpy as np

from src.ball_tracking_v2 import (BallTracker, BallPoint, annotate_frames,
                                  summarize)


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
    assert boxed == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    full = annotate_frames(_blank_frames(14), track)
    boxed_full = [i for i, (_idx, _ts, img) in enumerate(full) if img.any()]
    assert boxed_full == list(range(14))


def test_no_impact_keeps_full_track():
    track = _ball([(100.0, 100.0 + i) for i in range(8)])
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    assert info["impact_idx"] is None
    clipped = track if info["impact_idx"] is None else [p for p in track
                                                        if p.frame_idx <= info["impact_idx"]]
    assert clipped == track


def test_split_segments_splits_on_jump_and_gap():
    b = lambda i, x, y: BallPoint(i, i / 30.0, x, y, 0.5, detected=True, w=16, h=16)
    raw = [b(0, 0, 0), b(1, 10, 0), b(2, 20, 0),   # segment A
           b(3, 200, 0),                           # 180 px teleport -> new segment
           b(5, 210, 0)]                           # frame gap (reset) -> new segment
    segs = BallTracker._split_segments(raw)
    assert [len(s) for s in segs] == [3, 1, 1]


def test_summarize_fps_reports_px_per_second():
    track = _ball([(100.0, 100.0), (130.0, 100.0), (160.0, 100.0)])  # 30 px/frame @ 30 fps
    s = summarize(track, fps=30.0)
    assert s["max_speed_px_s"] == 900.0
    assert s["avg_speed_px_s"] == 900.0


class _Box:
    def __init__(self, cx, cy, half, conf):
        self.xyxy = np.array([[cx - half, cy - half, cx + half, cy + half]])
        self.conf = [conf]


class _Result:
    def __init__(self, box):
        self.boxes = [box] if box else []


class _FakeModel:
    """A scripted YOLO that reports `positions[i]` (or nothing) on frame i."""

    def __init__(self, positions, conf=0.9, half=8):
        self.positions = positions
        self.conf = conf
        self.half = half
        self._i = 0

    def predict(self, frame, classes=None, conf=None, verbose=False):
        i = self._i
        self._i += 1
        pos = self.positions[i] if i < len(self.positions) else None
        if pos is None:
            return [_Result(None)]
        cx, cy = pos
        half = self.half(i) if callable(self.half) else self.half
        return [_Result(_Box(cx, cy, half, self.conf))]


def _run_track(model_predict, n_frames=31):
    t = BallTracker.__new__(BallTracker)
    t._model = _FakeModel(model_predict)
    t.conf_threshold = 0.1
    t.seed_min_conf = 0.3
    return t.track(_blank_frames(n_frames), fps=30.0)


def test_stationary_track_rejected():
    """YOLO confidently re-firing on a fixed object must yield NO track."""
    positions = [(100.0, 100.0)] * 31
    track, stats = _run_track(positions)
    assert track == []
    assert stats["n_frames"] == 0


def test_moving_ball_track_survives():
    """A ball travelling across the frame is a valid track."""
    positions = [(20.0 + 8.0 * i, 100.0) for i in range(31)]
    track, stats = _run_track(positions)
    assert len(track) == 31
    assert track[0].frame_idx == 0 and track[-1].frame_idx == 30
    assert stats["coverage_pct"] == 100.0


def test_slow_growing_object_rejected():
    """A person walking toward the camera grows in size and moves slowly:
    rejected even though it travels enough to pass the spread/path checks."""
    positions = [(300.0 - 4.0 * i, 100.0) for i in range(31)]
    t = BallTracker.__new__(BallTracker)
    t._model = _FakeModel(positions, half=lambda i: 4 if i < 20 else 20)
    t.conf_threshold = 0.1
    t.seed_min_conf = 0.3
    track, stats = t.track(_blank_frames(31), fps=30.0)
    assert track == []


def test_duplicate_yolo_boxes_do_not_split_the_ball():
    """Overlapping YOLO boxes on the same ball must not spawn a competing
    duplicate track that steals later detections from the real one."""
    t = BallTracker.__new__(BallTracker)
    t._model = object()          # model present: only YOLO may seed
    t.seed_min_conf = 0.3
    cands = [(100.0, 100.0, 0.9, 16, 16, "yolo"),
             (105.0, 100.0, 0.8, 16, 16, "yolo")]   # 5 px apart: same object
    still_active, _done, _next = t._step_all(0, 0.0, cands, [], 0)
    assert len(still_active) == 1


def _delivery_positions():
    """Slow jitter -> straight release -> ball stops at the bat (impact)."""
    return ([(100.0 + 2.0 * i, 100.0) for i in range(5)] +
            [(110.0 + 8.0 * (i - 5), 100.0) for i in range(5, 26)] +
            [(270.0, 100.0)] * 5)


def test_outcome_ok_for_delivery():
    track, stats = _run_track(_delivery_positions(), n_frames=31)
    assert len(track) == 31
    assert stats["outcome"] == "ok"


def test_outcome_release_no_impact_for_straight_flight():
    positions = [(20.0 + 8.0 * i, 100.0) for i in range(31)]
    track, stats = _run_track(positions)
    assert stats["outcome"] == "release_no_impact"


def test_outcome_track_too_short_when_rejected():
    positions = [(100.0, 100.0)] * 31
    track, stats = _run_track(positions)
    assert track == []
    assert stats["outcome"] == "track_too_short"


def _release_with_stall(deflect: bool):
    """Fast straight release with a one-frame transient stall mid-flight
    (a jittery motion point where the ball appears to pause), then either a
    real impact deflection or a straight continuation."""
    pts = [
        (0.0, 100.0), (1.0, 99.0), (2.0, 100.0), (1.0, 101.0), (3.0, 100.0),
        (23.0, 100.0), (43.0, 100.0), (63.0, 100.0), (83.0, 100.0),
        (83.0, 100.0),                       # one-frame transient stall (speed 0)
        (103.0, 100.0), (123.0, 100.0), (143.0, 100.0), (163.0, 100.0),
        (183.0, 100.0),
    ]
    if deflect:
        pts += [(183.0, 140.0), (203.0, 180.0)]
    else:
        pts += [(203.0, 100.0)]
    return _ball(pts)


def test_transient_stall_does_not_false_fire_impact():
    """A one-frame stall (speed 0) inside the flight must NOT be taken as
    bat/pad impact; the real deflection later is the impact."""
    track = _release_with_stall(deflect=True)
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    assert info["impact_idx"] == 15, info


def test_transient_stall_without_deflection_keeps_no_impact():
    """Stall then straight continuation: no impact at all (old 2-frame median
    window false-fired on the stall at frame 8)."""
    track = _release_with_stall(deflect=False)
    _trimmed, info = BallTracker._trim_to_release_impact(track)
    assert info["impact_idx"] is None, info


def test_genuine_stop_still_detected_as_impact():
    """A real bat/pad stop drops speed for several frames and must still
    register even with the widened median window."""
    pts = [
        (0.0, 100.0), (1.0, 99.0), (2.0, 100.0), (1.0, 101.0), (3.0, 100.0),
        (23.0, 100.0), (43.0, 100.0), (63.0, 100.0), (83.0, 100.0),
        (83.0, 100.0), (83.0, 100.0), (83.0, 100.0), (83.0, 100.0),
    ]
    _trimmed, info = BallTracker._trim_to_release_impact(_ball(pts))
    assert info["impact_idx"] == 7, info


def test_gap_out_reseed_stitches_full_flight():
    """A ball that disappears for more than MAX_GAP_FRAMES mid-flight
    (occlusion) and re-appears on the ballistic path must be re-seeded into a
    single stitched track spanning the WHOLE delivery -- including when the
    ancestor is the longer (predicted-tail-inflated) endpoint."""
    positions = ([(20.0 + 8.0 * i, 100.0) for i in range(12)] +
                 [None] * 9 +
                 [(20.0 + 8.0 * i, 100.0) for i in range(21, 31)])
    track, stats = _run_track(positions, n_frames=31)
    assert len(track) == 31
    assert track[0].frame_idx == 0 and track[-1].frame_idx == 30
    assert sum(1 for p in track if p.detected) == 22


def test_gap_out_reseed_release_reported_at_flight_start():
    """The stitched trajectory starts at the ball's first (pre-gap) detection,
    so release is reported from the beginning of the flight, not from where the
    ball reappeared."""
    positions = ([(20.0 + 8.0 * i, 100.0) for i in range(5)] +
                 [None] * 10 +
                 [(20.0 + 8.0 * i, 100.0) for i in range(16, 26)])
    track, stats = _run_track(positions, n_frames=26)
    assert track[0].frame_idx == 0
    assert stats["outcome"] in ("release_no_impact", "ok")


def test_gap_out_without_reappearance_keeps_no_reseed():
    """If the ball never re-appears near the ballistic prediction, no stitched
    track is produced -- the pre-gap segment stands on its own."""
    positions = ([(20.0 + 8.0 * i, 100.0) for i in range(12)] +
                 [None] * 19)
    track, stats = _run_track(positions, n_frames=31)
    assert track
    assert track[-1].frame_idx <= 20
    assert stats["outcome"] == "release_no_impact"
