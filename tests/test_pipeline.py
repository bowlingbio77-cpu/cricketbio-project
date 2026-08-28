"""Tests for the orchestration layer: bowler-crop helpers and the in-memory
tracking path (IoU fallback), which run without ultralytics/MediaPipe."""
import numpy as np
import pytest

from src import pipeline, tracking
from src.tracking import Track
from src.pose_estimation import PoseFrame, draw_skeleton


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


def test_select_bowler_track_empty():
    assert tracking.select_bowler_track({}) is None
    assert tracking.select_bowler_track_with_meta({}) == (None, None)


def test_crop_frames_to_bowler_bounded_carry_stops_when_bowler_gone():
    # Bowler is tracked only on frames 0..2 of a 10-frame clip. The crop bbox
    # may be carried forward briefly (<= BOWLER_CARRY_GAP_FRAMES), but once the
    # bowler is gone the remaining frames MUST stay uncropped -- no substitute,
    # no stale box chasing a batsman/keeper off into the clip.
    frames = [(i, 0.05 * i, _make_frame()) for i in range(10)]
    track = Track(track_id=1, frames=[0, 1, 2],
                  bboxes=[(20, 20, 80, 80)] * 3)
    cropped = pipeline._crop_frames_to_bowler(frames, {1: track}, track)
    shapes = [cut.shape for _, _, cut in cropped]
    full = _make_frame().shape
    # frames 0..7 stay bowler-cropped (0,1,2 tracked + carry on 3..7)...
    n_cropped = sum(1 for s in shapes if s != full)
    assert n_cropped == 8
    assert shapes[2] != full
    assert shapes[7] != full
    # ...and frames 8..9 (bowler long gone) are full-frame again.
    assert shapes[8:] == [full, full]


def test_padded_bowler_bboxes_bounded_no_box_after_gap():
    h, w, n = 360, 640, 10
    bow = Track(track_id=1, frames=[0, 1, 2],
                bboxes=[(100, 100, 200, 260)] * 3)
    padded = pipeline._padded_bowler_bboxes(bow, n, h, w)
    assert sorted(padded.keys()) == list(range(0, 8))  # 0..2 + carry 3..7
    assert 8 not in padded and 9 not in padded
    for box in padded.values():
        x1, y1, x2, y2 = box
        assert x1 < x2 and y1 < y2
        assert 0 <= x1 and 0 <= y1 and x2 <= w and y2 <= h


def _mover(start, end, frames):
    """A Track that travels straight from (start,0) px to (end,0) px across
    `frames` (moving like a run-up), at a fixed height."""
    xs = np.linspace(start, end, len(frames))
    bboxes = [(int(x), 100, int(x) + 60, 260) for x in xs]
    return Track(track_id=1, frames=list(frames), bboxes=bboxes)


def test_select_bowler_track_moving_bowler_beats_larger_static_batsman():
    # The classic failure: a big static batsman is the largest bbox in frame.
    # Cricket evidence must rank the RUNNING bowler above the giant wicketkeeper
    # / batsman -- the selector must NOT be a "biggest person" test.
    moving = Track(track_id=1, frames=list(range(10)),
                   bboxes=[(i * 20, 100, i * 20 + 60, 260) for i in range(10)])
    static_big = Track(track_id=2, frames=list(range(10)),
                       bboxes=[(300, 40, 520, 300)] * 10)
    static_mid = Track(track_id=3, frames=list(range(10)),
                       bboxes=[(80, 80, 160, 240)] * 10)
    winner, meta = tracking.select_bowler_track_with_meta(
        {1: moving, 2: static_big, 3: static_mid},
        frame_dims=(360, 640), total_frames=10)
    assert winner is not None and winner.track_id == 1
    assert meta["track_id"] == 1
    assert meta["confidence"] > 0.5
    assert meta["motion"] > 0.1
    assert meta["active"] > 0.9


def test_select_bowler_track_all_static_scene_returns_none():
    # No one runs/delivers -> no bowler. Previously the "longest x largest"
    # heuristic crowned a stationary batsman; now the honest answer is None.
    t1 = Track(track_id=1, frames=list(range(10)),
               bboxes=[(0, 0, 200, 200)] * 10)
    t2 = Track(track_id=2, frames=list(range(3)),
               bboxes=[(0, 0, 150, 150)] * 3)
    t3 = Track(track_id=3, frames=list(range(6)),
               bboxes=[(0, 0, 40, 40)] * 6)
    assert tracking.select_bowler_track({1: t1, 2: t2, 3: t3},
                                        frame_dims=(360, 640),
                                        total_frames=10) is None


def test_select_bowler_track_identity_lock_never_swaps_to_batsman():
    # Even when a batsman is bigger AND longer in the frame, the bowler keeps
    # winning because it is the only track that MOVES like a bowler.
    bowler = Track(track_id=1, frames=list(range(15)),
                   bboxes=[(int(i * 25), 100, int(i * 25) + 60, 280) for i in range(15)])
    batsman = Track(track_id=2, frames=list(range(15)),
                    bboxes=[(320, 30, 600, 330)] * 15)
    winner, _ = tracking.select_bowler_track_with_meta(
        {1: bowler, 2: batsman}, frame_dims=(360, 640), total_frames=15)
    assert winner is not None and winner.track_id == 1
    # Identity lock: the winner must only contain the bowler's own boxes --
    # the batsman's giant boxes are never merged/substituted in.
    assert winner.frames == bowler.frames
    assert winner.bboxes == bowler.bboxes


def test_select_bowler_track_stitches_continuation_not_static_track():
    # The bowler's ByteTrack id resets after an occlusion (gap of 2 frames).
    # The continuation (same physical person, spatially continuous) must be
    # stitched back into the locked bowler; a separate static track is not.
    part1_frames = list(range(0, 5))
    part1 = Track(track_id=1,
                  frames=part1_frames,
                  bboxes=[(int(i * 40), 100, int(i * 40) + 60, 260) for i in part1_frames])
    part2_frames = list(range(7, 10))
    part2 = Track(track_id=9,
                  frames=part2_frames,
                  bboxes=[(int(i * 40), 100, int(i * 40) + 60, 260) for i in part2_frames])
    static = Track(track_id=5, frames=list(range(10)), bboxes=[(200, 150, 260, 200)] * 10)
    winner, _ = tracking.select_bowler_track_with_meta(
        {1: part1, 9: part2, 5: static}, frame_dims=(360, 640), total_frames=10)
    assert winner is not None
    stitched = set(part1.frames) | set(part2.frames)
    assert stitched.issubset(set(winner.frames))
    assert len(winner.frames) > 7  # both segments merged (not an alias for static)


def test_select_bowler_track_wrapper_stays_compatible():
    # scripts/test_yolo_detection.py and evaluation code call the 1-arg form.
    tr = _mover(0, 600, list(range(10)))
    assert tracking.select_bowler_track({1: tr}).track_id == 1
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


def _pose_frame(visibility=0.9):
    """Synthetic full-visibility PoseFrame over a 33-landmark skeleton."""
    lm = np.zeros((33, 4))
    for i in range(33):
        lm[i] = [0.2 + 0.02 * i, 0.3 + 0.01 * i, 0.0, visibility]
    return PoseFrame(0, 0.0, lm, np.zeros((33, 3)))


def test_draw_skeleton_returns_matching_shape_and_does_not_mutate():
    img = _make_frame(h=120, w=160)
    original = img.copy()
    out = draw_skeleton(img, _pose_frame())
    assert out.shape == img.shape and out.dtype == img.dtype
    # bones + joints must draw pixels onto the frame
    assert int((out > 0).sum()) > 0
    # the source frame must not be mutated (we return a copy)
    assert np.array_equal(img, original)


def test_draw_skeleton_skips_low_visibility_landmarks():
    img = _make_frame(h=120, w=160)
    out = draw_skeleton(img, _pose_frame(visibility=0.05), min_visibility=0.4)
    # no visible landmarks -> nothing drawn
    assert int((out > 0).sum()) == 0
    assert np.array_equal(out, img)
