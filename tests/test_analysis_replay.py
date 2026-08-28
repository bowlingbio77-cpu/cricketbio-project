"""Tests for the unified Analysis Replay renderer overlays.

These verify the user-facing contract of the analysis video:
  * the BALL is drawn as ONE small box -- red for every real ball source
    (YOLO/wrist-proxy/blended), amber+dashed for predicted points
  * the BOWLER is drawn only in frames where the bowler crop bbox really
    exists (identity lock; never a substitute person)
  * the debug diagnostic panel appears only when `debug=True`
"""
import numpy as np
import pytest

from src import analysis_replay
from src.ball_tracking_v2 import BallPoint


def _frame(h=90, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _count_color(img, color):
    mask = np.all(img == np.asarray(color, dtype=np.uint8), axis=-1)
    return int(mask.sum())


def test_ball_box_red_for_detected_yolo():
    img = _frame()
    pt = BallPoint(0, 0.0, 100, 45, 0.92, detected=True, w=12, h=12, source="yolo")
    analysis_replay._draw_ball_box(img, pt, *img.shape[:2])
    assert _count_color(img, analysis_replay._BALL_DETECTED) > 0


def test_ball_box_red_for_wrist_proxy():
    # Wrist-proxy pre-release points must use the SAME red box as real detections
    # (previously cyan) -- the replay uses one unambiguous ball color.
    img = _frame()
    pt = BallPoint(3, 0.15, 80, 60, 0.0, detected=False, w=10, h=10, source="wrist_proxy")
    analysis_replay._draw_ball_box(img, pt, *img.shape[:2])
    assert _count_color(img, analysis_replay._BALL_DETECTED) > 0


def test_ball_box_red_for_blended_handoff():
    img = _frame()
    pt = BallPoint(4, 0.2, 70, 55, 0.0, detected=False, w=10, h=10, source="blended")
    analysis_replay._draw_ball_box(img, pt, *img.shape[:2])
    assert _count_color(img, analysis_replay._BALL_DETECTED) > 0


def test_ball_box_amber_dashed_for_predicted_no_red():
    img = _frame()
    pt = BallPoint(5, 0.25, 110, 40, 0.0, detected=False, w=10, h=10, source="predicted")
    analysis_replay._draw_ball_box(img, pt, *img.shape[:2])
    assert _count_color(img, analysis_replay._BALL_PRED) > 0
    assert _count_color(img, analysis_replay._BALL_DETECTED) == 0


def test_bowler_box_drawn_only_when_bbox_exists():
    img = _frame()
    analysis_replay._draw_bowler_box(img, None, *img.shape[:2])
    assert _count_color(img, analysis_replay._BOWLER_BOX) == 0

    img2 = _frame()
    analysis_replay._draw_bowler_box(img2, (40, 20, 120, 80), *img2.shape[:2])
    assert _count_color(img2, analysis_replay._BOWLER_BOX) > 0


def test_debug_panel_only_when_requested():
    img = _frame(h=200, w=320)
    analysis_replay._draw_debug_panel(img, 320, 200, ["BOWLER TRACK #3", "ball: 5 real + 2 pred"])
    assert _count_color(img, analysis_replay._DEBUG_TEXT) > 0

    img2 = _frame(h=200, w=320)
    analysis_replay._draw_debug_panel(img2, 320, 200, [])
    assert _count_color(img2, analysis_replay._DEBUG_TEXT) == 0


def test_render_analysis_replay_writes_playable_file(tmp_path):
    frames = [(i, 0.05 * i, _frame()) for i in range(4)]
    traj = [
        BallPoint(1, 0.05, 90, 45, 0.95, detected=True, w=12, h=12, source="yolo"),
        BallPoint(2, 0.10, 100, 45, 0.0, detected=False, w=12, h=12, source="wrist_proxy"),
        BallPoint(3, 0.15, 110, 45, 0.0, detected=False, w=12, h=12, source="predicted"),
    ]
    bowler_bboxes = {1: (30, 10, 120, 80)}
    out = tmp_path / "replay.mp4"
    path = analysis_replay.render_analysis_replay(
        frames, traj, [], bowler_bboxes, release_frame=2, output_path=str(out),
        fps=5.0, frame_dims=(90, 160),
        debug=True, bowler_track_id=1, bowler_confidence=0.93,
    )
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_render_analysis_replay_accepts_empty_trajectory(tmp_path):
    frames = [(0, 0.0, _frame()), (1, 0.05, _frame())]
    out = tmp_path / "replay_empty.mp4"
    path = analysis_replay.render_analysis_replay(
        frames, [], [], {}, release_frame=None, output_path=str(out),
        fps=5.0, frame_dims=(90, 160), debug=False,
    )
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0