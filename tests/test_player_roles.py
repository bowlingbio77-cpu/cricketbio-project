"""Tests for cricket role classification (batsman, wicketkeeper, umpire)
and the broadcast-style overlay renderer."""
import numpy as np
import pytest

from src import tracking
from src.tracking import (Track, classify_player_roles,
                          ROLE_BATSMAN, ROLE_WICKETKEEPER, ROLE_UMPIRE,
                          ROLE_FIELDER)
from src import analysis_replay


def _make_track(tid, frames, bboxes):
    return Track(track_id=tid, frames=frames, bboxes=bboxes)


class TestRoleClassification:

    def test_empty_tracks_returns_empty(self):
        assert classify_player_roles({}, bowler_track_id=1) == {}

    def test_bowler_excluded(self):
        bowler = _make_track(1, list(range(10)),
                             [(100, 50, 200, 300)] * 10)
        result = classify_player_roles({1: bowler}, bowler_track_id=1,
                                        frame_dims=(360, 640))
        assert 1 not in result

    def test_static_far_person_classified_as_batsman(self):
        # Person at the top of frame (far from camera), barely moving
        bowler = _make_track(1, list(range(10)),
                             [(i * 30, 200, i * 30 + 60, 350) for i in range(10)])
        batsman = _make_track(2, list(range(10)),
                              [(300, 30, 480, 250)] * 10)  # static, upper frame
        result = classify_player_roles(
            {1: bowler, 2: batsman}, bowler_track_id=1,
            frame_dims=(360, 640), total_frames=10)
        assert 2 in result
        assert result[2]["role"] == ROLE_BATSMAN
        assert result[2]["confidence"] > 0.0

    def test_crouching_near_person_classified_as_keeper(self):
        # Person at the bottom of frame (near camera), crouching (wide bbox)
        bowler = _make_track(1, list(range(10)),
                             [(i * 30, 200, i * 30 + 60, 350) for i in range(10)])
        keeper = _make_track(3, list(range(10)),
                             [(250, 280, 400, 340)] * 10)  # wide, short, near bottom
        result = classify_player_roles(
            {1: bowler, 3: keeper}, bowler_track_id=1,
            frame_dims=(360, 640), total_frames=10)
        assert 3 in result
        assert result[3]["role"] == ROLE_WICKETKEEPER
        assert result[3]["confidence"] > 0.0

    def test_upright_mid_person_classified_as_umpire(self):
        # Person standing upright (tall narrow bbox), mid-position, very still
        bowler = _make_track(1, list(range(10)),
                             [(i * 30, 200, i * 30 + 60, 350) for i in range(10)])
        umpire = _make_track(4, list(range(10)),
                             [(310, 100, 360, 310)] * 10)  # tall, narrow, mid-frame
        result = classify_player_roles(
            {1: bowler, 4: umpire}, bowler_track_id=1,
            frame_dims=(360, 640), total_frames=10)
        assert 4 in result
        # Umpire should score reasonably high on the umpire component
        scores = result[4]["scores"]
        assert scores[ROLE_UMPIRE] > 0.0

    def test_single_frame_track_classified(self):
        bowler = _make_track(1, list(range(10)),
                             [(100, 50, 200, 300)] * 10)
        short = _make_track(5, [0], [(100, 50, 200, 300)])
        result = classify_player_roles(
            {1: bowler, 5: short}, bowler_track_id=1,
            frame_dims=(360, 640))
        assert 5 in result
        assert "role" in result[5]
        assert "confidence" in result[5]

    def test_all_roles_have_valid_structure(self):
        bowler = _make_track(1, list(range(5)),
                             [(i * 40, 100, i * 40 + 60, 280) for i in range(5)])
        others = {
            2: _make_track(2, list(range(5)), [(300, 30, 480, 250)] * 5),
            3: _make_track(3, list(range(5)), [(250, 280, 400, 340)] * 5),
        }
        result = classify_player_roles(
            {1: bowler, **others}, bowler_track_id=1,
            frame_dims=(360, 640), total_frames=5)
        for tid, info in result.items():
            assert "role" in info
            assert "confidence" in info
            assert "scores" in info
            assert info["role"] in (ROLE_BATSMAN, ROLE_WICKETKEEPER,
                                    ROLE_UMPIRE, ROLE_FIELDER)
            assert 0.0 <= info["confidence"] <= 1.0


class TestBroadcastOverlay:

    def _frame(self, h=360, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_draw_player_role_draws_batsman_box(self):
        img = self._frame()
        analysis_replay._draw_player_role(
            img, (100, 50, 250, 280), ROLE_BATSMAN, 0.90, 360, 640)
        # Should draw red pixels (batsman color)
        mask = np.all(img == np.array([0, 0, 255], dtype=np.uint8), axis=-1)
        assert mask.sum() > 0

    def test_draw_player_role_draws_keeper_box(self):
        img = self._frame()
        analysis_replay._draw_player_role(
            img, (250, 280, 400, 340), ROLE_WICKETKEEPER, 0.94, 360, 640)
        mask = np.all(img == np.array([0, 200, 255], dtype=np.uint8), axis=-1)
        assert mask.sum() > 0

    def test_draw_player_role_draws_umpire_box(self):
        img = self._frame()
        analysis_replay._draw_player_role(
            img, (310, 100, 360, 310), ROLE_UMPIRE, 0.92, 360, 640)
        mask = np.all(img == np.array([200, 200, 0], dtype=np.uint8), axis=-1)
        assert mask.sum() > 0

    def test_draw_player_role_none_bbox_does_nothing(self):
        img = self._frame()
        original = img.copy()
        analysis_replay._draw_player_role(
            img, None, ROLE_BATSMAN, 0.90, 360, 640)
        assert np.array_equal(img, original)

    def test_replay_renders_with_player_roles(self, tmp_path):
        from src.ball_tracking_v2 import BallPoint
        from src.pose_estimation import PoseFrame
        frames = [(i, 0.05 * i, self._frame()) for i in range(3)]
        lm = np.zeros((33, 4))
        for i in range(33):
            lm[i] = [0.5, 0.5, 0.0, 0.9]
        pose_seq = [PoseFrame(i, 0.05 * i, lm, np.zeros((33, 3))) for i in range(3)]
        bowler_bboxes = {0: (50, 100, 150, 300), 1: (80, 100, 180, 300),
                         2: (110, 100, 210, 300)}
        all_tracks = {
            1: _make_track(1, [0, 1, 2], [(50, 100, 150, 300),
                                           (80, 100, 180, 300),
                                           (110, 100, 210, 300)]),
            2: _make_track(2, [0, 1, 2], [(400, 30, 580, 250)] * 3),
        }
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 0.85,
                "scores": {ROLE_BATSMAN: 0.85, ROLE_WICKETKEEPER: 0.1,
                           ROLE_UMPIRE: 0.05}}
        }
        out = tmp_path / "replay_roles.mp4"
        path = analysis_replay.render_analysis_replay(
            frames, [], pose_seq, bowler_bboxes, release_frame=None,
            output_path=str(out), fps=5.0, frame_dims=(360, 640),
            bowler_track_id=1, bowler_confidence=0.9,
            player_roles=player_roles, all_tracks=all_tracks)
        assert out.exists() and out.stat().st_size > 0

    def test_replay_works_without_player_roles(self, tmp_path):
        """Backwards compatibility: replay still works when player_roles is None."""
        frames = [(i, 0.05 * i, self._frame()) for i in range(2)]
        out = tmp_path / "replay_no_roles.mp4"
        path = analysis_replay.render_analysis_replay(
            frames, [], [], {}, release_frame=None,
            output_path=str(out), fps=5.0, frame_dims=(360, 640))
        assert out.exists() and out.stat().st_size > 0
