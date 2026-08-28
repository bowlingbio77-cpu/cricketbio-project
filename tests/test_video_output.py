"""PaceAI video-output regression tests.

Guards the two bugs that caused the final analysis video to be unplayable in
the browser:

1. The OpenCV ``mp4v`` fallback produced MPEG-4 Part 2 files that browsers
   cannot play. ``write_mp4`` must ALWAYS return an H.264 (avc1) MP4.
2. An unread ffmpeg subprocess pipe could deadlock, silently falling back to
   the broken codec.

These tests only touch video encoding/file handling -- never the ML/CV stages.
"""
import os

import numpy as np

from src.ball_tracking_v2 import write_mp4, validate_video


def _blank_frames(n, w=300, h=200):
    rng = np.random.default_rng(0)
    return [(i, i / 20.0, rng.integers(0, 60, (h, w, 3), dtype=np.uint8)) for i in range(n)]


def test_write_mp4_produces_browser_compatible_h264(tmp_path):
    """The main output must be H.264 (avc1) so the browser can play it."""
    out = str(tmp_path / "out.mp4")
    write_mp4(_blank_frames(20), out, fps=20.0)
    info = validate_video(out)
    assert info["exists"] is True
    assert info["size"] > 0
    assert "avc1" in (info["codec"] or "") or "h264" in (info["codec"] or "").lower()
    assert info["frames"] == 20
    assert info["fps"] == 20.0
    assert info["width"] == 300 and info["height"] == 200


def test_write_mp4_handles_dirty_frames(tmp_path):
    """None / grayscale / RGBA / mixed-resolution frames must be normalized, not
    produce a corrupt file, and must never hang or fall back to MPEG-4 Part 2."""
    base = np.zeros((100, 200, 3), np.uint8)
    small = base[:50, :100].copy()
    gray = np.zeros((100, 200), np.uint8)
    rgba = np.zeros((100, 200, 4), np.uint8)
    frames = [(0, 0.0, base), (1, 0.05, small), (2, 0.1, None),
              (3, 0.15, gray), (4, 0.2, rgba), (5, 0.25, base)]
    out = str(tmp_path / "dirty.mp4")
    write_mp4(frames, out, fps=25.0)
    info = validate_video(out)
    assert "avc1" in (info["codec"] or "") or "h264" in (info["codec"] or "").lower()
    assert info["width"] == 200 and info["height"] == 100


def test_write_mp4_invalid_fps_falls_back_safely(tmp_path):
    """A missing/invalid/zero FPS must not crash; a safe fallback is used."""
    out = str(tmp_path / "fps.mp4")
    write_mp4(_blank_frames(5), out, fps=0.0)
    info = validate_video(out)
    assert info["fps"] > 0


def test_validate_video_rejects_missing_file():
    import pytest
    with pytest.raises(RuntimeError):
        validate_video(os.path.join(os.path.dirname(__file__), "does_not_exist.mp4"))
