"""Tests for pose-person selection (who to analyze when several people are in
frame). `_primary_person` must pick the bowler = largest landmark bbox."""
import numpy as np

from src.pose_estimation import _primary_person


def _person(cx, cy, w, h):
    """Synthetic landmark list forming a rectangle of given size/position."""
    lm = []
    for i in range(33):
        x = cx + (w / 2) * np.cos(i)
        y = cy + (h / 2) * np.sin(i)
        lm.append(type("L", (), {"x": float(x), "y": float(y)})())
    return lm


def test_primary_person_picks_largest_bbox():
    # three people: a small fielder, a mid keeper, a large bowler
    small = _person(0.2, 0.2, 0.05, 0.12)
    mid = _person(0.5, 0.5, 0.10, 0.25)
    big = _person(0.7, 0.6, 0.25, 0.60)
    assert _primary_person([small, mid, big]) == 2


def test_primary_person_picks_first_when_single_person():
    lone = _person(0.5, 0.5, 0.2, 0.5)
    assert _primary_person([lone]) == 0
