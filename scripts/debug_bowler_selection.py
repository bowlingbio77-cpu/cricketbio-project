"""Adversarial bowler-selection harness (Phase 2/3 proof).

Feeds the LIVE scorer (tracking.score_bowler_tracks /
select_bowler_track_with_meta) synthetic Track objects that model real
cricket cinematography and tracker behaviour, and prints the ranked
candidates with every evidence component.

Run:  python scripts/debug_bowler_selection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.tracking import Track, score_bowler_tracks, select_bowler_track_with_meta

FRAME_H, FRAME_W = 360, 640
FRAMES = 60


def _lin(start, end, count):
    return list(np.linspace(start, end, count))


def bowler_side_view(n=30, start_frame=15, step=13.7, diag_grow=1.18, drop=50):
    frames = list(range(start_frame, start_frame + n))
    xs = _lin(470.0, 475.0 - step * (n - 1), n)
    ys = _lin(140.0, 140.0 + drop, n)
    bboxes = []
    for k in range(n):
        w = 34.0 + (diag_grow - 1.0) * 34.0 * abs(k - 0) / (n - 1)
        h = 95.0 + (diag_grow - 1.0) * 95.0 * k / (n - 1)
        bboxes.append((int(xs[k] - w / 2), int(ys[k] - h / 2),
                       int(xs[k] + w / 2), int(ys[k] + h / 2)))
    return Track(track_id=1, frames=frames, bboxes=bboxes)


def bowler_far_with_occlusion():
    part1 = Track(track_id=1, frames=list(range(0, 12)),
                  bboxes=[(int(140 - i * 14), 20, int(140 - i * 14) + 30, 110)
                          for i in range(12)])
    part2 = Track(track_id=7, frames=list(range(15, 28)),
                  bboxes=[(int(-40 - i * 14), 24, int(-40 - i * 14) + 30, 118)
                          for i in range(13)])
    return part1, part2


def striker_advances(n=60, amp=16.0, hop_every=5, hop=16.0):
    frames = list(range(n))
    x = 300.0
    bboxes = []
    for k in range(n):
        x += amp
        if k % hop_every == hop_every - 1:
            x += hop
        bboxes.append((int(x - 120), 30, int(x + 120), 310))
    return Track(track_id=2, frames=frames, bboxes=bboxes)


def static_big(track_id=2, n=60, x=250, w=200, h=260, y0=20):
    return Track(track_id=track_id, frames=list(range(n)),
                 bboxes=[(x, y0, x + w, y0 + h)] * n)


def bowler_behind_view(n=40, grow_to=1.5, step=11.0):
    frames = list(range(n))
    xs = _lin(320.0, 320.0, n)
    ys = _lin(60.0, 60.0 + step * (n - 1), n)
    bboxes = []
    for k in range(n):
        s = 1.0 + (grow_to - 1.0) * k / (n - 1)
        bboxes.append((int(xs[k] - 30 * s), int(ys[k] - 80 * s),
                       int(xs[k] + 30 * s), int(ys[k] + 80 * s)))
    return Track(track_id=1, frames=frames, bboxes=bboxes)


def run_case(name, tracks):
    print("=" * 90)
    print(f"CASE: {name}")
    ranked = score_bowler_tracks(tracks, frame_dims=(FRAME_H, FRAME_W),
                                 total_frames=FRAMES)
    if not ranked:
        print("  (no tracks)")
        return
    hdr = ("track  score  conf  frames  motion active growth span "
           "straight range  band decel  vert delivery")
    print("  " + hdr)
    for r in ranked:
        print("  %5d %6.3f %5.2f  %5d  %6.3f %6.3f %6.3f %4.2f "
              "%8.3f %5.2f %5.2f %5.2f %5.2f %8.3f" % (
                  r["track_id"], r["score"], r["confidence"], r["n_frames"],
                  r["motion"], r["active"], r["growth"], r["span"],
                  r["straight"], r["range"], r["band"], r["decel"],
                  r["vert"], r["delivery"]))
    winner, meta = select_bowler_track_with_meta(
        tracks, frame_dims=(FRAME_H, FRAME_W), total_frames=FRAMES)
    if winner is None:
        print("  >> SELECTED BOWLER: None (no-bowler gate fired)")
    else:
        print(f"  >> SELECTED BOWLER: track #{winner.track_id} "
              f"(score={meta['score']:.3f} conf={meta['confidence']:.3f})")


def main():
    print(f"Scoring window: {FRAMES} frames @ {FRAME_W}x{FRAME_H} "
          "(frame diag=%.1f)" % np.hypot(FRAME_W, FRAME_H))

    # Control: the camera the scorer was tuned for (behind-bowler view).
    # Bowler runs DOWN frame (vert=1) and grows; batsman static at end.
    run_case("CONTROL behind-view: bowler approaches + static batsman",
             {1: bowler_behind_view(), 2: static_big()})

    # Adversarial 1: broadcast/side view. Small far bowler runs horizontally
    # right->left; near-camera STRIKER advances down the pitch (footwork +
    # down-track drive) for the whole clip, so it has ~3.5x the raw pixel path
    # of the bowler and is tracked continuously (span=1.0).
    run_case("ADVERSARIAL-1 broadcast/side view, near striker advances vs "
             "far horizontal bowler",
             {1: bowler_side_view(), 2: striker_advances()})

    # Adversarial 2: bowler loses its track (occlusion) mid-run; the pieces are
    # two short tracks. A continuous, near-camera, footworking batsman is the
    # longest + biggest raw-motion track.
    part1, part2 = bowler_far_with_occlusion()
    run_case("ADVERSARIAL-2 fragmented bowler (gap) + continuous footworking "
             "batsman",
             {1: part1, 7: part2, 2: striker_advances(amp=8.0, hop_every=4)})

    # Adversarial 3: all-static scene must still return None (gate must survive
    # the rewrite).
    run_case("CONTROL all-static scene -> must be None",
             {1: static_big(track_id=1, x=0, w=160, h=220),
              2: static_big(track_id=2, x=300, w=120, h=180, y0=60)})

    print("=" * 90)


if __name__ == "__main__":
    main()