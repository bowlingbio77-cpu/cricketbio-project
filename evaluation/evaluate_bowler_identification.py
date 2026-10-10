"""A/B evaluation of bowler identification on real annotated footage.

Compares two identity decisions on identical frames and identical ground truth:

* **baseline** -- ``src/tracking.py::select_bowler_track_with_meta``, the legacy
  geometry + motion scorer.
* **role engine** -- ``src/cricket_roles.py::CricketRoleEngine``.

Ground truth comes from ``evaluation/build_bowler_gt.py`` (PER_DELIVERY policy), so
both systems are scored against the same conservative denominator.

The distribution matters as much as the mean. The failure this work targets is
*bimodal*: a correct pick has high IoU, a wrong pick has IoU of exactly 0
because it is a different person. A mean IoU hides that, so ``exact_zero_rate``
is reported alongside the mean and median.

Only the identity decision is measured here -- detection, tracking and role
assignment. Pose/biomechanics accuracy is NOT measured: there is no human
per-delivery biomechanical ground truth in this repository (see
``bowler_identity_audit.md``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from src import cricket_roles, tracking  # noqa: E402

GT_DIR = REPO / "evaluation" / "bowler_identification" / "ground_truth"
VIDEO_DIR = REPO / "evaluation" / "single_video_benchmark" / "dataset" / "videos"
OUT_DIR = REPO / "evaluation" / "bowler_identification"

IOU_MATCH = 0.5


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def frames_to_process(gt: dict, n_video_frames: int, pad: int) -> List[int]:
    """Frame indices worth decoding: each delivery window plus run-up context.

    The run-up starts before the annotated bowling window, so a small lead-in is
    included; without it the tracker never sees the approach that identifies the
    bowler.
    """
    groups = gt.get("delivery_groups") or []
    if not groups:
        return []
    wanted: set = set()
    for window in groups:
        lo = max(0, window[0] - pad)
        hi = min(n_video_frames - 1, window[-1] + pad)
        wanted.update(range(lo, hi + 1))
    return sorted(wanted)


def decode(path: Path, wanted: List[int]) -> Tuple[Dict[int, "np.ndarray"], Tuple[int, int]]:
    """Decode only the requested frames, reading the file once, sequentially."""
    want = set(wanted)
    out: Dict[int, "np.ndarray"] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last = wanted[-1] if wanted else 0
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            out[idx] = frame
        idx += 1
    cap.release()
    return out, (height, width)


def remap_track_indices(tracks: Dict[int, tracking.Track],
                        original: List[int]) -> Dict[int, tracking.Track]:
    """Undo ByteTrack's 0-based renumbering back to original video frame indices.

    ``_track_bytetrack_frames`` enumerates its results from 0, so when a frame
    subset is fed in, track frame ids are positions in ``original``, not video
    frame numbers. Without this remap every comparison would silently miss.
    """
    remapped: Dict[int, tracking.Track] = {}
    for tid, tr in tracks.items():
        new = tracking.Track(track_id=tr.track_id)
        for pos, bbox in zip(tr.frames, tr.bboxes):
            if 0 <= pos < len(original):
                new.frames.append(original[pos])
                new.bboxes.append(bbox)
        if new.frames:
            remapped[tid] = new
    return remapped


def decide_baseline(tracks, frame_dims, n_total) -> Optional[int]:
    try:
        bowler, meta = tracking.select_bowler_track_with_meta(
            tracks, frame_dims=frame_dims, total_frames=n_total)
    except Exception:
        return None
    if bowler is None:
        return None
    return bowler.track_id


def decide_engine(tracks, n_total, lower_half_weight: float = 1.0):
    weights = cricket_roles.RoleWeights()
    weights.lower_half_prior = lower_half_weight
    engine = cricket_roles.CricketRoleEngine(weights=weights)
    obs = cricket_roles.observations_from_tracks(tracks, total_frames=n_total)
    engine.extend(o for group in obs.values() for o in group)
    assignment = engine.assign(frame_count=n_total)
    return assignment.bowler_track_id, assignment


def score_against_gt(track: Optional[tracking.Track], gt_boxes: Dict[str, list]) -> dict:
    """Per-frame IoU of one selected track against the GT bowler."""
    ious: List[float] = []
    present = 0
    if track is not None:
        by_frame = dict(zip(track.frames, track.bboxes))
        for frame_str, gt_box in gt_boxes.items():
            f = int(frame_str)
            if f in by_frame:
                present += 1
                ious.append(iou(by_frame[f], gt_box))
    total = len(gt_boxes)
    matched = sum(1 for v in ious if v >= IOU_MATCH)
    exact_zero = sum(1 for v in ious if v == 0.0)
    return {
        "gt_frames": total,
        "frames_with_prediction": present,
        "detection_rate": round(present / total, 4) if total else 0.0,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "precision_matched": round(matched / present, 4) if present else 0.0,
        "mean_iou": round(statistics.fmean(ious), 4) if ious else 0.0,
        "median_iou": round(statistics.median(ious), 4) if ious else 0.0,
        "exact_zero_rate": round(exact_zero / total, 4) if total else 0.0,
        "matched_frames": matched,
        "exact_zero_frames": exact_zero,
        "_ious": ious,
    }


def aggregate(per_video: List[dict], system: str) -> dict:
    """Pool frames across videos. Mean-of-video means would hide per-frame scale."""
    ious: List[float] = []
    gt_total = pred_total = matched_total = zero_total = 0
    for v in per_video:
        s = v[system]
        ious.extend(s.pop("_ious", []))
        gt_total += s["gt_frames"]
        pred_total += s["frames_with_prediction"]
        matched_total += s["matched_frames"]
        zero_total += s["exact_zero_frames"]
    return {
        "videos": len(per_video),
        "gt_frames": gt_total,
        "frames_with_prediction": pred_total,
        "detection_rate": round(pred_total / gt_total, 4) if gt_total else 0.0,
        "match_rate": round(matched_total / gt_total, 4) if gt_total else 0.0,
        "precision_matched": round(matched_total / pred_total, 4) if pred_total else 0.0,
        "mean_iou": round(statistics.fmean(ious), 4) if ious else 0.0,
        "median_iou": round(statistics.median(ious), 4) if ious else 0.0,
        "exact_zero_rate": round(zero_total / gt_total, 4) if gt_total else 0.0,
    }


def evaluate_video(path: Path, pad: int, ablation: bool) -> Optional[dict]:
    gt_path = GT_DIR / f"{path.stem}.json"
    if not gt_path.exists():
        return None
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_boxes = gt.get("bowler_bboxes") or {}
    if not gt_boxes:
        return None

    cap = cv2.VideoCapture(str(path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_video <= 0:
        return None

    wanted = frames_to_process(gt, n_video, pad)
    if not wanted:
        return None
    frames, (h, w) = decode(path, wanted)
    if not frames:
        return None

    tracker = tracking.BowlerTracker()
    raw = tracker.track_frames([(i, frames[i]) for i in wanted])
    tracks = remap_track_indices(raw, wanted)
    if not tracks:
        return None

    n_total = max(wanted) + 1
    base_id = decide_baseline(tracks, (h, w), n_total)
    eng_id, assignment = decide_engine(tracks, n_total)
    eng_id_nolower, _ = decide_engine(tracks, n_total, lower_half_weight=0.0) if ablation else (None, None)

    record = {
        "video_id": path.stem,
        "frames_processed": len(wanted),
        "tracks": len(tracks),
        "gt_frames": len(gt_boxes),
        "deliveries": gt.get("detected_deliveries", 0),
        "baseline_track_id": base_id,
        "engine_track_id": eng_id,
        "engine_state": assignment.state.value if assignment else None,
        "engine_confirmed": bool(assignment.bowler_confirmed) if assignment else False,
        "engine_identity_confidence": (
            round(assignment.identity_confidence, 4) if assignment else 0.0
        ),
        "baseline": score_against_gt(tracks.get(base_id) if base_id is not None else None, gt_boxes),
        "role_engine": score_against_gt(tracks.get(eng_id) if eng_id is not None else None, gt_boxes),
    }
    if ablation:
        record["engine_no_lower_half"] = score_against_gt(
            tracks.get(eng_id_nolower) if eng_id_nolower is not None else None, gt_boxes)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    ap.add_argument("--window-pad", type=int, default=90,
                    help="frames of run-up context around each delivery window")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "bowler_identification_results.json")
    args = ap.parse_args()

    # Only videos that actually have ground truth can be scored. 7 of the 33
    # extracted videos ship without annotations, so they are excluded here rather
    # than silently diluting the result.
    videos = sorted(
        v for v in VIDEO_DIR.glob("*_2overs.mp4") if (GT_DIR / f"{v.stem}.json").exists()
    )
    if not videos:
        print(f"ERROR: no videos with ground truth under {GT_DIR}")
        print("Run evaluation/build_bowler_gt.py first.")
        return 1
    if args.limit:
        videos = videos[: args.limit]

    print(
        f"Evaluating {len(videos)} annotated videos "
        f"(window_pad={args.window_pad}, ablation={'off' if args.no_ablation else 'on'})\n"
    )
    per_video: List[dict] = []
    for i, video in enumerate(videos, 1):
        t0 = time.perf_counter()
        try:
            rec = evaluate_video(video, args.window_pad, not args.no_ablation)
        except Exception as exc:
            print(f"  [{i}/{len(videos)}] {video.stem}: ERROR {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        if rec is None:
            print(f"  [{i}/{len(videos)}] {video.stem}: skipped (no usable GT)")
            continue
        per_video.append(rec)
        b, e = rec["baseline"], rec["role_engine"]
        print(
            f"  [{i}/{len(videos)}] {video.stem}  ({dt:.0f}s, "
            f"{rec['frames_processed']} frames, {rec['tracks']} tracks)\n"
            f"          baseline: track={rec['baseline_track_id']} "
            f"match={b['match_rate']:.1%} medianIoU={b['median_iou']:.3f} "
            f"zeroIoU={b['exact_zero_rate']:.1%}\n"
            f"          engine:   track={rec['engine_track_id']} "
            f"state={rec['engine_state']} match={e['match_rate']:.1%} "
            f"medianIoU={e['median_iou']:.3f} zeroIoU={e['exact_zero_rate']:.1%}"
        )

    if not per_video:
        print("No videos evaluated.")
        return 1

    results = {
        "description": (
            "A/B bowler-identification accuracy on human-annotated broadcast "
            "footage. Identity decisions only; no biomechanics accuracy is claimed."
        ),
        "ground_truth": "evaluation/build_bowler_gt.py (PER_DELIVERY policy)",
        "window_pad_frames": args.window_pad,
        "iou_match_threshold": IOU_MATCH,
        "systems": {
            "baseline": "src/tracking.py::select_bowler_track_with_meta",
            "role_engine": "src/cricket_roles.py::CricketRoleEngine",
        },
        "aggregate": {
            "baseline": aggregate(per_video, "baseline"),
            "role_engine": aggregate(per_video, "role_engine"),
        },
        "per_video": per_video,
    }
    if not args.no_ablation:
        results["aggregate"]["engine_no_lower_half"] = aggregate(
            per_video, "engine_no_lower_half")
        results["systems"]["engine_no_lower_half"] = (
            "CricketRoleEngine with LOWER_HALF_THRESHOLD prior weight set to 0"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    agg = results["aggregate"]
    print("\n=== AGGREGATE (pooled frames) ===")
    for name in ("baseline", "role_engine", "engine_no_lower_half"):
        if name not in agg:
            continue
        a = agg[name]
        print(
            f"  {name:22s} match={a['match_rate']:.1%}  "
            f"medianIoU={a['median_iou']:.3f}  meanIoU={a['mean_iou']:.3f}  "
            f"zeroIoU={a['exact_zero_rate']:.1%}  detect={a['detection_rate']:.1%}"
        )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
