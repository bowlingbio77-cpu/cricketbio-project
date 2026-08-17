"""Evaluate ball-tracking accuracy against ground truth.

For every clip dir under data/gt_clips/ (clip.avi + gt.json [+ spec.json]):
  - spec.json present -> "oracle" mode: feed the tracker a scripted detector
    that reports the true ball position (with controlled conf / misses), so we
    measure the TRACKER's quality independent of YOLO.
  - spec.json absent   -> "yolo" mode: run the real YOLO detector end-to-end.
Real labeled clips (from scripts/label_ball.py) live in the same layout; when
clip.avi is missing, eval reads the source video path from source_path.txt and
preprocesses it exactly as the pipeline does.

Metrics per clip (GT frames = frames where the ball is visible):
  recall@25 / recall@15 : fraction of GT frames whose track point is within
                          25 / 15 px of the labeled ball centre
  err_mean / err_med    : L2 distance on GT frames that have a track point
  release_err/impact_err: |track - GT| release/impact frame (None if absent)
  outcome               : stats["outcome"] from the tracker

Usage: python scripts/evaluate_ball_tracking.py [--dir data/gt_clips]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import ball_tracking as bt1          # noqa: E402
from src import ball_tracking_v2 as bt2       # noqa: E402
from src import preprocessing                 # noqa: E402


class _Box:
    def __init__(self, cx, cy, half, conf):
        self.xyxy = np.array([[cx - half, cy - half, cx + half, cy + half]])
        self.conf = [conf]


class _Result:
    def __init__(self, box):
        self.boxes = [box] if box else []


class OracleDetector:
    """Scripted YOLO that reports the true ball with a realistic confidence /
    miss pattern (see spec.json). Occasionally emits a sub-seed detection."""

    def __init__(self, gt_frames, spec):
        self.gt = {int(k): v for k, v in gt_frames.items()}
        self.det_conf = spec.get("det_conf", 0.75)
        self.conf_jitter = spec.get("conf_jitter", 0.18)
        self.miss_prob = spec.get("miss_prob", 0.1)
        self.low_conf_prob = spec.get("low_conf_prob", 0.18)
        self.rng = random.Random(spec.get("seed", 0))
        self.i = 0

    def predict(self, frame, classes=None, conf=None, verbose=False):
        idx = self.i
        self.i += 1
        if idx not in self.gt:
            return [_Result(None)]
        roll = self.rng.random()
        if roll < self.miss_prob:
            return [_Result(None)]
        if roll < self.miss_prob + self.low_conf_prob:
            c = self.rng.uniform(0.2, 0.29)     # below seed gate: confirms only
        else:
            c = max(0.1, self.det_conf + self.rng.uniform(-self.conf_jitter, self.conf_jitter))
        x, y = self.gt[idx]
        return [_Result(_Box(x, y, half=5, conf=c))]


def load_frames(clip_dir):
    """Return a list of (idx, ts, frame) at tracker coordinate space."""
    vid = os.path.join(clip_dir, "clip.avi")
    if not os.path.exists(vid):
        src = os.path.join(clip_dir, "source_path.txt")
        if not os.path.exists(src):
            raise FileNotFoundError(f"neither clip.avi nor source_path.txt in {clip_dir}")
        with open(src) as f:
            vid = f.read().strip()
    return list(preprocessing.preprocess_video(vid))


def run_tracker(clip_dir, mod, model):
    frames = load_frames(clip_dir)
    tracker = mod.BallTracker(model=model)
    track, stats = tracker.track(frames, fps=20.0)
    return frames, track, stats


def clip_metrics(track, stats, gt):
    gt_frames = {int(k): v for k, v in gt["frames"].items()}
    by_idx = {p.frame_idx: p for p in track}
    dists = []
    for idx, (x, y) in gt_frames.items():
        p = by_idx.get(idx)
        if p is not None:
            dists.append(float(np.hypot(p.x - x, p.y - y)))
    dists = np.array(dists)
    recalls = {}
    for t in (15, 25):
        recalls[f"recall@{t}"] = float(np.sum(dists <= t)) / len(gt_frames) if gt_frames else 0.0
    m = {
        "n_gt": len(gt_frames),
        "n_track": len(track),
        "err_mean": float(np.mean(dists)) if len(dists) else None,
        "err_med": float(np.median(dists)) if len(dists) else None,
        "release": stats.get("release_idx"),
        "impact": stats.get("impact_idx"),
        "outcome": stats.get("outcome", "?"),
    }
    m.update(recalls)
    gt_rel, gt_imp = gt.get("release_idx"), gt.get("impact_idx")
    m["release_err"] = abs(m["release"] - gt_rel) if (m["release"] is not None and gt_rel is not None) else None
    if gt_imp is not None:
        m["impact_err"] = abs(m["impact"] - gt_imp) if m["impact"] is not None else None
    else:
        m["impact_err"] = None
    return m


def load_yolo():
    try:
        from ultralytics import YOLO
        from src.ball_tracking import resolve_weights
        return YOLO(resolve_weights())
    except Exception as exc:  # pragma: no cover
        print(f"  [yolo mode unavailable: {exc}]")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("data", "gt_clips"))
    ap.add_argument("--trackers", default="v1,v2")
    args = ap.parse_args()

    dirs = sorted(d for d in os.listdir(args.dir)
                  if os.path.isdir(os.path.join(args.dir, d)))
    mods = {"v1": bt1, "v2": bt2}
    want = [m.strip() for m in args.trackers.split(",") if m.strip()]
    yolo_model = load_yolo()

    per = {m: [] for m in want}
    print(f"\n{'clip':22s}" + "".join(f"{m:>9s}" for m in want) + "  detail")
    print("-" * 40 + "-" * len(want) * 9)
    for d in dirs:
        clip_dir = os.path.join(args.dir, d)
        gt_path = os.path.join(clip_dir, "gt.json")
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        spec = os.path.join(clip_dir, "spec.json")
        oracle_mode = os.path.exists(spec)
        spec = json.load(open(spec)) if oracle_mode else {}
        row = [d]
        for m in want:
            if oracle_mode:
                model = OracleDetector(gt["frames"], spec)
            else:
                model = yolo_model
            if model is None:
                row.append("--")
                continue
            _frames, track, stats = run_tracker(clip_dir, mods[m], model)
            met = clip_metrics(track, stats, gt)
            per[m].append(met)
            row.append(f"{met['recall@25'] * 100:5.0f}%")
        print(f"{row[0]:22s}" + "".join(f"{c:>9s}" for c in row[1:]))
        for m in want:
            met = per[m][-1]
            def fmt(x, suffix=".1f"):
                return "-" if x is None else f"{x:{suffix}}"
            print(f"    [{m}] r15={met['recall@15']*100:.0f}% "
                  f"err_mean={fmt(met['err_mean'])} err_med={fmt(met['err_med'])} "
                  f"rel={met['release']}->gt{gt.get('release_idx')} "
                  f"imp={met['impact']}->gt{gt.get('impact_idx')} "
                  f"outcome={met['outcome']}")

    print("\n==== aggregate ====")
    for m in want:
        if not per[m]:
            continue
        rs25 = [x["recall@25"] for x in per[m]]
        rs15 = [x["recall@15"] for x in per[m]]
        errs = [x["err_mean"] for x in per[m] if x["err_mean"] is not None]
        relok = sum(1 for x in per[m]
                    if x["release_err"] is not None and x["release_err"] <= 2)
        impok = sum(1 for x in per[m]
                    if x["impact_err"] is not None and x["impact_err"] <= 2)
        n_rel = sum(1 for x in per[m] if x["release_err"] is not None)
        n_imp = sum(1 for x in per[m] if x["impact_err"] is not None)
        print(f"  [{m}] clips={len(per[m])} mean_recall@25={np.mean(rs25)*100:.1f}% "
              f"mean_recall@15={np.mean(rs15)*100:.1f}% "
              f"mean_err={np.mean(errs) if errs else float('nan'):.1f}px "
              f"release_ok={relok}/{n_rel} impact_ok={impok}/{n_imp}")
        for x in per[m]:
            print(f"      {x['outcome']:20s} {x}")


if __name__ == "__main__":
    main()
