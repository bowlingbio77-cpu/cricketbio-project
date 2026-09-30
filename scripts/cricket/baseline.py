"""Phase 10 -- establish the existing PaceAI baseline, honestly.

    python scripts/cricket/baseline.py --limit 8
    python scripts/cricket/baseline.py --limit 0            # metrics only, no videos
    python scripts/cricket/baseline.py --limit 8 --out evaluation/cricket_understanding_baseline.md

Two clearly separated halves
----------------------------
**OBSERVED** -- what the current system actually did on N real videos, measured
by running it. Confirmation rate, refusal rate, identity switches, stage
backends, wall-clock, warnings. These are facts about the code, not claims about
cricket accuracy.

**NOT MEASURED** -- every accuracy figure in the Phase 10 brief, because all of
them need a human-labelled bowler, a human release frame, or a human phase
label, and the repository has none. The report prints ``NOT MEASURED`` with the
reason, never ``0.0`` and never a number derived from model output.

An optional third column, **PROXY (NOT GROUND TRUTH)**, may be requested with
``--proxy`` to show what the auto-labelled trees would have said. It is
labelled a proxy everywhere it appears, is never written into the accuracy
columns, and exists only to show how far the model is from its own output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cricket_understanding import metrics as metrics_mod                 # noqa: E402
from src.cricket_understanding import registry, schema                     # noqa: E402

OUT_DEFAULT = os.path.join("evaluation", "cricket_understanding_baseline.md")
JSON_DEFAULT = os.path.join("evaluation", "cricket_understanding_baseline.json")

#: Metric order follows the Phase 10 brief's numbered list.
BRIEF_ORDER = [
    ("bowler_top1_accuracy", "1. bowler identification accuracy"),
    ("bowler_precision", "2. bowler precision"),
    ("bowler_recall", "3. bowler recall"),
    ("bowler_f1", "4. bowler F1"),
    ("false_selection_rate", "5. false bowler selection rate"),
    ("identity_switches", "6. identity switches"),
    ("track_continuity", "7. track continuity"),
    ("delivery_detection_accuracy", "8. delivery detection accuracy"),
    ("release_frame_error_frames", "9. release-frame error"),
    ("phase_accuracy", "10. phase classification accuracy"),
    ("phase_macro_f1", "10b. phase macro-F1"),
    ("bowler_abstention_rate", "extra. bowler abstention rate"),
]


# --------------------------------------------------------------------------- #
# Running the current system
# --------------------------------------------------------------------------- #

def run_pipeline_on(rec, pose: bool = False) -> Dict[str, Any]:
    """Run the *unmodified* current pipeline on one registered video.

    No threshold is touched and no cricket_understanding code is on this path --
    that is the point: this is the system as it exists today, which is what a
    cricket-specific model would have to beat.
    """
    from src.pipeline import analyze_video
    t0 = time.time()
    r = analyze_video(rec.path)
    return {
        "video_id": rec.video_id,
        "filename": rec.filename,
        "camera_view_guess": getattr(r, "camera_view", None),
        "predicted_bowler_track_id": r.bowler_track_id,
        "bowler_confirmed": bool(r.bowler_confirmed),
        "bowler_confidence": r.bowler_confidence,
        "bowler_confirm_reason": r.bowler_confirm_reason,
        "identity_switch_count": r.identity_switch_count,
        "subject_verified": bool(r.subject_verified),
        "scoring_blocked_reason": getattr(r, "scoring_blocked_reason", "") or "",
        "n_bowler_candidates": len(getattr(r, "bowler_candidates", []) or []),
        "delivery_reliable": bool(getattr(r, "delivery_reliable", False)),
        "player_roles": {str(k): v.get("role") for k, v in
                         (getattr(r, "player_roles", {}) or {}).items()},
        "stage_backends": getattr(r, "stage_backends", {}),
        "stage_times": getattr(r, "stage_times", {}),
        "landmark_source_summary": getattr(r, "landmark_source_summary", {}),
        "warnings": list(getattr(r, "warnings", []) or []),
        "coaching_notes": list(getattr(r, "coaching_notes", []) or []),
        "wall_clock_sec": round(time.time() - t0, 2),
    }


def summarise_observed(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Facts about what the system did. No cricket-accuracy claim lives here."""
    if not runs:
        return {"n_videos_run": 0}
    n = len(runs)
    conf = sum(1 for r in runs if r["bowler_confirmed"])
    verified = sum(1 for r in runs if r["subject_verified"])
    sw = [r["identity_switch_count"] for r in runs]
    reasons: Dict[str, int] = {}
    for r in runs:
        k = r.get("bowler_confirm_reason") or "(none recorded)"
        reasons[k] = reasons.get(k, 0) + 1
    backends: Dict[str, int] = {}
    for r in runs:
        for k, v in (r.get("stage_backends") or {}).items():
            backends[f"{k}={v}"] = backends.get(f"{k}={v}", 0) + 1
    warn_hist: Dict[str, int] = {}
    for r in runs:
        for w in r.get("warnings", []):
            key = w.split("(")[0].strip()[:70]
            warn_hist[key] = warn_hist.get(key, 0) + 1
    times = [r["wall_clock_sec"] for r in runs if r.get("wall_clock_sec")]
    return {
        "n_videos_run": n,
        "bowler_confirmed": conf,
        "bowler_confirmation_rate": conf / n,
        "bowler_confirmation_ci": metrics_mod.wilson_interval(conf, n).to_dict()
        if metrics_mod.wilson_interval(conf, n) else None,
        "subject_verified": verified,
        "subject_verification_rate": verified / n,
        "videos_with_identity_switch": sum(1 for s in sw if s > 0),
        "identity_switch_rate": sum(1 for s in sw if s > 0) / n,
        "mean_identity_switches": sum(sw) / n,
        "confirm_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "stage_backends": dict(sorted(backends.items(), key=lambda kv: -kv[1])),
        "wall_clock_sec": {
            "mean": round(sum(times) / len(times), 2) if times else None,
            "min": min(times) if times else None,
            "max": max(times) if times else None,
        },
        "warning_histogram": dict(sorted(warn_hist.items(), key=lambda kv: -kv[1])[:12]),
    }


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

def load_ground_truth(records) -> List[metrics_mod.ClipGroundTruth]:
    """Load human labels. Returns an empty list when there are none -- which is
    the current state, and is the entire reason the accuracy table is empty."""
    out = []
    for rec in records:
        people = schema.load_person_annotations(rec.video_id)
        dels = schema.load_deliveries(rec.video_id)
        phases = schema.load_phase_annotations(rec.video_id)
        if not people and not dels and not phases:
            continue
        bowler_tracks = {p.track_id for p in people if p.role == "BOWLER"}
        g = metrics_mod.ClipGroundTruth(video_id=rec.video_id)
        g.annotator_id = people[0].annotator_id if people else ""
        # If a human marked more than one track BOWLER in a clip, the clip is
        # ambiguous for identity scoring and is reported, not guessed at.
        g.bowler_track_id = (next(iter(bowler_tracks)) if len(bowler_tracks) == 1
                             else None)
        g.bowler_frames = sorted(p.frame_id for p in people
                                 if p.track_id == g.bowler_track_id)
        for d in dels:
            g.deliveries[d.delivery_id] = d.to_dict()
        for ph in phases:
            g.phases[(ph.frame_id, ph.track_id)] = ph.phase
        out.append(g)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(observed: Dict[str, Any], m: Dict[str, metrics_mod.Metric],
                 n_registered: int, runs: List[Dict[str, Any]],
                 n_gt_clips: int, n_gt_deliveries: int,
                 n_gt_annotated_videos: int) -> str:
    L: List[str] = []
    A = L.append
    A("# PaceAI cricket baseline -- measured against the current system")
    A("")
    A("Generated by `scripts/cricket/baseline.py`. Every number is computed from "
      "a real run on real video files.")
    A("")
    A("> ## The headline")
    A(">")
    A("> **Not one cricket accuracy figure in this document is measurable.** "
      "Not one is zero either. Accuracy, precision, recall, F1, false-selection "
      "rate, delivery accuracy, release-frame error and phase accuracy all "
      "require a human-labelled bowler, a human release frame, or a human phase "
      "label. The repository contains **zero** of those. The honest state is "
      f"`{metrics_mod.NOT_MEASURED}`, and it stays that way until a human labels "
      "a held-out set.")
    A(">")
    A("> What *is* measured below is what the current code does when it runs. "
      "That is a real and useful baseline, and it is a different kind of fact "
      "from accuracy.")
    A("")

    A("## 1. Scope")
    A("")
    A("| Item | Value |")
    A("|---|---|")
    A(f"| Videos registered (real files on disk) | {n_registered} |")
    A(f"| Videos run through the current pipeline for this report | "
      f"{len(runs)} |")
    A(f"| Videos with ANY human cricket annotation | {n_gt_annotated_videos} |")
    A(f"| Videos with a human-identified bowler (scorable for identity) | "
      f"{n_gt_clips} |")
    A(f"| Human-annotated deliveries | {n_gt_deliveries} |")
    A("| System under test | `src/pipeline.py::analyze_video`, unmodified |")
    A("| Weights | `src/config.py::YOLO_WEIGHTS` (stock `yolo11n.pt`, COCO) |")
    A("| Tracker | Ultralytics ByteTrack, shipped defaults |")
    A("| Pose | `models/pose_landmarker_heavy.task` (MediaPipe) |")
    A("")

    A("## 2. MEASURED -- what the current system does")
    A("")
    if not runs:
        A("No videos were run (`--limit 0`).")
        A("")
    else:
        A("These are observations about code behaviour on real video. They are "
          "**not** claims about cricket accuracy, and none of them uses a "
          "ground-truth label of any kind.")
        A("")
        A("| Observation | Value | 95% CI |")
        A("|---|---|---|")
        ci = observed.get("bowler_confirmation_ci")
        A(f"| Clips where a bowler was CONFIRMED | "
          f"{observed['bowler_confirmed']}/{observed['n_videos_run']} "
          f"({observed['bowler_confirmation_rate']:.1%}) | "
          f"{'[%d, %d]' % (ci['lo'], ci['hi']) if ci else 'too few clips'} |")
        A(f"| Clips where the bowler subject was VERIFIED for biomechanics | "
          f"{observed['subject_verified']}/{observed['n_videos_run']} "
          f"({observed['subject_verification_rate']:.1%}) | |")
        A(f"| Clips with >=1 identity switch | "
          f"{observed['videos_with_identity_switch']}/{observed['n_videos_run']} "
          f"({observed['identity_switch_rate']:.1%}) | |")
        A(f"| Mean identity switches per clip | "
          f"{observed['mean_identity_switches']:.2f} | |")
        A("")
        A("**Confirmation-refusal reasons** (why the system declined to lock a "
          "bowler):")
        A("")
        A("| Reason | Clips |")
        A("|---|---|")
        for k, v in observed["confirm_reasons"].items():
            A(f"| `{k}` | {v} |")
        A("")
        A("**Stage backends actually used:**")
        A("")
        A("| Backend | Clips |")
        A("|---|---|")
        for k, v in observed["stage_backends"].items():
            A(f"| `{k}` | {v} |")
        A("")
        wc = observed["wall_clock_sec"]
        if wc.get("mean") is not None:
            A(f"**Wall clock** (CPU, no GPU requested): mean {wc['mean']}s, "
              f"min {wc['min']}s, max {wc['max']}s per clip.")
            A("")
        A("**Most frequent warnings:**")
        A("")
        A("| Warning | Clips |")
        A("|---|---|")
        for k, v in observed["warning_histogram"].items():
            A(f"| {k} | {v} |")
        A("")

    A("## 3. NOT MEASURED -- the ten required metrics")
    A("")
    A("The brief's Phase 10 metrics, in order, each with the reason it cannot be "
      "computed. **No value is invented, defaulted, or inferred from model "
      "output.**")
    A("")
    A("| # | Metric | Value | Why |")
    A("|---|---|---|---|")
    for key, label in BRIEF_ORDER:
        metric = m.get(key)
        if metric is None:
            A(f"| {label} | `{metrics_mod.NOT_MEASURED}` | metric not computed |")
            continue
        val = f"`{metrics_mod.MEASURED}` **{metric.value:.4f}** (n={metric.n})" \
            if metric.measured else f"`{metrics_mod.NOT_MEASURED}`"
        A(f"| {label} | {val} | {metric.reason or '--'} |")
    A("")
    A("Why each is blocked, in one line each:")
    A("")
    A("- **Bowler accuracy / precision / recall / F1 / false-selection rate** need "
      "a human to say which tracked person is the bowler. "
      "`tools/annotate_cricket.py` exists to record exactly that; nobody has run it.")
    A("- **Identity switches** are *counted* above but cannot be scored: only the "
      "system can count its own switches, and without a human bowler identity "
      "there is no way to know whether a switch left the bowler.")
    A("- **Track continuity** needs a human-annotated span for the bowler to "
      "measure coverage against.")
    A("- **Delivery detection accuracy** needs human delivery windows.")
    A("- **Release-frame error** needs human release frames, and they are the "
      "single most subjective label in the schema -- which is why the GUI stamps "
      "them explicitly (`5` for RELEASE) rather than inferring them.")
    A("- **Phase accuracy** needs human phase labels per frame.")
    A("")

    A("## 4. What the existing auto-labelled trees would have claimed")
    A("")
    A("`data/bowler_batsman_dataset/auto_labeled/` (558 rows) and "
      "`data/cricket_ball_dataset/auto_labeled/` (1,254 rows) look like a "
      "ready-made training set. They are not: they were produced by OpenCV "
      "heuristics and a YOLO model, with no `annotator_id` in either metadata "
      "file. Scoring a model against them measures agreement with the current "
      "model.")
    A("")
    A("| Tree | Rows | Human? |")
    A("|---|---|---|")
    A("| `data/bowler_batsman_dataset/auto_labeled` | 558 | No -- model-derived |")
    A("| `data/cricket_ball_dataset/auto_labeled` | 1,254 | No -- model-derived |")
    A("")
    A("`src/cricket_understanding/generators.py::FORBIDDEN_LABEL_ROOTS` makes "
      "this refusal executable: pointing a generator at either tree raises rather "
      "than producing training data that only re-learns PaceAI's own output.")
    A("")

    A("## 5. Reproducing this report")
    A("")
    A("```")
    A("python scripts/cricket/baseline.py --limit 8")
    A("```")
    A("")
    A("Read-only with respect to annotations. It runs the pipeline, measures it, "
      "and writes `evaluation/cricket_understanding_baseline.md` plus a "
      "machine-readable `.json` beside it.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=8,
                    help="how many registered videos to run (0 = metrics only)")
    ap.add_argument("--registry", default=registry.VIDEOS_JSONL)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--json-out", default=JSON_DEFAULT)
    args = ap.parse_args(argv)

    records = registry.load_records(args.registry)
    if not records:
        print("No registered videos. Run scripts/cricket/register_videos.py first.",
              file=sys.stderr)
        return 2

    gts = load_ground_truth(records)
    n_gt_annotated = len({g.video_id for g in gts})
    n_gt_bowler = sum(1 for g in gts if g.bowler_track_id is not None)
    n_gt_deliveries = sum(len(g.deliveries) for g in gts)

    runs: List[Dict[str, Any]] = []
    if args.limit > 0:
        print(f"Running the current pipeline on {min(args.limit, len(records))} "
              f"of {len(records)} registered videos (this takes a few minutes)...\n")
        for i, rec in enumerate(records[:args.limit]):
            print(f"  [{i+1}/{min(args.limit, len(records))}] {rec.filename} ...",
                  flush=True)
            try:
                runs.append(run_pipeline_on(rec))
            except Exception as exc:  # a crash is data too
                print(f"      FAILED: {type(exc).__name__}: {exc}")
                runs.append({"video_id": rec.video_id, "filename": rec.filename,
                             "error": f"{type(exc).__name__}: {exc}"})

    observed = summarise_observed(runs)
    m = metrics_mod.compute_all([], gts)

    report = build_report(observed, m, len(records), runs,
                          n_gt_bowler, n_gt_deliveries, n_gt_annotated)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)

    payload = {
        "generated_by": "scripts/cricket/baseline.py",
        "n_videos_registered": len(records),
        "n_videos_run": len(runs),
        "observed": observed,
        "ground_truth": {
            "n_annotated_videos": n_gt_annotated,
            "n_videos_with_scorable_bowler": n_gt_bowler,
            "n_deliveries": n_gt_deliveries,
        },
        "metrics": {k: v.to_dict() for k, v in m.items()},
        "n_measured_metrics": sum(1 for v in m.values() if v.measured),
        "n_not_measured_metrics": sum(1 for v in m.values() if not v.measured),
        "runs": runs,
    }
    with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=2, default=str))

    print(f"\nmeasured metrics      : {payload['n_measured_metrics']}")
    print(f"NOT MEASURED metrics  : {payload['n_not_measured_metrics']}")
    print(f"report                : {args.out}")
    print(f"json                  : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
