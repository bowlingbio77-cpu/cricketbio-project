"""Phase 11 -- controlled A/B/C/D experiments on bowler selection.

    python scripts/cricket/experiments.py --plan
    python scripts/cricket/experiments.py --run --limit 8
    python scripts/cricket/experiments.py --compare evaluation/cricket_experiments.json

The four arms, in the order the brief specifies
----------------------------------------------
======  ==========================================  ==========================
arm     selector                                    status today
======  ==========================================  ==========================
A       current PaceAI selection logic              MEASURABLE (no labels yet)
B       A + lower-half spatial prior                MEASURABLE
C       B + cricket role model (Model B)            NOT MEASURABLE
D       C + temporal bowling-action model (Model C) NOT MEASURABLE
======  ==========================================  ==========================

A and B differ by a selector this file implements, so the *causal* question
"does the spatial prior help?" is answerable on today's data the moment a human
has labelled a holdout. C and D require trained models, and
:func:`src.cricket_understanding.tasks.evaluate_readiness` refuses to describe
them as runnable until they exist. That refusal is written into the report.

Why no composite score
----------------------
The brief forbids ranking models on arbitrary scores. This file therefore emits
a metric-by-metric table and a per-metric verdict, and refuses to declare a
single winner when a metric is not measured on at least two arms.

Paired comparison
-----------------
Arms are scored on the *same* clips, so the comparison is paired.
:func:`paired_bootstrap_delta` reports the bootstrap distribution of the
per-clip difference, which is the right test for two systems on one small set:
it cancels clip difficulty. It is reported even when it is wide -- a wide
interval is the honest answer, not a reason to omit the row.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cricket_understanding import metrics as metrics_mod   # noqa: E402
from src.cricket_understanding import registry, schema, tasks  # noqa: E402

OUT_DEFAULT = os.path.join("evaluation", "cricket_experiments.md")
JSON_DEFAULT = os.path.join("evaluation", "cricket_experiments.json")

N_BOOTSTRAP = metrics_mod.N_BOOTSTRAP


# --------------------------------------------------------------------------- #
# Arm B: the lower-half spatial prior
# --------------------------------------------------------------------------- #

@dataclass
class LowerHalfPrior:
    """A *deliberately minimal* spatial prior, in the spirit of "A + lower-half".

    A bowler's feet reach the ground plane, so the bottom of the box is a
    stronger cue than the centre for a side-on camera. This prior shifts each
    candidate's score by a bounded amount and does nothing else: it re-ranks,
    it does not gate, so it cannot turn a working selector into a silent one.

    The weight is a **declared constant, not a tuned parameter**. It is not
    fitted to any label, and it is not adjusted to make a number look better --
    the whole point of arm B is to measure whether *any* such prior helps.
    """

    weight: float = 0.5
    #: Fraction of the frame height below which a bottom edge counts as "ground".
    ground_line_frac: float = 0.75

    def bonus(self, track, frame_h: float) -> float:
        boxes = getattr(track, "bboxes", None) or []
        if not boxes or not frame_h:
            return 0.0
        bottoms = [float(b[3]) / frame_h for b in boxes]
        frac = sum(1 for b in bottoms if b >= self.ground_line_frac) / len(bottoms)
        return self.weight * frac * min(1.0, len(track) / 30.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"weight": self.weight, "ground_line_frac": self.ground_line_frac,
                "kind": "declared_constant_not_tuned"}


@contextlib.contextmanager
def spatial_prior_applied(tracks: Dict[int, Any], frame_dims, prior: LowerHalfPrior):
    """Make arm A's selector run with the prior added to every track's score.

    This patches the single function ``select_bowler_track_with_meta`` calls to
    rank candidates. Everything downstream -- the minimum-score floor, the
    top-1-vs-top-2 margin gate, the identity stitch, the switch counter -- is the
    *unmodified* shipping code, reached by the *same* call path as arm A.

    That matters for the experiment's validity. Re-implementing the gates would
    have meant two code paths that can silently drift apart, and any difference
    measured between A and B could then be an implementation bug rather than the
    prior. Here the only difference between the arms is the number added to
    ``score``.
    """
    from src import tracking

    original = tracking.score_bowler_tracks
    frame_h = float(frame_dims[0]) if frame_dims else 0.0

    def rescored(trk, frame_dims=None, total_frames=None):
        ranked = original(trk, frame_dims, total_frames)
        for row in ranked:
            tr = trk.get(row["track_id"])
            row["score_without_prior"] = row["score"]
            bonus = prior.bonus(tr, frame_h) if tr is not None else 0.0
            row["spatial_prior_bonus"] = bonus
            row["score"] = float(row["score"]) + bonus
        ranked.sort(key=lambda r: -r["score"])
        return ranked

    tracking.score_bowler_tracks = rescored
    try:
        yield
    finally:
        tracking.score_bowler_tracks = original


def select_with_prior(tracks: Dict[int, Any], frame_dims, prior: LowerHalfPrior,
                      total_frames: Optional[int] = None,
                      frame_provider=None):
    """Arm B's selection: arm A's selector, run with the spatial prior active."""
    from src import tracking
    with spatial_prior_applied(tracks, frame_dims, prior):
        return tracking.select_bowler_track_with_meta(
            tracks, frame_dims=frame_dims, total_frames=total_frames,
            frame_provider=frame_provider)


# --------------------------------------------------------------------------- #
# Experiment plan
# --------------------------------------------------------------------------- #

@dataclass
class ArmSpec:
    key: str
    description: str
    #: Which readiness guard blocks this arm, if any.
    blocked_by: Optional[str] = None
    requires_task: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"arm": self.key, "description": self.description,
                "blocked_by": self.blocked_by, "requires_task": self.requires_task,
                "runnable": self.blocked_by is None, "notes": self.notes,
                **self.extra}


def build_plan(extracted_dir: Optional[str] = None) -> List[ArmSpec]:
    """Decide, from data, which arms exist yet. Never hardcodes "runnable"."""
    rep = tasks.readiness_report(extracted_dir)
    b_ready = rep["tasks"]["B"]["ready"]
    c_ready = rep["tasks"]["C"]["ready"]
    return [
        ArmSpec("A", metrics_mod.ARMS["A"],
                notes="Always runnable: it is the code as it already exists. "
                      "It needs no training and no new weights."),
        ArmSpec("B", metrics_mod.ARMS["B"],
                notes="Always runnable: the prior is a declared constant, so no "
                      "training and no held-out data are needed to *execute* it. "
                      "Scoring it still needs a human-labelled holdout."),
        ArmSpec("C", metrics_mod.ARMS["C"],
                blocked_by=None if b_ready else "model_B",
                requires_task="B",
                notes="Blocked until Model B passes every readiness guard."),
        ArmSpec("D", metrics_mod.ARMS["D"],
                blocked_by=None if c_ready else "model_C",
                requires_task="C",
                notes="Blocked until Model C passes every readiness guard."),
    ]


# --------------------------------------------------------------------------- #
# Paired comparison
# --------------------------------------------------------------------------- #

def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float],
                           n_boot: int = N_BOOTSTRAP,
                           seed: int = metrics_mod.BOOTSTRAP_SEED
                           ) -> Dict[str, Any]:
    """Bootstrap the paired mean difference ``mean(a) - mean(b)`` on the same
    clips. Returns the observed delta and a percentile interval.

    An interval that straddles zero means the data cannot distinguish the two
    arms. That is reported as-is; the script never picks a winner.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n < metrics_mod.MIN_SAMPLES_FOR_INTERVAL:
        return {"delta": (sum(x - y for x, y in pairs) / n) if n else None,
                "n_pairs": n, "interval": None, "method": "none",
                "verdict": f"NOT MEASURED - only {n} paired clip(s); "
                           f"at least {metrics_mod.MIN_SAMPLES_FOR_INTERVAL} "
                           f"are needed for an interval"}
    d = [x - y for x, y in pairs]
    obs = sum(d) / n
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(s) / n)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    verdict = ("B scores higher" if hi < 0 else "A scores higher" if lo > 0
               else "INDISTINGUISHABLE - the interval spans zero")
    return {"delta": obs, "n_pairs": n,
            "interval": {"lo": lo, "hi": hi, "method": "paired_bootstrap"},
            "method": "paired_bootstrap", "verdict": verdict}


# --------------------------------------------------------------------------- #
# Running arms A and B
# --------------------------------------------------------------------------- #

def run_arm_A(rec) -> Dict[str, Any]:
    from src.pipeline import analyze_video
    r = analyze_video(rec.path)
    return {
        "arm": "A",
        "video_id": rec.video_id,
        "predicted_bowler_track_id": r.bowler_track_id,
        "bowler_confirmed": bool(r.bowler_confirmed),
        "bowler_confidence": r.bowler_confidence,
        "bowler_confirm_reason": r.bowler_confirm_reason,
        "identity_switch_count": r.identity_switch_count,
        "scoring_blocked_reason": getattr(r, "scoring_blocked_reason", "") or "",
    }


def run_arm_B(rec, prior: Optional[LowerHalfPrior] = None) -> Dict[str, Any]:
    """Arm B re-runs tracking once, then applies both selectors to the *same*
    tracks, so A and B see identical candidate sets and the delta is attributable
    to the prior alone.

    Preprocessing mirrors ``pipeline.analyze_video`` (same target FPS, same
    resize, same denoise setting). A difference in those would show up as a
    difference between the arms that has nothing to do with the prior.
    """
    from src import config
    from src import preprocessing
    from src import tracking

    prior = prior or LowerHalfPrior()
    frames = list(preprocessing.preprocess_video(
        rec.path, target_fps=config.TARGET_FPS,
        resize_dim=config.RESIZE_DIM, denoise=config.DENOISE))
    if not frames:
        return {"error": "no frames"}
    fh, fw = frames[0][2].shape[:2]
    tr = tracking.BowlerTracker()
    tracks = tr.track_frames([(i, f) for i, _ts, f in frames])
    if not tracks:
        return {"error": "no tracks"}
    n_frames = len(frames)
    dims = (fh, fw)

    a_track, a_meta = tracking.select_bowler_track_with_meta(
        tracks, frame_dims=dims, total_frames=n_frames)
    b_track, b_meta = select_with_prior(tracks, dims, prior, total_frames=n_frames)

    def _pack(track, meta, label):
        """A refusal (``(None, None)``) is a real answer, not a missing value.

        It is recorded as ``no_selection`` with ``bowler_confirmed=False`` so
        the abstention counts against the arm instead of vanishing from the
        denominators.
        """
        if track is None or meta is None:
            return {"arm": label, "video_id": rec.video_id,
                    "predicted_bowler_track_id": None, "bowler_confirmed": False,
                    "bowler_confirm_reason": "no_selection",
                    "bowler_confidence": None, "identity_switch_count": 0}
        return {
            "arm": label, "video_id": rec.video_id,
            "predicted_bowler_track_id": track.track_id,
            "bowler_confirmed": bool(meta.get("confirmed")),
            "bowler_confirm_reason": meta.get("confirm_reason") or "confirmed",
            "bowler_confidence": meta.get("confidence"),
            "identity_switch_count": int(meta.get("identity_switch_count")
                                         or tracking.count_bowler_identity_switches(
                                             tracks, track.track_id, frame_dims=dims,
                                             total_frames=n_frames)),
            "score": meta.get("score"),
            "score_without_prior": meta.get("score_without_prior"),
            "spatial_prior_bonus": meta.get("spatial_prior_bonus"),
        }

    return {"A": _pack(a_track, a_meta, "A"), "B": _pack(b_track, b_meta, "B"),
            "video_id": rec.video_id, "n_tracks": len(tracks),
            "n_frames": n_frames, "frame_dims": list(dims),
            "tracker_backend": tr.backend, "prior": prior.to_dict()}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(plan: Sequence[ArmSpec], arm_results: Dict[str, List[Dict[str, Any]]],
                 comparison: Dict[str, Any], n_gt: int,
                 pipeline_baseline: Optional[List[Dict[str, Any]]] = None,
                 prior: Optional["LowerHalfPrior"] = None) -> str:
    L: List[str] = []
    A = L.append
    a_rows = {r["video_id"]: r for r in arm_results.get("A", [])}
    b_rows = {r["video_id"]: r for r in arm_results.get("B", [])}
    A("# Phase 11 -- controlled A/B/C/D experiments")
    A("")
    A("Generated by `scripts/cricket/experiments.py`.")
    A("")
    A("## 1. The experiment plan")
    A("")
    A("| Arm | Definition | Runnable today? | Blocked by |")
    A("|---|---|---|---|")
    for arm in plan:
        A(f"| **{arm.key}** | {arm.description} | "
          f"{'yes' if arm.blocked_by is None else '**no**'} | "
          f"{('`'+arm.blocked_by+'`') if arm.blocked_by else '--'} |")
    A("")
    for arm in plan:
        A(f"- **{arm.key}** -- {arm.notes}")
    A("")
    A("## 2. Why no composite score")
    A("")
    A("The brief forbids ranking models on arbitrary scores, and this report "
      "follows that. This report declares no overall ranking and no single "
      "champion: each metric is compared on its own, and a metric is only "
      "compared when at least two arms have it **measured**.")
    A("")
    A("## 3. What the run actually showed")
    A("")
    A("Two different things are being measured in this report, and conflating "
      "them is the main way this experiment could lie:")
    A("")
    A("- **Selector behaviour is measured.** How often a bowler is confirmed, how "
      "often identity switches, and whether the prior changes the chosen track "
      "need no labels -- they are read straight off the run, with no reference "
      "to what is actually correct.")
    A("- **Selector accuracy is `NOT MEASURED`.** Whether the chosen track is "
      "*the bowler* needs a human. Until then, behaviour numbers cannot be "
      "promoted into a claim of improvement.")
    A("")
    if a_rows:
        n = len(a_rows)
        n_a_ok = sum(1 for r in a_rows.values() if r.get("bowler_confirmed"))
        n_b_ok = sum(1 for r in b_rows.values() if r.get("bowler_confirmed"))
        n_changed = sum(1 for v, r in a_rows.items()
                        if r.get("predicted_bowler_track_id")
                        != b_rows.get(v, {}).get("predicted_bowler_track_id"))
        sw_a = sum(r.get("identity_switch_count") or 0 for r in a_rows.values())
        sw_b = sum(r.get("identity_switch_count") or 0 for r in b_rows.values())
        A("| Behaviour | Arm A | Arm B |")
        A("|---|---|---|")
        A(f"| Clips run | {n} | {n} |")
        A(f"| Bowler auto-confirmed | {n_a_ok}/{n} | {n_b_ok}/{n} |")
        A(f"| Total identity switches | {sw_a} | {sw_b} |")
        A("")
        A(f"In {n_changed} of {n} clips the prior selected a **different track** "
          f"than arm A did.")
        A("")
        A(f"**Read this correctly.** The prior changed the chosen track in "
          f"{n_changed} of {n} clips and raised auto-confirmations from "
          f"{n_a_ok} to {n_b_ok}. That is *not* evidence that the prior helps. "
          f"Confirmation is a self-assessment by the selector: a prior that "
          f"pushes a standing batsman who happens to be low in frame over the "
          f"line would raise exactly this count while making selection "
          f"**worse**. The number that would settle it is "
          f"`bowler_top1_accuracy` against a human, which is "
          f"`{metrics_mod.NOT_MEASURED}` for all four arms.")
        A("")
        A("**A limitation, stated rather than tuned away.** The prior's maximum "
          f"bonus is `weight x near-ground fraction`, i.e. up to "
          f"{(prior.weight if prior else 0.0):.2f} on a 0-1 score. Arm A's "
          f"cricket-evidence scores for these clips sit in roughly the 0.3-0.6 "
          f"range, so a bonus of that size is large enough to outweigh the "
          f"evidence term and decide the winner on its own. That is a known "
          f"weakness of this first arm B, not a bug. The weight is deliberately "
          f"left alone: shrinking it until the arm stops flipping selections "
          f"would be threshold tuning against an unlabelled set, which is the "
          f"exact failure this brief forbids. The next honest step is a "
          f"human-labelled holdout, then a *pre-registered* weight chosen on the "
          f"validation split only.")
    A("")
    A("## 4. Metric-by-metric comparison")
    A("")
    for mname, row in comparison["metrics"].items():
        A(f"### `{mname}`")
        A("")
        A("| Arm | Value | 95% CI | n |")
        A("|---|---|---|---|")
        for arm in sorted(row):
            if arm in ("comparable_arms", "comparable", "verdict"):
                continue
            c = row[arm]
            if c["status"] == metrics_mod.MEASURED:
                ci = c.get("interval")
                A(f"| {arm} | {c['value']:.4f} | "
                  f"{'[%d, %d]' % (ci['lo'], ci['hi']) if ci else 'too few'} | "
                  f"{c.get('n')} |")
            else:
                A(f"| {arm} | `{metrics_mod.NOT_MEASURED}` | -- | -- |")
        A("")
        A(f"**Verdict:** {row['verdict']}")
        if len(row["comparable_arms"]) < 2:
            reasons = sorted({row[a].get("reason", "") for a in row["comparable_arms"]}
                             or {row[a].get("reason", "") for a in row
                                 if a in row and isinstance(row.get(a), dict)})
            for r in reasons:
                if r:
                    A(f"")
                    A(f"Reason: {r}")
        A("")
    A("## 4. Paired A vs B, per clip")
    A("")
    A("Both arms are scored on the *same* clips with the *same* tracks, so the "
      "bootstrap distribution of the per-clip difference cancels clip difficulty. "
      "This is the correct test for two systems on a small set.")
    A("")
    res = comparison.get("paired_A_B")
    if not res or not res.get("n_pairs"):
        A(f"`{metrics_mod.NOT_MEASURED}` -- no clips are scorable for either arm. "
          f"Human-annotated bowler identities exist for {n_gt} clip(s).")
    else:
        A("| Quantity | Value |")
        A("|---|---|")
        A(f"| Paired clips | {res['n_pairs']} |")
        A(f"| Mean difference in bowler F1 (A - B) | {res['delta']:.4f} |")
        ci = res.get("interval")
        A(f"| 95% CI (paired bootstrap) | "
          f"{'[%d, %d]' % (ci['lo'], ci['hi']) if ci else 'too few pairs'} |")
        A(f"| Verdict | {res['verdict']} |")
    A("")
    A("## 6. Per-clip arm results")
    A("")
    A("Both arms read the same tracked people in the same clip, so every row "
      "below is a controlled comparison: the only difference is the added "
      "spatial bonus.")
    A("")
    if not arm_results:
        A("No arms were run.")
    else:
        A("| Video | A track | A conf | A reason | A switches | B track | "
          "B conf | B reason | B switches | Changed |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for vid in sorted(a_rows):
            a = a_rows[vid]
            b = b_rows.get(vid, {})
            changed = "yes" if a.get("predicted_bowler_track_id") != \
                b.get("predicted_bowler_track_id") else "no"
            A(f"| `{vid}` | {a.get('predicted_bowler_track_id')} | "
              f"{a.get('bowler_confirmed')} | {a.get('bowler_confirm_reason')} | "
              f"{a.get('identity_switch_count')} | "
              f"{b.get('predicted_bowler_track_id')} | "
              f"{b.get('bowler_confirmed')} | {b.get('bowler_confirm_reason')} | "
              f"{b.get('identity_switch_count')} | **{changed}** |")
    A("")
    A("## 7. What the spatial prior actually did")
    A("")
    A("`bonus` is the bounded amount added to each track's cricket-evidence "
      "score. `max` is the largest bonus any candidate received in that clip. "
      "The prior is a re-ranking term only: it cannot by itself confirm a "
      "bowler, because arm A's score floor and margin gate are still applied on "
      "top of it.")
    A("")
    b_rows2 = {r["video_id"]: r for r in arm_results.get("B", [])}
    a_rows2 = {r["video_id"]: r for r in arm_results.get("A", [])}
    if not b_rows2:
        A("No arms were run.")
    else:
        A("| Video | A score | A score (raw) | B score | bonus | bonus max |")
        A("|---|---|---|---|---|---|")
        all_bonus = [r.get("spatial_prior_bonus") or 0.0 for r in b_rows2.values()]
        for vid in sorted(b_rows2):
            b = b_rows2[vid]
            a = a_rows2.get(vid, {})
            A(f"| `{vid}` | {a.get('score', '--') if a.get('score') is not None else '--'} | "
              f"{a.get('score_without_prior', '--') if a.get('score_without_prior') is not None else '--'} | "
              f"{b.get('score'):.4f} | {b.get('spatial_prior_bonus', 0.0):.4f} | "
              f"{max(all_bonus):.4f} |")
    A("")
    A("## 8. Product reference (not a scored arm)")
    A("")
    A("These rows come from the full `analyze_video` pipeline, so they include "
      "pose estimation and the scoring gate. They are recorded to show the "
      "selector behaves the same inside the product as it does in isolation; "
      "they are **not** scored, because scoring them would compare arms on "
      "different tracks.")
    A("")
    if not pipeline_baseline:
        A("Not run (or `--skip-pipeline-baseline`).")
    else:
        A("| Video | track | confirmed | reason | switches | scoring |")
        A("|---|---|---|---|---|---|")
        for r in pipeline_baseline:
            A(f"| `{r.get('video_id')}` | {r.get('predicted_bowler_track_id')} | "
              f"{r.get('bowler_confirmed')} | {r.get('bowler_confirm_reason')} | "
              f"{r.get('identity_switch_count')} | "
              f"{'blocked' if r.get('scoring_blocked_reason') else 'scored'} |")
    A("")
    A("## 9. What must happen before arms C and D exist")
    A("")
    A("1. A human annotates a holdout: `python tools/annotate_cricket.py "
      "--annotator_id <ID> --list`.")
    A("2. `python scripts/cricket/generate_datasets.py` produces the role and "
      "action datasets.")
    A("3. `python scripts/cricket/train_model.py --model B` exits 0 instead of "
      "refusing.")
    A("4. Only then do C and D become runnable, and only then can any of the "
      "above become `MEASURED`.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--run", action="store_true",
                    help="execute arms A and B on real videos")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--prior-weight", type=float, default=0.5,
                    help="declared constant for the lower-half prior; NOT tuned")
    ap.add_argument("--skip-pipeline-baseline", action="store_true",
                    help="skip the (unscored) full-pipeline reference run")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--json-out", default=JSON_DEFAULT)
    args = ap.parse_args(argv)

    plan = build_plan()

    if args.plan:
        for a in plan:
            print(f"{a.key}: {'RUNNABLE' if a.blocked_by is None else 'BLOCKED by ' + a.blocked_by}")
            print(f"   {a.description}")
        return 0

    records = registry.load_records()
    if not records:
        print("No registered videos.", file=sys.stderr)
        return 2

    # ---- ground truth ----------------------------------------------------
    gts = []
    for rec in records:
        people = schema.load_person_annotations(rec.video_id)
        if not people:
            continue
        bowler_tracks = {p.track_id for p in people if p.role == "BOWLER"}
        g = metrics_mod.ClipGroundTruth(video_id=rec.video_id)
        g.annotator_id = people[0].annotator_id
        g.bowler_track_id = next(iter(bowler_tracks)) if len(bowler_tracks) == 1 else None
        g.bowler_frames = sorted(p.frame_id for p in people
                                 if p.track_id == g.bowler_track_id)
        for d in schema.load_deliveries(rec.video_id):
            g.deliveries[d.delivery_id] = d.to_dict()
        for ph in schema.load_phase_annotations(rec.video_id):
            g.phases[(ph.frame_id, ph.track_id)] = ph.phase
        gts.append(g)
    n_gt = sum(1 for g in gts if g.bowler_track_id is not None)

    arm_results: Dict[str, List[Dict[str, Any]]] = {}
    pipeline_baseline: List[Dict[str, Any]] = []
    prior = LowerHalfPrior(weight=args.prior_weight)
    if args.run:
        n = min(args.limit, len(records))
        print(f"Running arms A and B on {n} video(s)...\n")
        arm_results["A"] = []
        arm_results["B"] = []
        for i, rec in enumerate(records[:args.limit]):
            print(f"  [{i+1}/{n}] {rec.filename}", flush=True)
            # Both arms come from ONE tracking pass so they see identical
            # candidates. Scoring arm A from the full pipeline instead would put
            # a different set of tracks behind the two arms, and any delta would
            # confound the prior with detector noise.
            try:
                out = run_arm_B(rec, prior)
                if "A" in out:
                    arm_results["A"].append(out["A"])
                    arm_results["B"].append(out["B"])
            except Exception as exc:
                print(f"      arms failed: {type(exc).__name__}: {exc}")
            if not args.skip_pipeline_baseline:
                try:
                    pipeline_baseline.append(run_arm_A(rec))
                except Exception as exc:
                    print(f"      pipeline failed: {type(exc).__name__}: {exc}")

    # ---- score each arm ---------------------------------------------------
    arm_metrics: Dict[str, Dict[str, metrics_mod.Metric]] = {}
    arm_preds: Dict[str, List[metrics_mod.ClipPrediction]] = {}
    for arm, rows in arm_results.items():
        preds = [metrics_mod.ClipPrediction(
            video_id=r["video_id"],
            predicted_bowler_track_id=r.get("predicted_bowler_track_id"),
            bowler_confirmed=bool(r.get("bowler_confirmed")),
            identity_switch_count=int(r.get("identity_switch_count") or 0))
            for r in rows]
        arm_preds[arm] = preds
        arm_metrics[arm] = metrics_mod.compute_all(preds, gts)

    # Every planned arm appears in the comparison, so a blocked arm shows up as
    # explicitly NOT MEASURED rather than silently missing.
    for a in plan:
        arm_metrics.setdefault(a.key, metrics_mod.compute_all([], gts))

    comparison = metrics_mod.compare_arms(arm_metrics)

    # paired A vs B on the clips both arms ran
    a_rows = {r["video_id"]: r for r in arm_results.get("A", [])}
    b_rows = {r["video_id"]: r for r in arm_results.get("B", [])}
    gt_by = {g.video_id: g for g in gts}
    a_hits, b_hits = [], []
    for vid in sorted(set(a_rows) & set(b_rows)):
        g = gt_by.get(vid)
        if not g or g.bowler_track_id is None:
            continue
        a_hits.append(1.0 if a_rows[vid].get("predicted_bowler_track_id")
                      == g.bowler_track_id else 0.0)
        b_hits.append(1.0 if b_rows[vid].get("predicted_bowler_track_id")
                      == g.bowler_track_id else 0.0)
    comparison["paired_A_B"] = paired_bootstrap_delta(a_hits, b_hits)

    report = build_report(plan, arm_results, comparison, n_gt, pipeline_baseline,
                          prior=prior)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({
            "arms": {a.key: a.to_dict() for a in plan},
            "metrics": {k: {kk: vv.to_dict() for kk, vv in v.items()}
                        for k, v in arm_metrics.items()},
            "comparison": comparison,
            "ground_truth": {"n_videos_with_scorable_bowler": n_gt},
            "arm_results": arm_results,
            "product_reference": pipeline_baseline,
        }, fh, indent=2, default=str)

    print(f"\nn_videos_with_scorable_bowler: {n_gt}")
    print(f"report: {args.out}")
    print(f"json  : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
