"""
Phase 10 / 11 -- evaluation metrics for cricket understanding.

Design rules
------------
1. **A metric without ground truth is ``NOT MEASURED``.**  Never ``0.0``, never
   ``None`` rendered as a number, never a guess.  :class:`Metric` carries an
   explicit ``status`` so a report generator cannot accidentally print a number
   that was never computed.
2. **Uncertainty is reported, not implied.**  Every rate carries a Wilson score
   interval; F1/MAE carry a bootstrap interval.  A single point estimate on a
   handful of clips is reported *with* its interval so a reader can see it is
   meaningless rather than being misled by the number.
3. **No test-set tuning.**  Model selection and threshold choice use ``val``.
   ``test`` is read exactly once, at the end, by
   :func:`compare_arms`.  :func:`assert_no_test_selection` makes the discipline
   machine-checkable.
4. **Identity is scored separately from detection.**  A clip where the right
   person is found but the tracker fragments them is a different failure from
   picking the wrong person, and collapsing them hides which one to fix.

Metric definitions
------------------
``bowler_top1_accuracy``
    Fraction of clips where the selected track id equals the human bowler track
    id for that clip.
``bowler_precision``
    Over every clip that produced a selection, the fraction whose selection is
    the human bowler.  A clip with no selection counts against precision (the
    system abstained but occupied the decision slot), which is why
    ``false_selection_rate`` is reported alongside it.
``bowler_recall``
    Over every clip, the fraction where the human bowler was selected.
``bowler_f1``
    Harmonic mean of the above, computed from the pooled counts, not as the
    mean of per-clip F1s.
``false_selection_rate``
    Fraction of clips where a bowler was selected and it was NOT the human
    bowler.  This is the number that matters operationally: a wrong bowler means
    every downstream biomechanical number describes the wrong human.
``identity_switches``
    From ``src.tracking.count_bowler_identity_switches``.  Lower is better; the
    per-clip count and the clip-level rate are both reported.
``track_continuity``
    ``observed_frames / annotated_span_frames`` for the human bowler track --
    the fraction of the human-annotated span in which a track of that identity
    actually existed.  Detects fragmentation.
``delivery_detection_accuracy``
    Fraction of human deliveries matched one-to-one by a predicted delivery
    (IoU over frame windows >= :data:`DELIVERY_MATCH_IOU`).
``release_frame_error``
    MAE in frames between the predicted and human release frame, over matched
    deliveries only, reported with its bootstrap interval.
``phase_accuracy``
    Frame-level accuracy of the phase label over the bowler's annotated frames.
    Macro-F1 is reported too, because a phase model that always says
    ``RUN_UP`` can look accurate on a run-up-heavy corpus.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Two predicted/human delivery windows match when their temporal IoU is >= this.
DELIVERY_MATCH_IOU = 0.5

#: Bootstrapping resamples for interval estimation.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

#: Below this many paired samples an interval is not worth printing; the point
#: estimate is reported with an explicit "n too small" marker instead.
MIN_SAMPLES_FOR_INTERVAL = 5

MEASURED = "MEASURED"
NOT_MEASURED = "NOT MEASURED"

#: Substrings that mark an ``annotator_id`` as *not* a person. Ground truth means
#: a human made a decision; an id that names an automatic labeller, however
#: confident the file's provenance block claims to be, is a model output and can
#: never be scored as truth. This list is deliberately explicit rather than clever
#: -- a new automated labeller must be added here, not inferred at runtime.
NON_HUMAN_ANNOTATOR_MARKERS = (
    "auto", "model", "yolo", "mediapipe", "pase", "bot", "pseudo",
    "synthetic", "heuristic", "script", "track", "detector", "classifier",
)


def is_human_annotator(annotator_id: str) -> bool:
    """Is this id attributable to a person?

    Ground truth in this project is defined by who made the call, not by where
    the file lives. An annotation written by a model is a *prediction*, and
    scoring predictions against predictions yields a confident, meaningless
    number -- the single easiest way for this project to report a fabricated
    accuracy. An unrecognised id is accepted (people will be named in ways this
    list does not anticipate), but it is always echoed into the metric's
    ``extra`` block so the provenance stays auditable after the fact.
    """
    a = str(annotator_id or "").strip()
    if not a:
        return False
    low = a.lower()
    return not any(m in low for m in NON_HUMAN_ANNOTATOR_MARKERS)


def _human_gts(gts: Sequence["ClipGroundTruth"]) -> Tuple[Dict[str, "ClipGroundTruth"], List[str]]:
    """Split ground truth into human-attributable clips and the rest.

    The rejected ids are returned (not just counted) so a caller can print
    *which* annotation files were excluded and why.
    """
    human, rejected = {}, []
    for g in gts:
        if is_human_annotator(g.annotator_id):
            human[g.video_id] = g
        else:
            rejected.append(f"{g.video_id}:{g.annotator_id or '<empty>'}")
    return human, rejected


@dataclass
class Interval:
    lo: float
    hi: float
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {"lo": round(self.lo, 4), "hi": round(self.hi, 4), "method": self.method}


@dataclass
class Metric:
    """A single named measurement, honest about whether it exists."""

    name: str
    value: Optional[float] = None
    status: str = NOT_MEASURED
    reason: str = ""
    n: int = 0
    interval: Optional[Interval] = None
    unit: str = ""
    higher_is_better: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.status == MEASURED and self.value is not None

    def render(self) -> str:
        if not self.measured:
            return f"{NOT_MEASURED} - {self.reason}" if self.reason else NOT_MEASURED
        s = f"{self.value:.4f}".rstrip("0").rstrip(".")
        if self.unit:
            s += f" {self.unit}"
        s += f" (n={self.n}"
        if self.interval is not None:
            s += (f", 95% CI [{self.interval.lo:.3f}, {self.interval.hi:.3f}] "
                  f"{self.interval.method}")
        else:
            s += ", interval omitted: too few paired samples"
        s += ")"
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "value": self.value, "status": self.status,
            "reason": self.reason, "n": self.n, "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "interval": self.interval.to_dict() if self.interval else None,
            "extra": self.extra,
        }


# --------------------------------------------------------------------------- #
# Interval estimators
# --------------------------------------------------------------------------- #

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[Interval]:
    """Wilson score interval -- correct for small n and for rates near 0/1,
    where the naive normal interval leaves the unit range."""
    if n <= 0:
        return None
    if n < MIN_SAMPLES_FOR_INTERVAL:
        return None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(max(0.0, centre - half), min(1.0, centre + half), "wilson")


def bootstrap_interval(values: Sequence[float], stat,
                       n_boot: int = N_BOOTSTRAP,
                       seed: int = BOOTSTRAP_SEED) -> Optional[Interval]:
    """Percentile bootstrap over the per-sample values."""
    vals = [v for v in values if v is not None]
    if len(vals) < MIN_SAMPLES_FOR_INTERVAL:
        return None
    rng = random.Random(seed)
    stats = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        s = stat(sample)
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            stats.append(s)
    if len(stats) < 2:
        return None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return Interval(lo, hi, "bootstrap")


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return 2.0 * tp / denom


# --------------------------------------------------------------------------- #
# Per-clip prediction / ground-truth pairing
# --------------------------------------------------------------------------- #

@dataclass
class ClipPrediction:
    """What the system produced for one clip, in a form the metrics can score."""

    video_id: str
    predicted_bowler_track_id: Optional[int] = None
    bowler_confirmed: bool = False
    identity_switch_count: int = 0
    n_tracks: int = 0
    predicted_deliveries: List[Tuple[int, int]] = field(default_factory=list)
    predicted_release_frames: Dict[str, Optional[int]] = field(default_factory=dict)
    predicted_phases: Dict[Tuple[int, int], str] = field(default_factory=dict)
    # The track a tracker assigned to the human-identified bowler, for
    # continuity scoring. Filled by the harness, not by the system under test.
    observed_bowler_frames: int = 0
    stage_times: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClipGroundTruth:
    """What a human annotated for the same clip. Built only from
    ``data/cricket_understanding/annotations/``."""

    video_id: str
    annotator_id: str = ""
    bowler_track_id: Optional[int] = None
    bowler_frames: List[int] = field(default_factory=list)
    deliveries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    phases: Dict[Tuple[int, int], str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

_NO_GT = ("no human cricket annotation exists for this clip, so the metric "
          "cannot be computed; it is not zero")


def bowler_metrics(preds: Sequence[ClipPrediction],
                   gts: Sequence[ClipGroundTruth]) -> Dict[str, Metric]:
    """Metrics 1-7 of the Phase 10 brief."""
    gt_by, rejected = _human_gts(gts)
    scored = [p for p in preds if p.video_id in gt_by
              and gt_by[p.video_id].bowler_track_id is not None]
    n_all = len(preds)
    #: Every annotator whose clips were scored, so a reader can verify provenance
    #: without re-reading the annotation tree.
    annotators = sorted({gt_by[p.video_id].annotator_id
                         for p in scored}) if scored else []

    if not scored:
        if rejected:
            why = ("ground truth exists but is not attributable to a human "
                   f"(rejected: {', '.join(sorted(rejected)[:5])}); model-derived "
                   "labels are predictions, not truth, so the metric is "
                   "NOT MEASURED rather than computed against them")
        elif not gt_by:
            why = _NO_GT
        else:
            why = ("no annotated clip has a human-identified bowler track, so "
                   "bowler identity cannot be scored")
        return {k: Metric(k, status=NOT_MEASURED, reason=why, n=n_all)
                for k in ("bowler_top1_accuracy", "bowler_precision",
                          "bowler_recall", "bowler_f1", "false_selection_rate",
                          "identity_switches", "track_continuity",
                          "bowler_abstention_rate")}

    tp = fp = fn = 0
    correct = 0
    abstentions = 0
    for p in scored:
        g = gt_by[p.video_id]
        truth = g.bowler_track_id
        pred = p.predicted_bowler_track_id
        if pred is None:
            abstentions += 1
            fn += 1
            continue
        if pred == truth:
            tp += 1
            correct += 1
        else:
            fp += 1
            fn += 1
    n_scored = len(scored)

    switches = [float(p.identity_switch_count) for p in scored]
    switch_mean = sum(switches) / len(switches) if switches else 0.0
    switch_clips = sum(1 for s in switches if s > 0)

    cont_pairs = [(p.observed_bowler_frames, len(gt_by[p.video_id].bowler_frames))
                  for p in scored if gt_by[p.video_id].bowler_frames]
    cont_values = [o / n for o, n in cont_pairs if n > 0]

    acc = correct / n_scored
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = _f1(tp, fp, fn)
    false_rate = fp / n_scored if n_scored else None
    abstention = abstentions / n_scored if n_scored else None

    cont_mean = (sum(cont_values) / len(cont_values)) if cont_values else None

    return {
        "bowler_top1_accuracy": Metric(
            "bowler_top1_accuracy", acc, MEASURED, n=n_scored,
            interval=wilson_interval(correct, n_scored), unit="fraction",
            higher_is_better=True,
            extra={"correct": correct, "scored_clips": n_scored,
                   "unscored_clips": n_all - n_scored,
                   "annotators": annotators,
                   "rejected_ground_truth": rejected}),
        "bowler_precision": Metric(
            "bowler_precision", prec, MEASURED, n=tp + fp,
            interval=wilson_interval(tp, tp + fp) if (tp + fp) else None,
            unit="fraction", higher_is_better=True,
            extra={"tp": tp, "fp": fp}),
        "bowler_recall": Metric(
            "bowler_recall", rec, MEASURED, n=n_scored,
            interval=wilson_interval(tp, n_scored), unit="fraction",
            higher_is_better=True,
            extra={"tp": tp, "fn": fn}),
        "bowler_f1": Metric(
            "bowler_f1", f1, MEASURED, n=n_scored, unit="fraction",
            higher_is_better=True,
            extra={"tp": tp, "fp": fp, "fn": fn,
                   "note": "pooled counts, not the mean of per-clip F1"}),
        "false_selection_rate": Metric(
            "false_selection_rate", false_rate, MEASURED, n=n_scored,
            interval=wilson_interval(fp, n_scored), unit="fraction",
            higher_is_better=False,
            extra={"wrong_person_clips": fp,
                   "meaning": "a wrong bowler means every downstream "
                              "biomechanical number describes the wrong human"}),
        "bowler_abstention_rate": Metric(
            "bowler_abstention_rate", abstention, MEASURED, n=n_scored,
            interval=wilson_interval(abstentions, n_scored), unit="fraction",
            higher_is_better=None,
            extra={"abstained_clips": abstentions}),
        "identity_switches": Metric(
            "identity_switches", switch_mean, MEASURED, n=n_scored,
            unit="switches/clip", higher_is_better=False,
            interval=bootstrap_interval(switches, lambda s: sum(s) / len(s)),
            extra={"clips_with_at_least_one_switch": switch_clips,
                   "total_switches": int(sum(switches))}),
        "track_continuity": Metric(
            "track_continuity", cont_mean,
            MEASURED if cont_mean is not None else NOT_MEASURED,
            reason="" if cont_mean is not None else
            "no annotated bowler track with a frame span",
            n=len(cont_values), unit="fraction", higher_is_better=True,
            interval=bootstrap_interval(cont_values, lambda s: sum(s) / len(s)),
            extra={"definition": "observed_bowler_frames / human-annotated span"}),
    }


def _temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    union = (max(a[1], b[1]) - min(a[0], b[0]) + 1)
    return inter / union if union > 0 else 0.0


def _match_deliveries(pred: ClipPrediction, gt: ClipGroundTruth
                      ) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """Greedy best-IoU one-to-one matching. Deterministic: ties break on id."""
    cands = []
    for did, d in sorted(gt.deliveries.items()):
        gwin = (int(d["start_frame"]), int(d["end_frame"]))
        for i, pwin in enumerate(pred.predicted_deliveries):
            iou = _temporal_iou(gwin, pwin)
            if iou >= DELIVERY_MATCH_IOU:
                cands.append((iou, did, i, gwin, pwin))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_g, used_p, out = set(), set(), []
    for iou, did, i, gwin, pwin in cands:
        if did in used_g or i in used_p:
            continue
        used_g.add(did)
        used_p.add(i)
        out.append((did, gwin, pwin))
    return out


def delivery_metrics(preds: Sequence[ClipPrediction],
                     gts: Sequence[ClipGroundTruth]) -> Dict[str, Metric]:
    """Metrics 8-9 of the Phase 10 brief."""
    gt_by, rejected = _human_gts(gts)
    total_human, matched = 0, 0
    per_clip_hits: List[float] = []
    release_err: List[float] = []
    for p in preds:
        g = gt_by.get(p.video_id)
        if g is None or not g.deliveries:
            continue
        m = _match_deliveries(p, g)
        total_human += len(g.deliveries)
        matched += len(m)
        per_clip_hits.append(len(m) / len(g.deliveries))
        for did, _gwin, _pwin in m:
            truth_r = g.deliveries[did].get("release_frame")
            pred_r = p.predicted_release_frames.get(did)
            if truth_r is not None and pred_r is not None:
                release_err.append(abs(float(pred_r) - float(truth_r)))

    if not total_human:
        why = (("ground truth exists but is not attributable to a human "
                f"(rejected: {', '.join(sorted(rejected)[:5])})")
               if rejected else
               "no human delivery annotation exists, so delivery detection "
               "cannot be scored; it is not zero")
        return {k: Metric(k, status=NOT_MEASURED, reason=why, n=len(preds))
                for k in ("delivery_detection_accuracy", "release_frame_error_frames")}

    acc = matched / total_human
    mae = (sum(release_err) / len(release_err)) if release_err else None
    return {
        "delivery_detection_accuracy": Metric(
            "delivery_detection_accuracy", acc, MEASURED, n=total_human,
            interval=bootstrap_interval(per_clip_hits, lambda s: sum(s) / len(s)),
            unit="fraction", higher_is_better=True,
            extra={"matched": matched, "human_deliveries": total_human,
                   "match_rule": f"temporal IoU >= {DELIVERY_MATCH_IOU}, one-to-one"}),
        "release_frame_error_frames": Metric(
            "release_frame_error_frames", mae,
            MEASURED if mae is not None else NOT_MEASURED,
            reason="" if mae is not None else
            "deliveries were matched but no predicted release frame was produced",
            n=len(release_err), unit="frames", higher_is_better=False,
            interval=bootstrap_interval(release_err, lambda s: sum(s) / len(s)),
            extra={"n_matched_with_both_release_frames": len(release_err)}),
    }


def phase_metrics(preds: Sequence[ClipPrediction],
                  gts: Sequence[ClipGroundTruth]) -> Dict[str, Metric]:
    """Metric 10 of the Phase 10 brief: frame-level phase agreement."""
    gt_by, rejected = _human_gts(gts)
    correct = total = 0
    confusion: Dict[str, Dict[str, int]] = {}
    for p in preds:
        g = gt_by.get(p.video_id)
        if g is None or not g.phases:
            continue
        for key, truth in g.phases.items():
            pred = p.predicted_phases.get(key)
            total += 1
            confusion.setdefault(truth, {}).setdefault(pred or "NO_PREDICTION", 0)
            confusion[truth][pred or "NO_PREDICTION"] += 1
            if pred == truth:
                correct += 1
    if not total:
        why = (("ground truth exists but is not attributable to a human "
                f"(rejected: {', '.join(sorted(rejected)[:5])})")
               if rejected else
               "no human bowling-phase annotation exists, so phase accuracy "
               "cannot be scored; it is not zero")
        return {k: Metric(k, status=NOT_MEASURED, reason=why, n=len(preds))
                for k in ("phase_accuracy", "phase_macro_f1")}

    per_class_f1 = []
    labels = sorted(confusion)
    for lbl in labels:
        tp = confusion[lbl].get(lbl, 0)
        fp = sum(confusion[o].get(lbl, 0) for o in labels if o != lbl)
        fn = sum(v for k, v in confusion[lbl].items() if k != lbl)
        f = _f1(tp, fp, fn)
        if f is not None:
            per_class_f1.append(f)
    return {
        "phase_accuracy": Metric(
            "phase_accuracy", correct / total, MEASURED, n=total,
            interval=wilson_interval(correct, total), unit="fraction",
            higher_is_better=True, extra={"confusion": confusion}),
        "phase_macro_f1": Metric(
            "phase_macro_f1",
            (sum(per_class_f1) / len(per_class_f1)) if per_class_f1 else None,
            MEASURED, n=len(per_class_f1), unit="fraction", higher_is_better=True,
            extra={"per_class_f1": {l: _f1(confusion[l].get(l, 0),
                                           sum(confusion[o].get(l, 0) for o in labels
                                               if o != l),
                                           sum(v for k, v in confusion[l].items()
                                               if k != l))
                                     for l in labels},
                   "note": "a phase model that always predicts the most common "
                           "phase can score well on accuracy alone"}),
    }


def compute_all(preds: Sequence[ClipPrediction],
                gts: Sequence[ClipGroundTruth]) -> Dict[str, Metric]:
    out: Dict[str, Metric] = {}
    out.update(bowler_metrics(preds, gts))
    out.update(delivery_metrics(preds, gts))
    out.update(phase_metrics(preds, gts))
    return out


# --------------------------------------------------------------------------- #
# Phase 11 -- A/B/C/D comparison
# --------------------------------------------------------------------------- #

ARMS: Dict[str, str] = {
    "A": "Current PaceAI selection logic (src/tracking.py select_bowler_track_with_meta)",
    "B": "A + lower-half spatial prior",
    "C": "B + cricket role model (Model B)",
    "D": "C + temporal bowling-action model (Model C)",
}


def assert_no_test_selection(selected: Dict[str, str],
                             test_ids: Iterable[str]) -> None:
    """Refuse an arm-selection decision that consulted the test split.

    ``selected`` maps arm -> the split its hyper-parameters/architecture were
    chosen on.  Anything other than ``"val"`` is a protocol violation, because
    a test-tuned arm makes the test set a validation set.
    """
    test = set(test_ids)
    for arm, split in selected.items():
        if split in ("test", *test) or "test" in str(split).lower():
            raise ProtocolViolationError(
                f"arm {arm!r} was selected on {split!r}. Model selection must "
                "use the validation split; consulting test invalidates the "
                "held-out estimate.")


class ProtocolViolationError(RuntimeError):
    pass


def compare_arms(arm_metrics: Dict[str, Dict[str, Metric]]) -> Dict[str, Any]:
    """Rank arms by measured metrics only.

    ``arm_metrics`` maps arm -> the result of :func:`compute_all`.  An arm is
    only comparable if the metric is ``MEASURED``; arms with ``NOT MEASURED``
    metrics are reported as not comparable rather than being given a score.  No
    composite score is invented: the brief forbids ranking on arbitrary scores.
    """
    names = sorted({metric.name
                    for arm in arm_metrics.values()
                    for metric in arm.values()})
    table: Dict[str, Any] = {}
    for mname in names:
        row: Dict[str, Any] = {}
        measured_arms = []
        for arm in sorted(arm_metrics):
            m = arm_metrics[arm].get(mname)
            if m is None or not m.measured:
                row[arm] = {"value": None, "status": NOT_MEASURED,
                            "reason": (m.reason if m else "metric not computed")}
                continue
            row[arm] = {"value": m.value, "status": MEASURED, "n": m.n,
                        "interval": m.interval.to_dict() if m.interval else None,
                        "higher_is_better": m.higher_is_better}
            measured_arms.append(arm)
        row["comparable_arms"] = measured_arms
        row["comparable"] = len(measured_arms) >= 2
        if not row["comparable"]:
            row["verdict"] = (
                "NOT COMPARABLE - " +
                ("fewer than two arms have this metric measured" if not measured_arms
                 else "only one arm has this metric measured"))
        else:
            # Two or more arms are measured, so the comparison is reportable.
            # A verdict is still *not* a ranking: it says whether the intervals
            # separate, and refuses to name a winner when they overlap. Declaring
            # a winner on overlapping intervals would be reading noise as signal.
            best = max(measured_arms, key=lambda a: row[a]["value"]
                       if (row[a].get("higher_is_better") is not False)
                       else -row[a]["value"])
            others = [a for a in measured_arms if a != best]
            overlap = [a for a in others
                       if _intervals_overlap(row[a].get("interval"),
                                             row[best].get("interval"))]
            direction = ("highest" if row[best].get("higher_is_better") is not False
                         else "lowest")
            if overlap:
                row["verdict"] = (
                    f"INCONCLUSIVE - {best} has the {direction} point estimate, "
                    f"but its interval overlaps {', '.join(overlap)}; with these "
                    f"clips the arms cannot be told apart")
            else:
                worse = ", ".join(f"{a} [{row[a]['interval']['lo']}, "
                                  f"{row[a]['interval']['hi']}]"
                                  if row[a].get("interval") else f"{a} (no interval)"
                                  for a in others)
                row["verdict"] = (
                    f"SEPARATED - {best} has the {direction} point estimate and "
                    f"its interval does not overlap {worse}. Verify on the test "
                    f"split once, and report the test number as the result")
            row["best_point_estimate"] = best
        table[mname] = row
    return {"arms": ARMS, "metrics": table}


def _intervals_overlap(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    """Do two reported intervals overlap? A missing interval means the sample was
    too small to characterise, which is treated as overlapping -- you cannot claim
    separation from an estimate that has no spread."""
    if not a or not b:
        return True
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
