"""
Phase 5 -- leakage-safe train/val/test splitting.

Hard rules enforced here
------------------------
1. The split is at **VIDEO** level.  A video_id lands in exactly one of
   train/val/test.  Frames from one video never straddle a split.
2. When a bowler id is known, the split is at **BOWLER** level (grouped), so
   the same bowler cannot appear in train and test.  This is the stricter rule
   and is used whenever any record carries ``bowler_id_if_known``.
3. When no bowler ids exist, groups fall back to the camera-view prefix, which
   in this corpus is a single session/operator.  That is *conservative*: it can
   only merge more aggressively, never leak less.
4. The RNG is seeded (``RANDOM_STATE = 42``), so the split is reproducible.
5. If the corpus is too small to hit 70/15/15, the **actual** allocation is
   returned and recorded.  Ratios are never faked.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import qc, schema
from .registry import VideoRecord

RANDOM_STATE = 42
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}

#: Below this many videos a ratio split is meaningless; we fall back to
#: "1 video per non-train split" and document the actual numbers.
MIN_VIDEOS_FOR_RATIO_SPLIT = 10

SPLIT_FILES = {
    "train": schema.p("splits", "train.txt"),
    "val": schema.p("splits", "val.txt"),
    "test": schema.p("splits", "test.txt"),
}


@dataclass
class SplitResult:
    splits: Dict[str, List[str]]
    groups: Dict[str, str]                 # video_id -> group key
    n_videos: int
    n_groups: int
    achieved: Dict[str, float]             # actual fraction per split
    target: Dict[str, float] = field(default_factory=lambda: dict(TARGET))
    ratio_split_applied: bool = True
    strategy: str = ""
    note: str = ""
    grouping_degraded: bool = False
    residual_leakage_risk: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "splits": {k: sorted(v) for k, v in self.splits.items()},
            "n_videos": self.n_videos,
            "n_groups": self.n_groups,
            "achieved_fractions": {k: round(v, 4) for k, v in self.achieved.items()},
            "target_fractions": self.target,
            "ratio_split_applied": self.ratio_split_applied,
            "strategy": self.strategy,
            "note": self.note,
            "grouping_degraded": self.grouping_degraded,
            "residual_leakage_risk": self.residual_leakage_risk,
            "group_assignment": dict(sorted(self.groups.items())),
        }


def _allocate(n: int, target: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder allocation that never produces a negative bucket."""
    raw = {k: n * v for k, v in target.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:max(0, remainder)]:
        base[k] += 1
    return base


def make_splits(records: List[VideoRecord],
                target: Optional[Dict[str, float]] = None,
                seed: int = RANDOM_STATE,
                group_by_bowler: bool = True) -> SplitResult:
    """Build leakage-safe splits from registered video records."""
    target = dict(target or TARGET)
    usable = [r for r in records if r and r.video_id]
    n = len(usable)

    if n == 0:
        return SplitResult(
            splits={"train": [], "val": [], "test": []},
            groups={}, n_videos=0, n_groups=0,
            achieved={k: 0.0 for k in target},
            ratio_split_applied=False, strategy="empty",
            note="No videos registered. Split is empty by construction, not by "
                 "accident; annotate videos before splitting.")

    # --- group assignment ---------------------------------------------------
    groups: Dict[str, str] = {}
    for r in usable:
        if group_by_bowler and r.bowler_id_if_known:
            groups[r.video_id] = f"bowler:{r.bowler_id_if_known}"
        else:
            groups[r.video_id] = r.group_key

    by_group: Dict[str, List[str]] = defaultdict(list)
    for vid, g in groups.items():
        by_group[g].append(vid)
    for g in by_group:
        by_group[g].sort()

    group_ids = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    # --- degenerate grouping: document it, do not hide it -------------------
    # If every clip collapses into a single group (e.g. the whole registration
    # came from one camera-view folder, so the conservative group key is the
    # same for all of them), a group-level split is impossible: it would put
    # 100% of the data in train and leave val/test empty. Rather than silently
    # reporting an empty test set, we fall back to a VIDEO-level split and
    # record the residual risk explicitly in the manifest.
    #
    # EXCEPTION -- known bowler.  If the single collapsed group is a *bowler*
    # group (``bowler:<id>``), a video-level fallback would put the SAME HUMAN
    # in train and test. That is the single worst failure this module exists to
    # prevent, so the fallback is REFUSED: everything stays in train, val/test
    # are empty, and the manifest says why. An empty holdout is a known gap; a
    # leaked holdout is a fabricated result.
    bowler_groups = {g for g in group_ids if g.startswith("bowler:")}
    degraded = len(group_ids) < 2
    refuse_bowler_leak = degraded and bool(bowler_groups)
    units = group_ids
    unit_of = dict(groups)
    # ``units_are_groups`` decides how a unit expands back into video ids. It
    # stays True for the refuse path so the train list contains VIDEO IDS, not
    # group keys.
    units_are_groups = True
    if degraded and not refuse_bowler_leak:
        units = sorted(groups)          # one unit per video
        unit_of = {vid: vid for vid in sorted(groups)}
        units_are_groups = False

    # --- ratio split on UNITS (so a whole bowler/video moves together) -------
    n_units = len(units)
    if refuse_bowler_leak:
        counts = {"train": n_units, "val": 0, "test": 0}
        strategy = "refuse_bowler_leak_all_train"
        ratio_applied = False
        note = (
            f"Every clip belongs to the single known bowler group "
            f"{sorted(bowler_groups)[0]!r}. A video-level fallback split would "
            "place the SAME bowler in train and test, so the fallback was "
            "REFUSED. All videos are in train; val and test are EMPTY BY "
            "DESIGN, not by accident. No held-out accuracy can be measured "
            "until clips from at least one additional bowler are registered "
            "with --bowler-id."
        )
    elif n >= MIN_VIDEOS_FOR_RATIO_SPLIT and n_units >= 3:
        counts = _allocate(n_units, target)
        strategy = "grouped_ratio"
        note = ""
        ratio_applied = True
    elif degraded:
        counts = _allocate(n_units, target) if n_units >= 3 else {
            "train": n_units, "val": 0, "test": 0}
        strategy = "video_level_fallback_single_group"
        ratio_applied = n_units >= 3
        note = (
            f"All {n} video(s) fell into a SINGLE group ({group_ids[0] if group_ids else '-'}), "
            "so a leakage-safe group split is impossible. Fell back to a "
            "VIDEO-level split. This is recorded as grouping_degraded=true and "
            "the residual leakage risk is reported."
        )
    else:
        # Too small for 70/15/15 to mean anything. Put everything in train
        # except one whole group in val and one in test (when they exist), and
        # say so explicitly rather than pretending the ratio was hit.
        n_test = 1 if n_units >= 3 else 0
        n_val = 1 if n_units >= 2 else 0
        counts = {"train": n_units - n_val - n_test, "val": n_val, "test": n_test}
        strategy = "fallback_one_group_per_holdout"
        ratio_applied = False
        note = (
            f"Only {n} video(s) in {n_units} group(s) "
            f"(< MIN_VIDEOS_FOR_RATIO_SPLIT={MIN_VIDEOS_FOR_RATIO_SPLIT}). "
            "70/15/15 was NOT achieved. Actual allocation is reported in "
            "'achieved_fractions'. Add videos before drawing accuracy "
            "conclusions."
        )

    risk = ""
    if refuse_bowler_leak:
        risk = (
            "NO HELD-OUT SET EXISTS. The registry contains clips of a single "
            "known bowler, so train/val/test cannot be separated without "
            "leaking that bowler across splits. val and test are empty on "
            "purpose. Any 'accuracy' quoted from this split would be "
            "TRAIN-SET ACCURACY and must not be reported as generalisation. "
            "Fix by registering clips from at least 2 more bowlers with "
            "--bowler-id."
        )
    elif degraded:
        risk = (
            "RESIDUAL LEAKAGE RISK: consecutive clips from the same session, and "
            "very likely the same bowler, can now appear in different splits. "
            "Any accuracy measured on this split is OPTIMISTIC and must be "
            "reported as such. Fix by registering more sessions/bowlers with "
            "--bowler-id so grouping has >1 group."
        )

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    idx = 0
    for name in ("train", "val", "test"):
        for _ in range(counts.get(name, 0)):
            u = units[idx]
            idx += 1
            splits[name].extend(by_group.get(u, [u]) if units_are_groups else [u])
    for k in splits:
        splits[k].sort()

    total = sum(len(v) for v in splits.values()) or 1
    achieved = {k: len(v) / total for k, v in splits.items()}

    return SplitResult(
        splits=splits, groups=groups, n_videos=n, n_groups=len(group_ids),
        achieved=achieved, ratio_split_applied=ratio_applied,
        strategy=strategy, note=note, grouping_degraded=degraded,
        residual_leakage_risk=risk,
    )


def verify(result: SplitResult) -> List[qc.QCIssue]:
    """Re-run the leakage check on the produced split."""
    return qc.check_split_leakage(result.splits)


# --------------------------------------------------------------------------- #

def save_splits(result: SplitResult) -> Dict[str, str]:
    os.makedirs(schema.SPLITS_DIR, exist_ok=True)
    written = {}
    for name, path in SPLIT_FILES.items():
        ids = sorted(result.splits.get(name, []))
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for vid in ids:
                fh.write(vid + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        written[name] = path
    with open(os.path.join(schema.SPLITS_DIR, "split_meta.json"),
              "w", encoding="utf-8") as fh:
        fh.write(schema.dumps(result.to_dict()))
    return written


def load_splits() -> Optional[Dict[str, List[str]]]:
    if not all(os.path.exists(p) for p in SPLIT_FILES.values()):
        return None
    out: Dict[str, List[str]] = {}
    for name, path in SPLIT_FILES.items():
        with open(path, encoding="utf-8") as fh:
            out[name] = [ln.strip() for ln in fh if ln.strip()]
    return out


def split_of(vid: str, splits: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    splits = splits or load_splits() or {}
    for name, ids in splits.items():
        if vid in ids:
            return name
    return None
