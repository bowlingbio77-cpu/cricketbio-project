"""
Phase 9 -- the three independent learning tasks, declared once, guarded hard.

This module is the single source of truth for what PaceAI is *allowed* to learn
about cricket, and for when it is allowed to claim anything about it.

Three tasks, deliberately independent
-------------------------------------
======  =====================  ====================================  ==========
task    question               dataset produced by                  human labels
======  =====================  ====================================  ==========
A       who/what is where?     ``det``        (YOLO detect)         roles+objects
B       what role is a track?  ``role``       (per-track features)   roles
C       what phase is this?    ``action``     (pose window)          phases
======  =====================  ====================================  ==========

Why no transformer by default
-----------------------------
The brief requires the *simplest defensible baseline* first.  With a corpus of
0 human-labelled deliveries there is no basis for capacity selection, so the
declared baseline for B and C is ``majority_class`` / ``mean_class`` and then
logistic regression.  A stronger architecture is a *recorded experiment*
(Phase 11), never a default.

The four guards
---------------
Every task must pass all four before a single gradient step is allowed:

1. ``human_labels_present``  -- at least one row whose ``annotator_id`` is a real
   human id.  Model-derived auto-labels do not count, by construction.
2. ``held_out_split_nonempty`` -- ``test`` must contain at least one sample.
   Without a holdout, any "accuracy" is training accuracy.
3. ``no_test_tuning`` -- the training entry point refuses to read ``test`` for
   early stopping, model selection, or threshold choice.
4. ``label_support_per_class`` -- every class to be predicted needs a minimum
   number of training samples, or the task reports which classes are
   unlearnable rather than silently producing a degenerate model.

A task that fails a guard is ``NOT READY``.  This is the normal, expected state
of this project today and is not an error condition to be worked around.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import schema

TASKS_VERSION = "cricket_understanding_tasks_v1"

#: Minimum labelled samples a class needs in TRAIN before it counts as learnable.
MIN_TRAIN_SAMPLES_PER_CLASS = 2

#: Minimum labelled samples in TEST before a held-out metric may be reported.
MIN_TEST_SAMPLES = 1


# --------------------------------------------------------------------------- #
# Declarations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    question: str
    dataset: str                 # generator key under generators.GENERATORS
    label_source: str            # which human annotation kind supplies labels
    labels: Tuple[str, ...]
    #: Human-readable definition of each label. Used verbatim in the docs so the
    #: documentation can never drift from the training code.
    label_definitions: Dict[str, str]
    baseline: str
    input_modality: str
    feature_names: Optional[Tuple[str, ...]] = None
    notes: str = ""

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "question": self.question,
            "dataset": self.dataset, "label_source": self.label_source,
            "labels": list(self.labels), "n_labels": self.n_labels,
            "label_definitions": dict(self.label_definitions),
            "baseline": self.baseline, "input_modality": self.input_modality,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "notes": self.notes,
        }


ROLE_DEFINITIONS = {
    "BOWLER": "The person who bowls the ball at the start of the clip. Exactly one "
              "per delivery, and the only person whose pose/biomechanics are analysed.",
    "STRIKER": "The batter on strike, i.e. facing the ball at this delivery.",
    "NON_STRIKER": "The batter at the non-striker's end, not facing this delivery.",
    "WICKETKEEPER": "The fielder behind the stumps at the striker's end.",
    "FIELDER": "Any other person on the field who is not in the four roles above.",
    "UNKNOWN": "A person is visible but the annotator cannot assign a role with "
               "confidence. This is a real label, not a missing value.",
}

PHASE_DEFINITIONS = {
    "NON_BOWLING": "Before the run-up starts; the person is present but not yet "
                   "beginning a delivery. Default label for annotated frames "
                   "outside any delivery window.",
    "RUN_UP": "First frame of the accelerating approach towards the crease.",
    "APPROACH": "Run-up continuing after the initial acceleration, before the "
                "final gathering stride.",
    "GATHER": "The loading/plant phase immediately before the delivery stride.",
    "DELIVERY_STRIDE": "The final stride in which the bowling arm is taken back "
                       "and over, ending at front-foot contact.",
    "FRONT_FOOT_CONTACT": "The instant the front foot contacts the ground, which "
                          "brackets the delivery stride and the release.",
    "RELEASE": "The instant the ball leaves the hand. A single frame.",
    "FOLLOW_THROUGH": "After release, the bowler's momentum carries them forward; "
                      "ends at the annotated follow-through end frame.",
    "UNKNOWN": "The annotator saw a person in a frame but could not determine "
               "the bowling phase. Never silently replaced by a model guess.",
}

OBJECT_DEFINITIONS = {
    "BALL": "The cricket ball at any point of visibility.",
    "STUMPS": "The set of stumps (all three bails included) at either end.",
}


TASK_A = TaskSpec(
    key="A",
    title="Cricket object detection",
    question="Which cricket-specific objects are present in this frame, and where?",
    dataset="det",
    label_source="annotations/roles/*.json (persons) + annotations/objects/*.json "
                 "(BALL, STUMPS)",
    labels=("BOWLER", "STRIKER", "NON_STRIKER", "WICKETKEEPER", "FIELDER",
            "UNKNOWN", "BALL", "STUMPS"),
    label_definitions={**ROLE_DEFINITIONS, **OBJECT_DEFINITIONS},
    baseline="yolo11n, stock COCO weights, transferred. Ultralytics already "
             "tracks this repository; no new framework is introduced.",
    input_modality="RGB frame",
    notes="Indices 0-5 are the six person roles, 10-11 are BALL/STUMPS. 6-9 are "
          "reserved so a future scene object does not renumber the dataset. "
          "The existing models/bowler_batsman_yolo11n.pt is NOT reused: it was "
          "trained on model-derived auto-labels and is dead weight at runtime.",
)

TASK_B = TaskSpec(
    key="B",
    title="Cricket role recognition",
    question="Given a tracked person and its temporal/spatial context, what is "
             "this person's cricket role?",
    dataset="role",
    label_source="annotations/roles/*.json",
    labels=tuple(schema.PLAYER_ROLES.values()),
    label_definitions=dict(ROLE_DEFINITIONS),
    baseline="majority_class, then multinomial logistic regression on the "
             "38 frozen geometry/temporal features. No neural network until a "
             "held-out result justifies one.",
    input_modality="one row per (video, track): box geometry + motion over the "
                   "whole track",
    notes="Features are computed from HUMAN boxes only. No PaceAI score is an "
          "input, so the model cannot learn to imitate the heuristic it is "
          "meant to replace.",
)

TASK_C = TaskSpec(
    key="C",
    title="Bowling action / phase model",
    question="Given a temporal pose sequence of the locked human bowler, which "
             "bowling phase is each frame in?",
    dataset="action",
    label_source="annotations/bowling_phases/*.json, anchored to the delivery's "
                 "human-identified bowler track",
    labels=tuple(schema.BOWLING_PHASES),
    label_definitions=dict(PHASE_DEFINITIONS),
    baseline="prior over phase frequencies, then multinomial logistic regression "
             "on hip-centred normalised keypoints + first/second differences, "
             "then a linear-chain CRF. A temporal transformer is explicitly "
             "out of scope until a holdout exists.",
    input_modality="pose window per delivery: normalised keypoints, velocity, "
                   "acceleration, per-frame phase label",
    notes="Requires Phase 8 pose sequences. Frames with no pose are emitted with "
          "pose_missing=1 and zeroed features, never interpolated. "
          "Phase ordering (RUN_UP <= ... <= FOLLOW_THROUGH) is a hard constraint "
          "the CRF enforces and the QC module verifies on the human labels.",
)

TASKS: Dict[str, TaskSpec] = {t.key: t for t in (TASK_A, TASK_B, TASK_C)}


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #

@dataclass
class GuardResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class TaskReadiness:
    task: TaskSpec
    guards: List[GuardResult] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    #: Train rows attributable to a human. The trainer uses only these; ``n_train``
    #: is the raw row count, kept so a report can show how many rows were rejected.
    n_train_human: int = 0
    n_model_derived: int = 0
    per_class_train: Dict[str, int] = field(default_factory=dict)
    unlearnable_classes: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(g.passed for g in self.guards)

    @property
    def status(self) -> str:
        if self.ready:
            return "READY"
        if self.n_train == 0:
            return "NOT READY - no human-labelled samples"
        return "NOT READY - guards failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.key,
            "title": self.task.title,
            "status": self.status,
            "ready": self.ready,
            "n_train": self.n_train, "n_val": self.n_val, "n_test": self.n_test,
            "n_train_human": self.n_train_human,
            "n_model_derived_rows_rejected": self.n_model_derived,
            "per_class_train": dict(sorted(self.per_class_train.items())),
            "unlearnable_classes": list(self.unlearnable_classes),
            "guards": [g.to_dict() for g in self.guards],
        }


class TaskNotReadyError(RuntimeError):
    """Raised when a training run is attempted before the task is trainable."""


def evaluate_readiness(task: TaskSpec, rows: Sequence[Dict[str, Any]]) -> TaskReadiness:
    """Decide, from data alone, whether ``task`` may be trained *and evaluated*.

    ``rows`` are generated-dataset samples: each needs at least ``split``,
    ``label`` and ``annotator_id``.
    """
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r.get("split", "unassigned"), []).append(r)

    # Guard 1 -- human labels. An empty annotator_id means the row did not come
    # from a human decision, whatever the file layout says -- and a non-empty id
    # that names an automatic labeller is rejected too, because a row labelled by
    # a model is a prediction wearing a provenance field. The same
    # is_human_annotator() check the metrics use guards this one, so a dataset
    # that can be trained on is exactly a dataset that can be scored against.
    from .metrics import is_human_annotator

    machine = [r for r in rows
               if str(r.get("annotator_id") or "").strip()
               and not is_human_annotator(r.get("annotator_id"))]
    # Only human train rows may be learned from *and* counted as class support.
    # Counting a model-derived row toward class support would let a task report
    # itself ready on data it is not allowed to train on.
    human_train = [r for r in by_split["train"]
                   if is_human_annotator(r.get("annotator_id"))]
    human_any = [r for r in rows if is_human_annotator(r.get("annotator_id"))]
    n_train = len(by_split["train"])
    n_val = len(by_split["val"])
    n_test = len(by_split["test"])

    per_class: Dict[str, int] = {}
    for r in human_train:
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1
    unlearnable = sorted(l for l in task.labels
                         if per_class.get(l, 0) < MIN_TRAIN_SAMPLES_PER_CLASS)

    if not human_train:
        human_detail = (
            f"{len(machine)} row(s) carry an annotator_id that names an automatic "
            f"labeller rather than a person, and {len(rows) - len(machine)} carry "
            f"none. There is no cricket ground truth in this repository, so "
            f"nothing can be trained or scored."
            if machine else
            "0 train rows are attributable to a human. There is no cricket ground "
            "truth in this repository, so nothing can be trained or scored.")
    else:
        human_detail = (
            f"{len(human_train)} of {n_train} train row(s) are attributable to a "
            f"human ({len(human_any)} row(s) overall). {len(machine)} "
            f"model-derived row(s) were rejected and are not counted.")

    guards = [
        GuardResult(
            "human_labels_present", bool(human_train), human_detail),
        GuardResult(
            "held_out_split_nonempty", n_test >= MIN_TEST_SAMPLES,
            f"test split holds {n_test} sample(s); the minimum is "
            f"{MIN_TEST_SAMPLES}."
            if n_test else
            "The test split is EMPTY. Any accuracy computed on train or val is "
            "in-sample and must never be reported as generalisation."),
        GuardResult(
            "label_support_per_class", not unlearnable,
            f"{len(task.labels) - len(unlearnable)} of {len(task.labels)} class(es) "
            f"have >= {MIN_TRAIN_SAMPLES_PER_CLASS} training sample(s)."
            if not unlearnable else
            f"{len(unlearnable)} class(es) have fewer than "
            f"{MIN_TRAIN_SAMPLES_PER_CLASS} training sample(s): "
            f"{', '.join(unlearnable)}. They cannot be learned; reporting a "
            f"confusion matrix over them would be misleading."),
    ]
    return TaskReadiness(task=task, guards=guards, n_train=n_train, n_val=n_val,
                         n_test=n_test, per_class_train=per_class,
                         unlearnable_classes=unlearnable,
                         n_train_human=len(human_train),
                         n_model_derived=len(machine))


def require_ready(readiness: TaskReadiness) -> None:
    if readiness.ready:
        return
    failed = [g for g in readiness.guards if not g.passed]
    raise TaskNotReadyError(
        f"Task {readiness.task.key} ({readiness.task.title}) is "
        f"{readiness.status}. Refusing to train.\n" +
        "\n".join(f"  - {g.name}: {g.detail}" for g in failed))


# --------------------------------------------------------------------------- #
# Reading generated datasets back
# --------------------------------------------------------------------------- #

def read_role_rows(extracted_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(extracted_dir, "role", "role_dataset.jsonl")
    return _read_jsonl(path)


def read_action_rows(extracted_dir: str) -> List[Dict[str, Any]]:
    """One row per (delivery, frame) with a human phase label."""
    rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        d = os.path.join(extracted_dir, "action", split)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                payload = json.load(fh)
            for w in payload.get("windows", []):
                if not w.get("phase"):
                    continue
                rows.append({
                    "sample_id": f"{payload['video_id']}:{payload['delivery_id']}:"
                                 f"{w['frame_id']}",
                    "split": payload.get("split", split),
                    "label": w["phase"],
                    "annotator_id": payload.get("annotator_id", ""),
                    "features": w.get("features", {}),
                    "pose_missing": bool(w.get("pose_missing")),
                })
    return rows


def read_det_rows(extracted_dir: str) -> List[Dict[str, Any]]:
    """One row per annotated box, derived from the det manifest (no pixels read)."""
    path = os.path.join(extracted_dir, "det", "manifest.jsonl")
    out = []
    for r in _read_jsonl(path):
        if str(r.get("transformation") or "").startswith("rejected:"):
            continue
        out.append(r)
    return out


_READERS = {"role": read_role_rows, "action": read_action_rows, "det": read_det_rows}


def read_rows(task: TaskSpec, extracted_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    extracted_dir = extracted_dir or schema.p("extracted")
    return _READERS[task.dataset](extracted_dir)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def readiness_report(extracted_dir: Optional[str] = None) -> Dict[str, Any]:
    """Full, honest Phase 9 status for all three tasks."""
    per_task = {}
    for key, task in TASKS.items():
        rows = read_rows(task, extracted_dir)
        if task.key == "A":
            # Model A labels are carried on the det manifest rows.
            rows = [{"split": r.get("split", "unassigned"), "label": r.get("label", ""),
                     "annotator_id": r.get("annotator_id", "")} for r in rows]
        per_task[key] = evaluate_readiness(task, rows).to_dict()
    return {
        "tasks_version": TASKS_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "min_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
        "min_test_samples": MIN_TEST_SAMPLES,
        "tasks": per_task,
        "n_ready": sum(1 for v in per_task.values() if v["ready"]),
        "n_total": len(per_task),
    }


def write_readiness_report(path: str, extracted_dir: Optional[str] = None) -> str:
    payload = readiness_report(extracted_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema.dumps(payload))
    return path


def tasks_markdown(extracted_dir: Optional[str] = None) -> str:
    """Human-readable Phase 9 section, generated so docs cannot drift from code."""
    rep = readiness_report(extracted_dir)
    L: List[str] = []
    A = L.append
    A("# Phase 9 -- Learning task definitions")
    A("")
    A(f"Generated by `scripts/cricket/train_model.py --report`. "
      f"Task spec version `{TASKS_VERSION}`.")
    A("")
    for key in ("A", "B", "C"):
        t = TASKS[key]
        r = rep["tasks"][key]
        A(f"## Model {t.key} -- {t.title}")
        A("")
        A(f"**Question.** {t.question}")
        A("")
        A(f"- **Human label source:** `{t.label_source}`")
        A(f"- **Input:** {t.input_modality}")
        A(f"- **Baseline:** {t.baseline}")
        if t.notes:
            A(f"- **Notes:** {t.notes}")
        A("")
        A(f"**Labels ({t.n_labels})**")
        A("")
        A("| Label | Definition |")
        A("|---|---|")
        for lbl in t.labels:
            A(f"| `{lbl}` | {t.label_definitions.get(lbl, '')} |")
        A("")
        A(f"**Status: {r['status']}** "
          f"(train {r['n_train']} / val {r['n_val']} / test {r['n_test']})")
        A("")
        A("| Guard | Result | Detail |")
        A("|---|---|---|")
        for g in r["guards"]:
            A(f"| `{g['name']}` | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |")
        A("")
    A("## Phase 9 conclusion")
    A("")
    n_ready = rep["n_ready"]
    if n_ready == 0:
        A(f"**{n_ready} of {rep['n_total']} tasks are trainable.** No cricket "
          "model can be trained or honestly evaluated, because the repository "
          "contains no human-labelled cricket data. This is the expected and "
          "correct state: the task definitions, the feature contracts and the "
          "guards are in place so the first real annotation can be trained "
          "immediately, and so no run can quietly skip the holdout.")
    else:
        A(f"**{n_ready} of {rep['n_total']} tasks are trainable.**")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    print(tasks_markdown())
