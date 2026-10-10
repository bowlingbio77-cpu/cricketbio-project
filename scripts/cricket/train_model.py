"""Phase 9 -- task readiness report and the guarded training entry point.

    python scripts/cricket/train_model.py --report
    python scripts/cricket/train_model.py --model B --dry-run
    python scripts/cricket/train_model.py --model B            # trains if, and only if, ready

The defaults are the safe ones.  ``--report`` and ``--dry-run`` never write
weights.  A real run is refused by :func:`tasks.require_ready` unless the task
has (a) human labels, (b) a non-empty test split, and (c) enough training
samples per class.

The declared baselines are deliberately the simplest defensible ones:
``majority_class`` and multinomial logistic regression.  scikit-learn is already
a dependency; nothing heavier is pulled in, and no hyper-parameter is tuned
against the test split -- model selection uses the VAL split only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cricket_understanding import schema, tasks  # noqa: E402

FEATURE_NAMES = {
    "B": None,   # filled from generators.ROLE_FEATURE_NAMES at runtime
    "C": None,   # filled from generators.ACTION_FEATURE_NAMES at runtime
}


def feature_names_for(key: str):
    from src.cricket_understanding import generators
    if key == "B":
        return generators.ROLE_FEATURE_NAMES
    if key == "C":
        return generators.ACTION_FEATURE_NAMES
    return None


def _vector(row, names):
    f = row.get("features") or {}
    return [float(f.get(n, 0.0)) for n in names]


def build_matrices(task_key: str, rows):
    names = feature_names_for(task_key)
    X = [_vector(r, names) for r in rows]
    y = [r["label"] for r in rows]
    return X, y, list(names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(tasks.TASKS), default=None,
                    help="which task to report on or train")
    ap.add_argument("--report", action="store_true",
                    help="print the Phase 9 task/readiness report and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate every guard and exit without training")
    ap.add_argument("--out", default=os.path.join(
        "evaluation", "cricket_task_readiness.json"))
    ap.add_argument("--model-out", default=os.path.join("models", "cricket"))
    args = ap.parse_args(argv)

    if args.report or args.model is None:
        md = tasks.tasks_markdown()
        print(md)
        tasks.write_readiness_report(args.out)
        print(f"Readiness JSON: {args.out}")
        return 0

    task = tasks.TASKS[args.model]
    rows = tasks.read_rows(task)
    readiness = tasks.evaluate_readiness(task, rows)
    print(json.dumps(readiness.to_dict(), indent=2))

    if args.dry_run:
        print(f"\nDRY RUN: {'would train' if readiness.ready else 'REFUSES to train'} "
              f"Model {task.key}.")
        return 0 if readiness.ready else 1

    if task.key == "A":
        print("\nModel A is an Ultralytics detector. Train it with:")
        print(f"  yolo detect train data={schema.DATASET_YAML} "
              f"model={os.path.join('models', 'yolo11n.pt')}")
        print("Ultralytics is deliberately not invoked from here: it would need a "
              "GPU and hours of compute, and the guards above must pass first.")
        return 0 if readiness.ready else 1

    # ---- guarded tabular training (Models B and C) ------------------------
    try:
        tasks.require_ready(readiness)
    except tasks.TaskNotReadyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    names = feature_names_for(task.key)
    by_split = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r.get("split", "unassigned"), []).append(r)

    Xtr, ytr, _ = build_matrices(task.key, by_split["train"])
    Xva, yva = (build_matrices(task.key, by_split["val"])[:2] if by_split["val"] else ([], []))
    Xte, yte = (build_matrices(task.key, by_split["test"])[:2] if by_split["test"] else ([], []))

    counts = {l: ytr.count(l) for l in set(ytr)}
    majority = max(counts, key=counts.get) if counts else None
    print(f"\nmajority class in train: {majority} ({counts.get(majority)}/{len(ytr)})")

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    clf.fit(np.asarray(Xtr), ytr)
    print("fitted multinomial logistic regression")

    results = {"task": task.key, "title": task.title, "majority_class": majority,
               "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
               "model": "multinomial_logistic_regression",
               "n_features": len(names)}
    if yva:
        results["val_accuracy"] = float(accuracy_score(yva, clf.predict(np.asarray(Xva))))
        results["val_f1_macro"] = float(f1_score(yva, clf.predict(np.asarray(Xva)),
                                                  average="macro", zero_division=0))
    if yte:
        pred = clf.predict(np.asarray(Xte))
        results["test_accuracy"] = float(accuracy_score(yte, pred))
        results["test_f1_macro"] = float(f1_score(yte, pred, average="macro",
                                                   zero_division=0))
        results["test_report"] = classification_report(yte, pred, zero_division=0)
    else:
        results["test_accuracy"] = "NOT MEASURED - empty test split"

    os.makedirs(args.model_out, exist_ok=True)
    out = os.path.join(args.model_out, f"cricket_model_{task.key}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "test_report"}, indent=2))
    print(f"\nmetrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
