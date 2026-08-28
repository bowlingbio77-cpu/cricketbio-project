"""
Train injury-risk models on real labeled datasets and export bundled artifacts.

Two datasets:
  --dataset sports   Kaggle "Multimodal Sports Injury Dataset" (15,420 per-session
                     rows, 156 athletes, 3-class target). Evaluation uses
                     StratifiedGroupKFold grouped by athlete_id so repeated
                     sessions of one athlete never leak across folds. Temporal
                     lag/rolling features are added, and cnn_lstm/transformer
                     are trained on per-athlete session sequences predicting the
                     NEXT session's risk class.
  --dataset cricket  Cricket Injury Dataset (1,272 player-season rows, binary
                     injury_status). Optionally train the ordinal severity target
                     (0 none / 1 minor / 2 major) with --target severity.

Usage:
  python train_sports_injury_model.py --dataset sports --model all
  python train_sports_injury_model.py --dataset sports --model random_forest
  python train_sports_injury_model.py --dataset cricket --model all
  python train_sports_injury_model.py --dataset cricket --target severity --model catboost
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from src import config, ml_models
from src.sports_injury_data import (load_sports_injury, preprocess_sports_injury,
                                    build_sequences, dataset_summary as sports_summary)
from src.cricket_injury_data import (load_cricket_injury, preprocess_cricket_injury,
                                     dataset_summary as cricket_summary)

FLAT_MODELS = ["random_forest", "xgboost", "catboost"]
SEQUENCE_MODELS = ["cnn_lstm", "transformer"]
ALL_MODELS = FLAT_MODELS + SEQUENCE_MODELS

SPORTS_LABEL_MAP = config.SPORTS_INJURY_LABEL_MAP
CRICKET_LABEL_MAP = {0: "no_injury", 1: "injury"}
CRICKET_SEVERITY_LABEL_MAP = config.CRICKET_INJURY_SEVERITY_LABEL_MAP


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_report(bundle, title: str = ""):
    cv = getattr(bundle, "cv_metrics", None) or {}
    base = getattr(bundle, "baseline_metrics", None) or {}
    tag = f" [{title}]" if title else ""
    print(f"  → {bundle.model_name}{tag}  ({cv.get('n_samples', 0)} samples, {cv.get('folds', 0)} folds)")
    print(f"    Accuracy : {cv.get('accuracy_mean', 0.0):.3f} ± {cv.get('accuracy_std', 0.0):.3f}"
          f"   (majority baseline {base.get('accuracy', 0.0):.3f}, "
          f"f1-macro baseline {base.get('f1_macro', 0.0):.3f})")
    print(f"    F1-macro : {cv.get('f1_mean', 0.0):.3f} ± {cv.get('f1_std', 0.0):.3f}   "
          f"Precision {cv.get('precision_mean', 0.0):.3f}   Recall {cv.get('recall_mean', 0.0):.3f}   "
          f"ROC-AUC {cv.get('roc_auc_mean', 0.0):.3f}")
    per_class = cv.get("per_class_report", {})
    for cls in sorted(bundle.label_map):
        key = str(cls)
        pc = per_class.get(key)
        if pc:
            print(f"      class {cls} {bundle.label_map[cls]:<18}: "
                  f"precision {pc.get('precision', 0):.3f}  recall {pc.get('recall', 0):.3f}  "
                  f"f1 {pc.get('f1-score', 0):.3f}  n={int(pc.get('support', 0))}")


def print_importances(bundle, top_n: int = 12):
    m = bundle.model
    try:
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        elif hasattr(m, "get_feature_importance"):
            imp = np.asarray(m.get_feature_importance(), dtype=float)
        else:
            return
        if imp.ndim != 1 or len(imp) != len(bundle.feature_names):
            return
        order = np.argsort(imp)[::-1][:top_n]
        print("    Top features:")
        for i in order:
            print(f"      {bundle.feature_names[i]:<28} {imp[i]:.4f}")
    except (ValueError, AttributeError, TypeError):
        return


def save_artifacts(bundle, stem: str):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, f"{stem}.joblib")
    meta_path = os.path.join(config.MODEL_DIR, f"{stem}.json")
    ml_models.save_bundle(bundle, model_path)
    metadata = {
        "model_name": bundle.model_name,
        "task": bundle.task,
        "data_source": bundle.data_source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": getattr(bundle, "cv_metrics", {}).get("n_samples", 0),
        "label_map": bundle.label_map,
        "feature_names": bundle.feature_names,
        "cv_metrics": bundle.cv_metrics,
        "baseline_metrics": bundle.baseline_metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✅ Saved {model_path}")
    return model_path


def train_sports_dataset(model_name: str, epochs: int, seed: int):
    print("\n" + "=" * 70)
    print("DATASET: Multimodal Sports Injury Dataset (3-class, grouped CV by athlete)")
    print("=" * 70)
    df = load_sports_injury()
    info = sports_summary(df)
    print(f"  samples={info['n_samples']} athletes={info['n_athletes']} "
          f"missing={info['missing_frac']:.2%} classes={info['class_dist']}")

    imputer = SimpleImputer(strategy="median")

    if model_name in SEQUENCE_MODELS:
        X_seq, y_seq, groups_seq, feat_names = build_sequences(df, config.SPORTS_INJURY_SEQUENCE_WINDOW)
        print(f"  sequences: {X_seq.shape} (window={config.SPORTS_INJURY_SEQUENCE_WINDOW}, "
              f"predicting next session)")
        t0 = time.time()
        bundle = ml_models.train_sequence_model(
            X_seq, y_seq, model_name, SPORTS_LABEL_MAP, groups=groups_seq,
            n_folds=5, epochs=epochs, data_source="real")
        print_report(bundle, title="sequence")
        save_artifacts(bundle, f"sports_injury_{model_name}")
        print(f"  elapsed {time.time() - t0:.0f}s")
        return bundle

    X, y, groups, feat_names = preprocess_sports_injury(df)
    print(f"  features={len(feat_names)} ({len(config.SPORTS_INJURY_NUMERIC_FEATURES)} numeric + "
          f"temporal + categorical)")
    t0 = time.time()
    bundle = ml_models.train_injury_model(
        X, y, model_name, feature_names=feat_names, label_map=SPORTS_LABEL_MAP,
        data_source="real", groups=groups, imputer=imputer)
    print_report(bundle)
    print_importances(bundle)
    save_artifacts(bundle, f"sports_injury_{model_name}")
    print(f"  elapsed {time.time() - t0:.0f}s")
    return bundle


def train_cricket_dataset(model_name: str, target: str, seed: int):
    if model_name in SEQUENCE_MODELS:
        print(f"\n⏭️  Skipping {model_name}: cricket dataset has one row per player-season "
              "(no per-player session time series for sequence models).")
        return None

    label_map = CRICKET_LABEL_MAP if target == "injury" else CRICKET_SEVERITY_LABEL_MAP
    print("\n" + "=" * 70)
    print(f"DATASET: Cricket Injury Dataset (target={target})")
    print("=" * 70)
    df = load_cricket_injury()
    info = cricket_summary(df)
    print(f"  samples={info['n_samples']} players={info['n_players']} "
          f"seasons={info['seasons']}")
    X, y, groups, feat_names, df_prep = preprocess_cricket_injury(df)
    target_col = (config.CRICKET_INJURY_TARGET if target == "injury"
                  else config.CRICKET_INJURY_SEVERITY_TARGET)
    y = df_prep[target_col].to_numpy(dtype=int)
    print(f"  class_dist({target}) = {df_prep[target_col].value_counts().sort_index().to_dict()}")
    imputer = SimpleImputer(strategy="median")
    t0 = time.time()
    bundle = ml_models.train_injury_model(
        X, y, model_name, feature_names=feat_names, label_map=label_map,
        data_source="real", groups=None, imputer=imputer)
    print_report(bundle)
    print_importances(bundle)
    stem = f"cricket_{target}_{model_name}"
    save_artifacts(bundle, stem)
    print(f"  elapsed {time.time() - t0:.0f}s")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train injury models on real datasets")
    parser.add_argument("--dataset", type=str, default="sports",
                        choices=["sports", "cricket", "both"])
    parser.add_argument("--model", type=str, default="all",
                        choices=ALL_MODELS + ["all"])
    parser.add_argument("--target", type=str, default="injury",
                        choices=["injury", "severity"],
                        help="cricket dataset target: injury_status (binary) or "
                             "injury_severity (ordinal 0/1/2)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="epochs for sequence models (cnn_lstm/transformer)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    models = ALL_MODELS if args.model == "all" else [args.model]

    if args.dataset in ("sports", "both"):
        for m in models:
            train_sports_dataset(m, args.epochs, args.seed)
    if args.dataset in ("cricket", "both"):
        for m in models:
            train_cricket_dataset(m, args.target, args.seed)

    print("\n" + "=" * 70)
    print("BACKENDS")
    for k, v in ml_models.BACKEND_INFO.items():
        print(f"  • {k}: {'available' if v else 'not installed'}")


if __name__ == "__main__":
    main()
