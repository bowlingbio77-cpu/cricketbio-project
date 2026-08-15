"""
Trains performance-assessment and injury-risk models and exports bundled
artifacts to models/ along with training metadata for dashboard inspection.

Usage:
    python train_demo_model.py                       # Clinical-benchmark synthetic, Random Forest
    python train_demo_model.py --model xgboost       # Train XGBoost
    python train_demo_model.py --model all           # Benchmark all available models
    python train_demo_model.py --data dataset.csv    # Train on real labeled data
    python train_demo_model.py --synthetic binary    # Old binary demo generator instead of clinical

Injury target: 0=low, 1=moderate, 2=high. The default synthetic source
(generate_clinical_synthetic_dataset) derives these 3 classes from the trigger
thresholds in data/cricket_injury_recovery_benchmarks.json.
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from src import config, ml_models
from src.synthetic_data import generate_synthetic_dataset, generate_clinical_synthetic_dataset

INJURY_LABEL_MAP = {0: "low", 1: "moderate", 2: "high"}

FEATURE_SANITY_RANGES = {
    "shoulder_rotation_deg": (0, 90),
    "elbow_flexion_deg": (0, 45),
    "wrist_angle_deg": (90, 180),
    "hip_rotation_deg": (0, 80),
    "knee_flexion_deg": (0, 60),
    "trunk_lean_deg": (0, 60),
    "stride_length_norm": (0.3, 1.6),
    "release_angle_deg": (30, 90),
    "angular_velocity_deg_s": (100, 1500),
    "ground_contact_time_s": (0.05, 0.35),
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate column types, targets, and check sanity bounds."""
    required = config.FEATURE_NAMES + [config.PERFORMANCE_TARGET, config.INJURY_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ Missing required columns: {missing}\nExpected: {required}")

    df = df.dropna(subset=required).copy()
    if len(df) == 0:
        raise SystemExit("❌ Dataset is empty after dropping missing values.")

    bad_risk = sorted(set(df[config.INJURY_TARGET].unique()) - {0, 1, 2})
    if bad_risk:
        raise SystemExit(f"❌ injury_risk must only contain {{0, 1, 2}}, found: {bad_risk}")

    non_numeric = [c for c in config.FEATURE_NAMES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SystemExit(f"❌ Non-numeric feature columns found: {non_numeric}")

    # Biomechanical plausibility check
    for c, (lo, hi) in FEATURE_SANITY_RANGES.items():
        outside = df[(df[c] < lo) | (df[c] > hi)]
        if len(outside):
            print(f"  ⚠️  Warning: {len(outside)} rows have '{c}' outside typical range "
                  f"[{lo}, {hi}] (min={df[c].min():.1f}, max={df[c].max():.1f}).")
    return df


def print_report(bundle):
    cv = getattr(bundle, "cv_metrics", None) or {}
    base = getattr(bundle, "baseline_metrics", None) or {}

    if bundle.task == "performance":
        cv_r2 = cv.get("r2_mean", 0.0)
        print(f"  • MAE  : {cv.get('mae_mean', 0.0):.2f} ± {cv.get('mae_std', 0.0):.2f} pts")
        print(f"  • RMSE : {cv.get('rmse_mean', 0.0):.2f}")
        print(f"  • R²   : {cv_r2:.3f} ± {cv.get('r2_std', 0.0):.3f} (Trivial Mean Baseline R² = {base.get('r2', 0.0):.3f})")
        if bundle.data_source == "synthetic":
            print("    ℹ️  NOTE: Trained on SYNTHETIC data. High fit is expected.")
        elif cv_r2 > 0.95:
            print("    ⚠️  WARNING: R² > 0.95 on real data indicates potential target leakage.")
    else:
        acc = cv.get("accuracy_mean", 0.0)
        print(f"  • Accuracy : {acc:.3f} ± {cv.get('accuracy_std', 0.0):.3f} (Majority Class Baseline = {base.get('accuracy', 0.0):.3f})")
        print(f"  • F1-Macro : {cv.get('f1_mean', 0.0):.3f} ± {cv.get('f1_std', 0.0):.3f}")
        print(f"  • Classes  : {sorted(bundle.label_map.items())}")
        if bundle.data_source == "synthetic":
            print("    ℹ️  NOTE: Trained on SYNTHETIC data.")
        elif acc > 0.98:
            print("    ⚠️  WARNING: Accuracy > 0.98 on real data indicates potential target leakage.")


def train_single_model(model_name: str, X: np.ndarray, y_perf: np.ndarray,
                       y_injury: np.ndarray, data_source: str):
    print(f"\n{'='*20} Training Model: {model_name.upper()} {'='*20}")
    t0 = time.time()

    print(f"▶ [1/2] Training Performance Regressor ({model_name})...")
    perf_bundle = ml_models.train_performance_model(X, y_perf, model_name=model_name, data_source=data_source)
    print_report(perf_bundle)

    print(f"\n▶ [2/2] Training Injury-Risk Classifier ({model_name})...")
    injury_bundle = ml_models.train_injury_model(X, y_injury, model_name=model_name,
                                                 label_map=INJURY_LABEL_MAP,
                                                 data_source=data_source)
    print_report(injury_bundle)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    perf_path = os.path.join(config.MODEL_DIR, f"performance_{model_name}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{model_name}.joblib")
    ml_models.save_bundle(perf_bundle, perf_path)
    ml_models.save_bundle(injury_bundle, injury_path)

    # Save summary metadata JSON for Streamlit integration
    meta_path = os.path.join(config.MODEL_DIR, f"metadata_{model_name}.json")
    metadata = {
        "model_name": model_name,
        "data_source": data_source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "injury_label_map": INJURY_LABEL_MAP,
        "performance_metrics": getattr(perf_bundle, "cv_metrics", {}),
        "injury_metrics": getattr(injury_bundle, "cv_metrics", {}),
        "training_time_sec": round(time.time() - t0, 2)
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Artifacts saved successfully:\n   • {perf_path}\n   • {injury_path}\n   • {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="PaceAI Model Training & Benchmark Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to labeled CSV (defaults to synthetic)")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer", "all"])
    parser.add_argument("--n_samples", type=int, default=1500, help="Number of synthetic samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--synthetic", type=str, default="clinical",
                        choices=["clinical", "binary"],
                        help="Which synthetic generator to use when --data is not given. "
                             "'clinical' = benchmark-grounded 3-class injury target (recommended); "
                             "'binary' = deprecated legacy alias (also emits 3-class severity)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load / Generate Data
    if args.data:
        df = pd.read_csv(args.data)
        print(f"📂 Loaded {len(df)} rows from {args.data}")
        df = validate_dataset(df)
        data_source = "real"
    else:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        if args.synthetic == "clinical":
            df = generate_clinical_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
        else:
            df = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
        synth_path = os.path.join(config.DATA_DIR, "synthetic_bowling_dataset.csv")
        df.to_csv(synth_path, index=False)
        print(f"🎲 Generated {len(df)} synthetic samples [{args.synthetic}] (Saved to: {synth_path})")
        data_source = "synthetic"

    X = df[config.FEATURE_NAMES].values
    y_perf = df[config.PERFORMANCE_TARGET].values
    y_injury = df[config.INJURY_TARGET].values

    # Check Class Balance for Injury Target
    class_counts = pd.Series(y_injury).value_counts().sort_index().to_dict()
    print(f"📊 Class Distribution ({INJURY_LABEL_MAP}): {class_counts}")

    models_to_train = (
        ["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"]
        if args.model == "all"
        else [args.model]
    )

    for m in models_to_train:
        train_single_model(m, X, y_perf, y_injury, data_source)

    print(f"\n🚀 System Backends:")
    for k, v in ml_models.BACKEND_INFO.items():
        print(f"   • {k}: {'Available' if v else 'Not installed (fallback active)'}")


if __name__ == "__main__":
    main()
