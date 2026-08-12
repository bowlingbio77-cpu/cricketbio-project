"""
Trains the performance-assessment and injury-risk models and saves them to
models/, so the Streamlit dashboard has something to load immediately.

Usage:
    python train_demo_model.py                       # synthetic demo data, random_forest
    python train_demo_model.py --model xgboost
    python train_demo_model.py --data path/to/real_dataset.csv   # once you have real labels

Real dataset CSV must have columns: config.FEATURE_NAMES + ["performance_score", "injury_risk"]
where injury_risk is in {0,1,2} = {low, moderate, high}.

Reports honest 5-fold cross-validated metrics and compares them against trivial
baselines (mean-predictor / majority-class) so you can see whether the model
actually adds predictive value. Models trained on the synthetic demo data are
flagged as such -- near-perfect metrics there are expected because the labels
are generated from the same features, and say nothing about real-world accuracy.
"""
import argparse
import os
import pandas as pd

from src import config, ml_models
from src.synthetic_data import generate_synthetic_dataset

# Expected plausible ranges per feature (used for sanity checks only -- not enforced).
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


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Raise clear errors on missing columns / bad labels; drop unusable rows."""
    required = config.FEATURE_NAMES + [config.PERFORMANCE_TARGET, config.INJURY_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Dataset is missing required columns: {missing}\n"
            f"Expected: {required}"
        )

    df = df.dropna(subset=required).copy()
    if len(df) == 0:
        raise SystemExit("No complete rows after dropping rows with missing values.")

    bad_risk = sorted(set(df[config.INJURY_TARGET].unique()) - {0, 1, 2})
    if bad_risk:
        raise SystemExit(
            f"injury_risk must contain only {{0,1,2}} (0=low,1=moderate,2=high), "
            f"found: {bad_risk}"
        )

    non_numeric = [c for c in config.FEATURE_NAMES
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SystemExit(f"Features must be numeric, found non-numeric: {non_numeric}")

    for c, (lo, hi) in FEATURE_SANITY_RANGES.items():
        outside = df[(df[c] < lo) | (df[c] > hi)]
        if len(outside):
            print(f"  ⚠ warning: {len(outside)} rows have {c} outside the plausible range "
                  f"[{lo}, {hi}] (min={df[c].min():.2f}, max={df[c].max():.2f}).")
    return df


def print_report(bundle):
    cv = getattr(bundle, "cv_metrics", None) or {}
    base = getattr(bundle, "baseline_metrics", None) or {}
    if bundle.task == "performance":
        cv_r2 = cv.get("r2_mean", 0.0)
        print(f"  MAE  = {cv.get('mae_mean', 0.0):.2f} +- {cv.get('mae_std', 0.0):.2f} / 100")
        print(f"  RMSE = {cv.get('rmse_mean', 0.0):.2f}")
        print(f"  R2   = {cv_r2:.3f} +- {cv.get('r2_std', 0.0):.3f}   "
              f"(baseline 'always predict mean' R2 = {base.get('r2', 0.0):.3f})")
        if bundle.data_source == "synthetic":
            print("  NOTE: SYNTHETIC demo data -- labels are generated from these features, so "
                  "metrics only measure fit to the demo generator, not real bowling performance.")
        elif cv_r2 > 0.95:
            print("  WARNING: R2 > 0.95 on real data is suspicious -- check for label leakage "
                  "or a feature that is an input to the target.")
    else:
        acc = cv.get("accuracy_mean", 0.0)
        print(f"  Accuracy = {acc:.3f} +- {cv.get('accuracy_std', 0.0):.3f}   "
              f"(baseline 'always majority class' = {base.get('accuracy', 0.0):.3f})")
        print(f"  F1 (macro) = {cv.get('f1_mean', 0.0):.3f} +- {cv.get('f1_std', 0.0):.3f}")
        if bundle.data_source == "synthetic":
            print("  NOTE: SYNTHETIC demo data -- near-perfect accuracy is expected because labels "
                  "are generated from these features. Not evidence of real-world performance.")
        elif acc > 0.98:
            print("  WARNING: Accuracy > 0.98 on real data is suspicious -- check for leakage.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to a labeled CSV. Defaults to synthetic demo data.")
    parser.add_argument("--model", type=str, default="random_forest",
                         choices=["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"])
    parser.add_argument("--n_samples", type=int, default=1500)
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} rows from {args.data}")
        df = validate_dataset(df)
        data_source = "real"
        print(f"Validated: {len(df)} usable rows.")
    else:
        df = generate_synthetic_dataset(n_samples=args.n_samples)
        df.to_csv(os.path.join(config.DATA_DIR, "synthetic_bowling_dataset.csv"), index=False)
        print(f"Generated {len(df)} synthetic rows (no --data given; "
              f"saved to data/synthetic_bowling_dataset.csv)")
        data_source = "synthetic"

    X = df[config.FEATURE_NAMES].values
    y_perf = df[config.PERFORMANCE_TARGET].values
    y_injury = df[config.INJURY_TARGET].values

    print(f"\nTraining performance model ({args.model}) on {data_source} data...")
    perf_bundle = ml_models.train_performance_model(X, y_perf, model_name=args.model,
                                                     data_source=data_source)
    print_report(perf_bundle)

    print(f"\nTraining injury-risk model ({args.model}) on {data_source} data...")
    injury_bundle = ml_models.train_injury_model(X, y_injury, model_name=args.model,
                                                  data_source=data_source)
    print_report(injury_bundle)

    perf_path = os.path.join(config.MODEL_DIR, f"performance_{args.model}.joblib")
    injury_path = os.path.join(config.MODEL_DIR, f"injury_{args.model}.joblib")
    ml_models.save_bundle(perf_bundle, perf_path)
    ml_models.save_bundle(injury_bundle, injury_path)
    print(f"\nSaved:\n  {perf_path}\n  {injury_path}")

    print(f"\nBackends: xgboost={ml_models.BACKEND_INFO['xgboost_available']}  "
          f"catboost={ml_models.BACKEND_INFO['catboost_available']}  "
          f"torch={ml_models.BACKEND_INFO['torch_available']}")


if __name__ == "__main__":
    main()
