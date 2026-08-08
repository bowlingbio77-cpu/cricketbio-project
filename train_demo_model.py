"""
Trains the performance-assessment and injury-risk models and saves them to
models/, so the Streamlit dashboard has something to load immediately.

Usage:
    python train_demo_model.py                       # synthetic demo data, random_forest
    python train_demo_model.py --model xgboost
    python train_demo_model.py --data path/to/real_dataset.csv   # once you have real labels

Real dataset CSV must have columns: config.FEATURE_NAMES + ["performance_score", "injury_risk"]
where injury_risk is in {0,1,2} = {low, moderate, high}.
"""
import argparse
import os
import pandas as pd

from src import config, ml_models
from src.synthetic_data import generate_synthetic_dataset


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
    else:
        df = generate_synthetic_dataset(n_samples=args.n_samples)
        df.to_csv(os.path.join(config.DATA_DIR, "synthetic_bowling_dataset.csv"), index=False)
        print(f"Generated {len(df)} synthetic rows (no --data given; "
              f"saved to data/synthetic_bowling_dataset.csv)")

    X = df[config.FEATURE_NAMES].values
    y_perf = df[config.PERFORMANCE_TARGET].values
    y_injury = df[config.INJURY_TARGET].values

    print(f"\nTraining performance model ({args.model})...")
    perf_bundle = ml_models.train_performance_model(X, y_perf, model_name=args.model)
    print(f"  MAE={perf_bundle.metrics['mae']:.2f}  R2={perf_bundle.metrics['r2']:.3f}")

    print(f"\nTraining injury-risk model ({args.model})...")
    injury_bundle = ml_models.train_injury_model(X, y_injury, model_name=args.model)
    print(f"  Accuracy={injury_bundle.metrics['accuracy']:.3f}  F1(macro)={injury_bundle.metrics['f1_macro']:.3f}")

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
