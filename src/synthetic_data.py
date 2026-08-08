"""
Synthetic demo dataset generator.

No public, labeled (features -> performance score, injury risk) dataset for
cricket bowling biomechanics ships with this project. To make the dashboard
usable immediately (and to let the ML/SHAP/coaching modules be exercised and
tested end-to-end), this module generates a synthetic dataset whose feature
ranges and label-generating rules are grounded in published fast-bowling
biomechanics ranges (ICC elbow-extension law, typical elite trunk-lean/knee-
flexion/stride-length bands referenced in coaching.py).

Replace this with `pd.read_csv(<your real labeled dataset>)` as soon as you
have one -- train_demo_model.py works with either.
"""
import numpy as np
import pandas as pd
from . import config


def generate_synthetic_dataset(n_samples: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "shoulder_rotation_deg": rng.normal(40, 15, n_samples).clip(0, 90),
        "elbow_flexion_deg": rng.gamma(2.0, 5.0, n_samples).clip(0, 45),  # most bowlers legal (<15), tail illegal
        "wrist_angle_deg": rng.normal(150, 20, n_samples).clip(90, 180),
        "hip_rotation_deg": rng.normal(35, 12, n_samples).clip(0, 80),
        "knee_flexion_deg": rng.normal(18, 10, n_samples).clip(0, 60),
        "trunk_lean_deg": rng.normal(27, 10, n_samples).clip(0, 60),
        "stride_length_norm": rng.normal(0.95, 0.2, n_samples).clip(0.3, 1.6),
        "release_angle_deg": rng.normal(75, 10, n_samples).clip(30, 90),
        "angular_velocity_deg_s": rng.normal(700, 200, n_samples).clip(100, 1500),
        "ground_contact_time_s": rng.normal(0.15, 0.05, n_samples).clip(0.05, 0.35),
    })

    # --- Performance score: rewards good rotation/velocity, penalizes poor mechanics ---
    perf = (
        0.25 * _norm01(df.shoulder_rotation_deg, 0, 90)
        + 0.20 * _norm01(df.angular_velocity_deg_s, 100, 1500)
        + 0.15 * _norm01(df.hip_rotation_deg, 0, 80)
        + 0.15 * (1 - np.abs(df.stride_length_norm - 1.0) / 0.7).clip(0, 1)
        + 0.15 * _norm01(df.release_angle_deg, 30, 90)
        - 0.10 * _norm01(df.trunk_lean_deg, 0, 60)
    )
    perf = (perf * 100).clip(0, 100) + rng.normal(0, 5, n_samples)
    df["performance_score"] = perf.clip(0, 100)

    # --- Injury risk: driven by elbow extension, trunk lean, knee flexion, contact time ---
    risk_score = (
        1.5 * (df.elbow_flexion_deg > config.ICC_ELBOW_EXTENSION_LIMIT_DEG).astype(float)
        + 1.2 * _norm01(df.trunk_lean_deg, 15, 60)
        + 1.0 * _norm01(df.knee_flexion_deg, 5, 60)
        + 0.8 * _norm01(df.ground_contact_time_s, 0.1, 0.35)
        + rng.normal(0, 0.3, n_samples)
    )
    df["injury_risk"] = pd.cut(
        risk_score, bins=[-np.inf, 1.0, 2.2, np.inf], labels=[0, 1, 2]
    ).astype(int)  # 0=low, 1=moderate, 2=high

    return df


def _norm01(series, lo, hi):
    return ((series - lo) / (hi - lo)).clip(0, 1)


if __name__ == "__main__":
    import os
    df = generate_synthetic_dataset()
    out_path = os.path.join(config.DATA_DIR, "synthetic_bowling_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.describe())
