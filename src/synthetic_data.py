"""
Synthetic demo dataset generator.

No public, labeled (features -> performance score, injury risk) dataset for
cricket bowling biomechanics ships with this project. To make the dashboard
usable immediately (and to let the ML/SHAP/coaching modules be exercised and
tested end-to-end), this module generates a synthetic dataset whose feature
ranges and label-generating rules are grounded in published fast-bowling
biomechanics ranges (ICC elbow-extension law, typical elite trunk-lean/knee-
flexion/stride-length bands referenced in coaching.py).

Injury risk is modelled as a *3-class severity level* (0=low, 1=moderate,
2=high) derived from clinical trigger thresholds in
data/cricket_injury_recovery_benchmarks.json (see
generate_clinical_synthetic_dataset). The trained model therefore predicts
P(severity) per delivery.

Replace this with `pd.read_csv(<your real labeled dataset>)` as soon as you
have one -- train_demo_model.py works with either.
"""
import warnings

import numpy as np
import pandas as pd
from . import config


def generate_synthetic_dataset(n_samples: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Deprecated alias for generate_clinical_synthetic_dataset.

    Kept for backward compatibility (the app's no-bundle fallback and the
    ``--synthetic binary`` CLI path both call it). The earlier binary hazard
    model saturated to a single class for typical exposures, so it is no longer
    generated here.
    """
    warnings.warn(
        "generate_synthetic_dataset is deprecated; use generate_clinical_synthetic_dataset.",
        DeprecationWarning, stacklevel=2)
    return generate_clinical_synthetic_dataset(n_samples=n_samples, seed=seed)


def _norm01(series, lo, hi):
    return ((series - lo) / (hi - lo)).clip(0, 1)


def generate_clinical_synthetic_dataset(n_samples: int = 1500,
                                        seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Benchmark-grounded synthetic dataset for the 3-class injury-risk model.

    Injury severity (0=low, 1=moderate, 2=high) is derived directly from the
    clinical trigger thresholds in data/cricket_injury_recovery_benchmarks.json:

      * shoulder counter-rotation  > 30 deg   (lumbar stress fracture)
      * lateral trunk flexion      > 35 deg   (lumbar / abdominal)
      * front-knee flexion         > 30 deg   (patellar: knee angle < 150 deg)
      * stride length              > 88% stature (hamstring overstriding)

    The trigger count is mapped 0 -> low, 1 -> moderate, >=2 -> high, with a
    small label-noise term so the task is not trivially separable. Performance
    score uses the same kinematic formula as generate_synthetic_dataset.
    """
    from . import injury_knowledge_base as kb

    bench = kb._kb()

    def _threshold(key, fallback: float) -> float:
        for entry in bench.get("injuries", []):
            if key in entry.get("primary_triggers", {}):
                return float(entry["primary_triggers"][key]["threshold"])
        return fallback

    shoulder_th = _threshold("shoulder_counter_rotation", 30.0)
    trunk_th = _threshold("lateral_trunk_flexion", 35.0)
    knee_th = 180.0 - _threshold("knee_angle_ffc", 150.0)      # -> flexion threshold
    stride_th = _threshold("stride_length_norm", 0.88)

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "shoulder_rotation_deg": rng.normal(30, 11, n_samples).clip(0, 90),
        "elbow_flexion_deg": rng.gamma(2.0, 5.0, n_samples).clip(0, 45),
        "wrist_angle_deg": rng.normal(150, 20, n_samples).clip(90, 180),
        "hip_rotation_deg": rng.normal(35, 12, n_samples).clip(0, 80),
        "knee_flexion_deg": rng.normal(12, 9, n_samples).clip(0, 60),
        "trunk_lean_deg": rng.normal(20, 10, n_samples).clip(0, 60),
        "stride_length_norm": rng.normal(0.82, 0.22, n_samples).clip(0.3, 1.6),
        "release_angle_deg": rng.normal(75, 10, n_samples).clip(30, 90),
        "angular_velocity_deg_s": rng.normal(700, 200, n_samples).clip(100, 1500),
        "ground_contact_time_s": rng.normal(0.15, 0.05, n_samples).clip(0.05, 0.35),
    })

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

    # --- 3-class clinical severity from benchmark trigger thresholds ---
    triggers = (
        (df.shoulder_rotation_deg > shoulder_th).astype(int)
        + (df.trunk_lean_deg > trunk_th).astype(int)
        + (df.knee_flexion_deg > knee_th).astype(int)
        + (df.stride_length_norm > stride_th).astype(int)
    )
    severity = np.where(triggers >= 2, 2, np.where(triggers == 1, 1, 0))

    # Small label noise (~8%) so the task isn't trivially separable
    flip = rng.uniform(size=n_samples) < 0.08
    jitter = rng.integers(-1, 2, size=n_samples)
    severity = severity + np.where(flip, jitter, 0)
    severity = np.clip(severity, 0, 2)

    df[config.INJURY_EXPOSURE_FEATURE] = rng.uniform(100, 1500, n_samples)
    df[config.INJURY_TARGET] = severity.astype(int)
    return df


if __name__ == "__main__":
    import os
    df = generate_synthetic_dataset()
    out_path = os.path.join(config.DATA_DIR, "synthetic_bowling_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.describe())
