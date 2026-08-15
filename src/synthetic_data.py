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


# Feature ranges mirrored from app.py FEATURE_LABELS sliders so that every
# value reachable in the UI is inside the training distribution.
FEATURE_BOUNDS = {
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


def compute_performance_score(df: pd.DataFrame) -> pd.Series:
    """
    Biomechanically-grounded 0-100 performance target for fast bowling.

    Each component is a 0-1 "how close to elite" score; weights sum to 1.0.
    Grounding (fast-bowling literature / ICC law 21.5):

      * angular velocity  -> raw arm speed (pace driver), higher is better
      * knee flexion      -> front-knee BRACE at landing: ideal small (< ~15 deg);
                             41 deg knee is a lever-efficiency failure (w = 0)
      * elbow flexion     -> ICC action legality: flexion should be small (<= 15 deg)
      * shoulder rotation -> counter-rotation: low is efficient + lumbar-safe
      * trunk lateral lean-> moderate (~22 deg) is optimal; extreme lean = lumbar risk
      * stride length     -> long but NOT over-striding (ideal ~1.05 x stature)
      * hip rotation      -> pelvis/shoulder separation, moderate ~40 deg optimal
      * release angle     -> release high and over, ideal ~75 deg
      * ground contact    -> short, explosive front-foot contact, ideal ~0.09 s
      * wrist angle       -> wrist snap at release, higher is better

    This deliberately makes performance AGREE with the injury model: the same
    mechanics that score low (weak knee brace, high counter-rotation, trunk
    collapse) are the ones the clinical trigger thresholds flag as injury risk.
    """
    w_av = 0.18 * _norm01(df.angular_velocity_deg_s, 100, 1500)
    w_knee = 0.16 * (1 - (df.knee_flexion_deg - 6).clip(lower=0) / 30).clip(0, 1)
    w_elbow = 0.13 * (1 - df.elbow_flexion_deg / 30).clip(0, 1)
    w_shoulder = 0.11 * (1 - df.shoulder_rotation_deg / 60).clip(0, 1)
    w_trunk = 0.09 * (1 - (df.trunk_lean_deg - 22).abs() / 30).clip(0, 1)
    w_stride = 0.09 * (1 - (df.stride_length_norm - 1.05).abs() / 0.6).clip(0, 1)
    w_hip = 0.08 * (1 - (df.hip_rotation_deg - 40).abs() / 40).clip(0, 1)
    w_release = 0.07 * (1 - (df.release_angle_deg - 75).abs() / 40).clip(0, 1)
    w_gct = 0.06 * (1 - (df.ground_contact_time_s - 0.09).clip(lower=0) / 0.2).clip(0, 1)
    w_wrist = 0.03 * _norm01(df.wrist_angle_deg, 90, 180)
    score = (w_av + w_knee + w_elbow + w_shoulder + w_trunk
             + w_stride + w_hip + w_release + w_gct + w_wrist) * 100
    return score.clip(0, 100)


def generate_clinical_synthetic_dataset(n_samples: int = 4000,
                                        seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Benchmark-grounded synthetic dataset for the 3-class injury-risk model.

    Injury severity (0=low, 1=moderate, 2=high) is derived directly from the
    clinical trigger thresholds in data/cricket_injury_recovery_benchmarks.json:

      * shoulder counter-rotation  > 30 deg   (lumbar stress fracture)
      * lateral trunk flexion      > 35 deg   (lumbar / abdominal)
      * front-knee flexion         > 30 deg   (patellar: knee angle < 150 deg)
      * stride length              > 120% stature (hamstring overstriding)

    The trigger count is mapped 0 -> low, 1 -> moderate, >=2 -> high, with a
    small label-noise term so the task is not trivially separable. Performance
    score uses compute_performance_score (biomechanically grounded, agrees with
    the injury triggers). ~30% of samples are drawn uniformly over the full UI
    slider ranges so the model stays accurate at the extremes users can reach.
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
    stride_th = _threshold("stride_length_norm", 1.2)

    rng = np.random.default_rng(seed)

    # ~70% of samples from realistic clinical marginals, ~30% drawn uniformly
    # over the full UI slider ranges so the trained models learn the surface
    # at the corners too (users can push every slider to its extreme).
    n_clin = int(round(0.7 * n_samples))
    n_unif = n_samples - n_clin

    clin = pd.DataFrame({
        "shoulder_rotation_deg": rng.normal(30, 11, n_clin).clip(0, 90),
        "elbow_flexion_deg": rng.gamma(2.0, 5.0, n_clin).clip(0, 45),
        "wrist_angle_deg": rng.normal(150, 20, n_clin).clip(90, 180),
        "hip_rotation_deg": rng.normal(35, 12, n_clin).clip(0, 80),
        "knee_flexion_deg": rng.normal(12, 9, n_clin).clip(0, 60),
        "trunk_lean_deg": rng.normal(20, 10, n_clin).clip(0, 60),
        "stride_length_norm": rng.normal(0.82, 0.22, n_clin).clip(0.3, 1.6),
        "release_angle_deg": rng.normal(75, 10, n_clin).clip(30, 90),
        "angular_velocity_deg_s": rng.normal(700, 200, n_clin).clip(100, 1500),
        "ground_contact_time_s": rng.normal(0.15, 0.05, n_clin).clip(0.05, 0.35),
    })
    unif = pd.DataFrame({
        name: rng.uniform(lo, hi, n_unif)
        for name, (lo, hi) in FEATURE_BOUNDS.items()
    })
    df = pd.concat([clin, unif], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df["performance_score"] = compute_performance_score(df) + rng.normal(0, 4, n_samples)
    df["performance_score"] = df["performance_score"].clip(0, 100)

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
