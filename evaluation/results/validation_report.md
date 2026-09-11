# PaceAI P1 -- Pilot Validation Report

> **Ground-truth validation is performed independently of the PaceAI prediction pipeline.**
>
> **The current 8-clip dataset represents pilot validation and is insufficient by itself to establish population-level generalization.**
>
> **The validation evaluator is implemented, but no real accuracy statistics are reported until independent annotations are available.**

## Status

**NO REAL GROUND TRUTH -- INSUFFICIENT DATA.**

No independent ground-truth annotations have been entered yet (`evaluation/ground_truth/ground_truth.csv` is schema-only), so no accuracy statistic of any kind is reported.

The 67.5 -> 80.0 value previously reported is a **plausibility-vs-published-ranges** score and a bug-fix verification -- it is NOT measurement accuracy and is NOT validation.

All feature statistics below report **insufficient data**.
## Cross-checks

- Ground truth source: `D:\cricket_biomech_ai\evaluation\ground_truth\ground_truth.csv`
- Prediction cache: `D:\cricket_biomech_ai\evaluation\predictions`
- Annotated clips: none
- Clips with predictions: ['fast_left_00000001', 'fast_left_0000001', 'fast_right_00000001', 'fast_right_00000002', 'leg_right_00000001', 'leg_right_00000029', 'off_left_00000001', 'off_right_00000042']

### Excluded features (no valid comparison declared)

| Feature | Reason |
|---|---|
| stride_length_px | unit mismatch: ground truth is planar pixels, PaceAI estimates a normalized value. Comparable only after a shared normalized unit is agreed in the protocol. |
| release_speed_mps | not measurable from uncalibrated camera footage (needs pixel-per-metre calibration) and not produced by the PaceAI prediction cache. |

## Ground-truth data validation

No validation issues in the current ground-truth rows.

## Per-feature agreement summary

`insufficient data` = fewer valid paired samples than the statistic requires.

| Feature | n | status | bias | MAE | RMSE | sd(err) | r | R^2 | LoA low | LoA high | ICC(2,1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| front_knee_angle_deg | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |
| elbow_flexion_deg | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |
| trunk_lean_deg | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |
| release_angle_deg | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |
| release_frame | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |
| front_contact_frame | 0 | no_paired_samples | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data | insufficient data |

### Frame-event errors (frames; milliseconds at preprocessed 20 FPS for GT-vs-prediction)

| Feature | n | mean abs (frames) | median abs (frames) | mean abs (ms) | median abs (ms) |
|---|---|---|---|---|---|

## Statistical integrity

- Min samples for bias/MAE/RMSE: 1; sd of error: 2; Bland-Altman LoA: 3; correlation/R^2: 4; ICC(2,1): 3.
- ICC(2,1): two-way random-effects, single measures (ground-truth and prediction treated as two raters).
- R^2 reported here is the squared Pearson correlation.
- Results are **PILOT VALIDATION**, not a population-level generalization claim.