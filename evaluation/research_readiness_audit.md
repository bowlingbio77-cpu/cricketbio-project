# PaceAI Research Readiness Audit

Date: 2026-09-10
Scope: full read-only audit of the codebase, datasets, ground-truth availability, biomechanical math, ML methodology, claims, and validation evidence. NO code was modified during this audit.

---

## 1. Executive summary

PaceAI is a genuinely engineered prototype: a full CV pipeline (detection -> tracking -> MediaPipe pose -> biomechanical feature engineering -> ML -> coaching/SHAP) with 184 passing tests, honest synthetic-data disclaimers, grouped-by-person cross-validation, baseline comparisons, OOD guards, and an evaluation framework. What it is NOT (yet) is a validated research measurement system.

The single most important finding: **there is no real biomechanical ground truth anywhere in the repository.** Every dataset is either synthetic (feature-derived labels), auto-generated (ball annotations), or real but unannotated video (8 clips). Consequently no accuracy/reliability/validity metric against a known delivery can currently be computed. The 67.5 -> 80.0 "improvement" from the recent fix is a *plausibility-vs-published-ranges* score, not a validation result.

Research-readiness score: **47 / 100** (see Section 10). The engineering is strong; the evidence base for scientific claims is weak and is dependent on data collection that does not exist yet.

---

## 2. Research question evaluation

Question posed by the project: *"Can consumer-grade side-view cricket videos, MediaPipe world landmarks, and small tree/sequence models provide biomechanically-grounded fast-bowling performance assessment and injury-risk screening?"*

| Aspect | Assessment |
|---|---|
| Falsifiable | PARTIAL. The risk question is operationalized (3-class ordinal from published trigger thresholds) but the thresholds are applied to synthetic inputs, so the real-world hypothesis is never tested. |
| Answerable with current data | NO. Requires (a) labeled deliveries (release frame, ground truth joint angles) and (b) an outcome or expert-label dataset. Neither exists in repo. |
| Specific | PARTIAL. "Performance score" target is generated from the same features used as input (compute_performance_score), making the ML performance task circular. |
| Novel/citable angle | The *methodological* contribution (single-camera, markerless, view-independent world-landmark kinematics + honest uncertainty reporting) is real but needs a validation study to be claimed. |

Recommendation: reframe the research contribution as *"markerless single-camera estimation of fast-bowling delivery kinematics, with calibration and validation against mocap/results"* rather than *"injury-risk prediction"* (which cannot be claimed yet).

---

## 3. Dataset and ground-truth inventory

| Dataset | Location | Real? | Samples | Players | Deliveries | Annotations | Biomech GT | Same features as pipeline? | Research suitability |
|---|---|---|---|---|---|---|---|---|---|
| Synthetic bowling | data/synthetic_bowling_dataset.csv (4001 rows) | NO (synthetic) | 4001 | n/a | n/a | labels derived FROM features | YES (self-referential) | default training set | Only for smoke/sanity |
| Multimodal sports injury | data/multimodal_sports_injury_dataset.csv (15421 rows) | YES (Kaggle, multi-sport) | 15421 | 156 | n/a | injury_occurred + loads | NO (load/fatigue, not kinematics) | no | 2nd-tier real-data model only; not cricket |
| Cricket injury | data/cricket_injury_dataset.csv (1272 rows) | YES | 1272 | 1272 | n/a | injury_status, severity | NO (age/exposure/days) | no | outcome associations only |
| Ball GT clips | data/gt_clips/ (25 dirs) | NO (rendered synthetic) | ~25 | n/a | synthetic | known ball positions, release/impact | ball-track only | no | tracker unit eval |
| Ball auto-labels | data/cricket_ball_dataset/ (auto_labeled, yolo_dataset) | auto-generated | n/a | n/a | n/a | auto boxes | no | no | YOLO fine-tune demo |
| Recovery benchmarks | data/cricket_injury_recovery_benchmarks.json/.csv | literature | 9 injuries | n/a | n/a | trigger thresholds | no | used for labels | threshold provenance |
| Real video | corrected_all_data/bowling/*.avi (8 clips) | YES | 8 | >=1 | 8 (assumed) | NONE | NO | MANUAL annotation needed | **primary validation target** |
| Eval dataset | evaluation/videos, annotations, predictions | empty | 0 | 0 | 0 | 0 | no | n/a | create infra |
| Eval metadata | evaluation/metadata.csv | header-only | 0 | 0 | 0 | 0 | no | n/a | populate |

**Key conclusion:** per-delivery biomechanical ground truth is currently UNAVAILABLE. All performance/injury "metrics" in the app (categorical claims aside) are computed on models trained on self-referential synthetic data, or on real-but-non-kinematic outcome data.

---

## 4. 25-point audit (0-10, higher = more research-ready)

Legend: STATUS = OK / PARTIAL / FAIL / N-A (not applicable/not attempted).

| # | Criterion | STATUS | SCORE /10 | EVIDENCE | GAP | RECOMMENDATION |
|---|---|---|---|---|---|---|
| 1 | Research question clearly defined | PARTIAL | 5 | README + app framing | Questions are product-shaped, not study-shaped | Write explicit hypothesis + primary/secondary endpoints |
| 2 | Ethical approval / oversight | PARTIAL | 6 | Synthetic data used by default | Real video is used w/o documented consent | Add consent/provenance policy for corrected_all_data |
| 3 | Data provenance documented | OK | 7 | config, dataset_summary(), README; bundles tagged synthetic/real | provenance not machine-readable end-to-end | Emit provenance manifest per artifact |
| 4 | Real ground-truth availability | FAIL | 1 | No mocap/radar/labeled deliveries in repo | Zero GT for features/release/frame | Annotate 8 real clips (release, front-knee, joint angles) |
| 5 | Annotation tooling | PARTIAL | 5 | label_ball.py exists (ball boxes) | No pose/release annotation tool | Add release-frame + pose annotation UI |
| 6 | Dataset size for validation | FAIL | 2 | 8 unannotated clips | n tiny; unknown player diversity | Collect ~100-300 deliveries, >=10 bowlers |
| 7 | Class balance & population repr. | FAIL | 2 | Synthetic severity derived from thresholds w/ 8% noise | No real severity distributions | Use real outcome data or clearly mark demo |
| 8 | Feature-name/math-to-definition match | PARTIAL | 6 | Formulas in feature_engineering.py; Y-axis fixed | "hip_rotation" is pelvic tilt-from-horizontal (angle to horizontal), name misleading; "angular_velocity" is trunk/arm-derived shoulder-rotation rate not arm speed | Rename/annotate features; document each derivation |
| 9 | 3D origin/convention correctness | OK | 8 | world Y-down verified empirically; fixes applied & tested | Only 1 video verified | Spot-check all 8 clips |
| 10 | Release-frame detection validity | PARTIAL | 4 | Heuristic (min elbow flexion after contact, wrist-below-shoulder); edge cases flagged | No GT to validate; 2/8 clips marked unreliable | Validate against annotated releases |
| 11 | Temporal quantities (velocities, durations) | PARTIAL | 5 | finite differences at fixed 20fps | no smoothing; jitter amplifies dtheta/dt | Add Savitzky-Golay + report signal-to-noise |
| 12 | Missing-landmark handling | PARTIAL | 5 | visibility thresholding in pose, wrist-proxy fallback | impact on features unquantified | Report per-frame landmark visibility stats |
| 13 | Coordinate reprojection pitfalls | PARTIAL | 4 | world landmarks used (view-independent) | world origin at hip; no camera calibration needed but arm-swing plane assumptions exist | Document assumptions; test on behind-view + side-view |
| 14 | ML target independence (no leakage) | PARTIAL | 5 | grouped CV, fold-wise scaler/imputer, baselines, OOD | PERFORMANCE target = function of the input features (circular) | Drop synthetic performance model from research claims |
| 15 | Cross-validation correctness | OK | 8 | StratifiedGroupKFold by player/athlete, no leakage | none significant | Recompute on real data when available |
| 16 | Model uncertainty reporting | PARTIAL | 5 | RF prediction interval (16-84%) exists | not calibrated; no conformal | Calibrate on real data; report PICP |
| 17 | Statistical analysis of results | FAIL | 2 | None for real clips | No CI, no Bland-Altman, no ICC vs criterion | Add agreement analysis per metric |
| 18 | Validation framework usable | PARTIAL | 5 | evaluation/evaluate.py complete (mAP, coverage, release error, pose RMSE) | Needs GT + predictions cache; currently returns NO_DATA | Populate dataset; add feature-level and BIAS/SD metrics |
| 19 | Baseline comparison | OK | 8 | trivial baselines (mean / majority) in ml_models | no chance/plausibility baseline for 67.5 score | Add random-permutation baseline |
| 20 | Ablations | FAIL | 0 | none present | no isolation of contributions | Design X-factor, smoothing, world-vs-2D ablations |
| 21 | Robustness testing | PARTIAL | 3 | synthetic GT clips w/ distractors (growing person, occlusion, fixed object) for tracker | no lighting/compression/arm-swap pose tests | Add pose robustness suite using gt_clips + real clips |
| 22 | Claims discipline (UI/docs) | OK | 8 | app.py disclaimers (753-758, 794-796, 1534-38, 1767); README caveats (230-235) | injury-risk wording still "probability" in some copy; coaching ICC wording is "threshold" (caveated) | Sweep UI copy: "biomechanical risk indicator"; ICC -> "screening, not certified test" |
| 23 | Reproducibility (seed/env/artifacts) | PARTIAL | 7 | seeds set, bundles saved w/ metadata, RANDOM_STATE=42 | no requirements lockfile/version pinning for env | Add requirements pins + artifact hash manifest |
| 24 | Literature grounding | PARTIAL | 5 | thresholds from benchmarks.json; README cites ranges | no citation list / DOI in repo | Add CITATION file with primary sources |
| 25 | Engineering quality / tests | OK | 9 | 184 pass; modular; eval harness; OOD warnings; honest reporting | yolo/ByteTrack untested live (torch blocked) | CI + synthetic live-detector test when env permits |
|    | **MEAN** | | **5.2** | | | |

---

## 5. Biomechanical-math audit (per feature)

All 3D features use MediaPipe *world* landmarks (meters, origin at hip centre, Y DOWN verified).

| Feature | Formula (current) | Correct/issue | Risk |
|---|---|---|---|
| shoulder_rotation_deg | angle between horizontal-projections of shoulder line vs hip line; min(theta, 180-theta) | OK X-factor proxy | projection assumes horizontal plane = camera/pitch plane |
| elbow_flexion_deg | 180 - angle_3pt(shoulder,elbow,wrist) | OK "flexion-from-straight"; ICC <=15 is extension-from-straight = same convention | markerless 3D angle is NOT the ICC protocol |
| wrist_angle_deg | angle_3pt(elbow,wrist,index) | OK | index often low-visibility -> invalid |
| hip_rotation_deg | abs(angle_to_horizontal(hip_line)) | NAME MISLEADING: this is pelvic tilt from horizontal, not hip internal/external rotation | misinterpretation |
| knee_flexion_deg | 180 - angle_3pt(hip,knee,ankle) (front leg) | OK | front-leg choice heuristic |
| trunk_lean_deg | angle(trunk, [0,-1,0]) | FIXED (was ~180 offset bug) | none now |
| stride_length_norm | ankle-base horizontal distance at foot contact / body-height (max-y - min-y) | OK proxy | body-height via landmark extent biases short clips |
| release_angle_deg | abs(angle_to_horizontal(shoulder->wrist)) | FIXED sign; OK | defined at release frame only |
| angular_velocity_deg_s | peak |d(shoulder_rotation)/dt| finite diff +/-3 frames | Heuristic IS shoulder-rotation rate, but FEATURE_LABELS labels it "Peak Angular Velocity (arm speed)" | doc/name mismatch |
| ground_contact_time_s | front ankle stationary window (vel < 2% height/s) overlapping release+/- | OK proxy | threshold fractions heuristic |

Correctness fixes already landed: trunk_lean ref vector, _angle_to_horizontal arcsin(-v_y/norm), _wrist_elevation sign. Verified empirically vs y direction. These all pass 184 tests.

Remaining math concerns (no known bug): (1) "hip_rotation" naming, (2) "angular_velocity" label vs derivation, (3) wrist-angle reliance on index visibility, (4) no signal smoothing before finite differences.

---

## 6. Claims audit (what the repo and UI say vs can support)

| Claim | Where | Verdict | Action |
|---|---|---|---|
| "Models trained on SYNTHETIC demo data... say nothing about real-world accuracy" | app.py 753-758 | HONEST | keep |
| "Treat all scores on dashboard as illustrative" | app.py 794-796 | HONEST | keep |
| "This is decision-support, not a substitute for qualified coach/biomechanist/sports physician" | README 239-240 | HONEST | keep |
| "Elbow flexion is a good screening signal, not a certified throwing test; ICC rulings need lab mocap" | README 232-234 | HONEST | keep |
| "ICC 15deg legal-delivery threshold" (surfaced in coaching/app copy) | coaching.py 19 | PARTIAL: presented as flag; README caveat covers it, but UI phrase may over-claim | Use "screening indicator; not an official ICC measurement" in UI copy |
| "Injury risk / probability" wording | app/assistant copy | PARTIAL | Replace with "biomechanical risk indicator" unless clinical validation exists |
| Injury triggers = "clinical benchmark thresholds" | synthetic_data/injury_knowledge_base | PARTIAL: sourced from literature file w/ disclaimer; fine as literature, not clinical validation | cite sources; keep "literature-derived" prefix |
| Real-data sports/cricket injury models (ROC-AUC 0.69-0.72) | README | HONEST (grouped CV on real data; non-kinematic features) | clearly separate from CV-derived kinematics claims |

Overall: claims discipline is unusually good for a prototype. No outright fabricated accuracy claims found. Highest-risk leftover items are UI word choices ("injury probability", "ICC threshold") and the headline performance/injury models being synthetic.

---

## 7. Validation framework design (what to build — NOT implemented now)

Target: report, for each real clip, per-metric agreement against manual expert annotation + optional mocap/radar criterion.

1. Ground truth: annotate release frame, front-knee angle at release, stride/contact window, ball release angle (image), release speed (radar if available).
2. Metrics: per-feature bias (mean difference), SD, MAE, RMSE, R2, correlation, and Bland-Altman limits-of-agreement (LOA 95%) per bowler; ICC(2,1) vs criterion; release-frame |error| in frames and ms.
3. Robustness: 2x pose detector runs, brightness/compression augmentation of real clips, tracking distractor suite (already in gt_clips); report worst-case spread.
4. Research question endpoints: primary = release-frame error distribution; secondary = knee/trunk/elbow angle agreement; hypothesis = single-camera world-landmark kinematics meet pre-specified LOA thresholds (state them BEFORE measuring).
5. Package: extend evaluation/evaluate.py to add feature-level accumulators + Bland-Altman + per-bowler stats; produce evaluation/results/agreement_report.csv + json.
   IMPORTANT: NO ground truth exists yet, so all these metrics are DESIGNED but not computed.

---

## 8. Baseline / ablation / robustness experimental design (no results)

Baselines:
- Feature level: (a) mean-prediction, (b) pose-only no-tracking, (c) 2D image coords instead of world landmarks, (d) no-smoothing vs SG-smoothing.
- Model level: single-delivery RF fit on real features (once GT/outcome exists) vs trivial mean/majority vs a linear regression.
- Tracker: v1 ball_tracking vs v2 multi-hypothesis on gt_clips (recall@15/25, release/impact err) — datasets exist, runnable.

Ablations:
- World vs 2D landmarks (view-origin sensitivity).
- With/without denoise, with/without tracking stabilization.
- Release-detection: elbow-min heuristic vs wrist-below-shoulder vs annotated release.
- Feature set: -wrist_angle (index visibility), -hip_rotation (naming/proxy value), smoothing window sizes.

Robustness:
- Pose repeatability: 5x same clip, landmark std.
- Synthetic pose stress: occlude/freeze joints in gt_clips; measure feature drift.
- Real-clip multi-pass: bounce & lighting variant checks.
Nothing is run in this audit; designs only.

---

## 9. Honest audit of the "67.5 -> 80.0" result

What was done: a scoring script (scripts/compare_validation.py) compares pipeline-extracted features (8 real unannotated clips) against "published fast-bowling ranges" (app.py FEATURE_LABELS targets) using an in-range rubric; identical rubric before/after a fix to the MediaPipe world-Y sign.

Facts:
- Overall plausibility score: BEFORE 67.5/100 -> AFTER 80.0/100 (+12.5).
- Main driver: trunk_lean_deg in-range went 0% -> 75%; shoulder/elbow/knee/stride mostly unchanged; wrist, ground-contact still fail many clips; release frames shifted (e.g. 50-frame clip: release 30->21) which legitimately changes every downstream feature.
- 2/8 clips flagged unreliable (release at edge) and 1 clip (fast_left_0000001 vs 01 naming duplicate) caused double-counting risk.

What this DOES and DOES NOT show:
- DOES show: the sign fix makes extracted kinematics consistent with the documented world-Y-down convention (a real bug fix; the old values were physically implausible, e.g. trunk lean ~90-175 deg).
- DOES NOT show: measurement accuracy, reliability, agreement with any criterion, or that the pipeline is "80% correct". 80.0 is the mean in-range score against a target profile chosen from the same code that calls the features — i.e. a *consistency plausibility* check, not validation against ground truth.

Conclusion: keep the 67.5/80 pair as a smoke-test metric for regression (a "BUG FIX VERIFICATION"), but NEVER present it as research validation. In any publication/README table, label it "plausibility vs internal target ranges".

---

## 10. Research-readiness score

- 25-point mean: 5.2/10.
- Weighted summary (reads: engineering-heavy, evidence-light):
  - Data/GT: 1/10 (no real GT)
  - Math/feature correctness: 6.5/10
  - CV/ML hygiene: 7/10 (no leakage, baselines, OOD)
  - Validation evidence: 2/10
  - Claims discipline: 7.5/10
  - Reproducibility: 6/10
  - Statistical rigor: 3/10

**Composite research-readiness: 47/100.** This is deliberately SEPARATE from the 80.0/100 plausibility score in Section 9, which tests internal consistency, not research validity.

---

## 11. Roadmap

### P0 (before any research claim is made) — make it honest & repeatable
1. UI copy sweep: replace "injury probability/risk" -> "biomechanical risk indicator"; "ICC threshold" -> "screening indicator (not official ICC measurement)".
2. Add `CITATION.md` / sources list for every threshold in cricket_injury_recovery_benchmarks.json.
3. Persist input videos' provenance + consent flag for corrected_all_data.
4. Add per-feature provenance (world vs 2D source) into result JSON; warn when 2D fallback is used.
5. Split "demo (synthetic)" vs "research (real)" model badges explicitly in UI.

### P1 (validation study) — the actual research
6. Annotate the 8 real clips: release frame, front-knee/trunk/elbow at release, contact window, ball release angle/speed.
7. Extend evaluation/evaluate.py with feature-level bias/SD/MAE/RMSE/R2/ICC/Bland-Altman + per-bowler aggregation (design in Section 7).
8. Collect 100-300 delivery dataset (>=10 bowlers, ESPNcricinfo-style metadata) with release-frame labels; add release-speed radar if possible.
9. Implement SG smoothing + report SNR; add wrist-index-visibility mask.
10. Add pose repeatability + synthetic stress suites (ablation design in Section 8).

### P2 (claims upgrade) — only after P1 data exists
11. Refit performance model on real feature->(out-of-sample target) or drop; report honest CV (grouped) with CIs.
12. Retrospective injury outcome study on cricket_injury_dataset using non-kinematic features (already grouped-CV clean); then seek real kinematic + outcome linkage.
13. Conformal prediction intervals for all app scores.
14. Calibrate all thresholds to the collected population; replace literature "elite" profile with measured percentiles.

---

## 12. Limitations (explicit)

- No real biomechanical ground truth in repo; every accuracy-style claim is unsupported until P1.
- torch/ultralytics unloadable in this sandbox (WinError 4551 / AppLocker) -> tracker/YOLO live paths not exercised here; fallback paths tested.
- 8 real clips is a tiny sample from (likely) a single camera/scene; distribution unknown.
- MediaPipe world landmarks are an estimate of 3D pose; metric accuracy vs mocap unquantified.
- "Performance score" target is derived from input features (self-referential) -> no predictive validity.
- Ball tracking/wrist-proxy robustness validated only on synthetic strips (gt_clips), not real deliveries.

---

## 13. What must be true before calling this a research-grade deliverable

1. A validation dataset with real ground truth (release, joint angles, contact) and outcome/expert labels.
2. Agreement stats (Section 7) reported with CIs, done per bowler, on held-out clips.
3. All "accuracy"-flavored UI/model claims replaced by validated numbers or honest caveats.
4. Math audit items resolved: hip_rotation naming, angular_velocity label, wrist-index visibility, smoothing, SNR.
5. Reproducibility: pinned env, hash manifests, seeds, artifact provenance end-to-end.
6. An explicit methods section (threshold provenance, annotation protocol) written for any external write-up.