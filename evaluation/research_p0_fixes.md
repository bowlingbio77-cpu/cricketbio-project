# P0 Research-Fixes Report

**Date:** 2026-09-10
**Scope:** P0 only. P1/P2 (ground-truth annotation, 100–300 clip collection,
MAE/RMSE/ICC/Bland-Altman, ablations, robustness, new ML training) were **not**
started, per the stop-condition.

---

## Why P0 mattered

The prior audit (`evaluation/research_readiness_audit.md`, research readiness
**47/100**) found the system soundly engineered but with **terminology and
claims** that could be read as stronger than the evidence supports. The core
problems were honest-under-the-hood ("synthetic demo" warnings existed) but the
words *injury risk*, *ICC legality*, and *performance score* are loaded terms,
and the UI/messages/reports repeated them, implying predictions and legality/
accuracy that the data cannot support. P0 made every user-facing claim match the
actual evidence level.

---

## What changed (P0-1 → P0-11)

| # | Item | What was done | Status |
|---|---|---|---|
| P0-1 | Terminology | ~35+ user-facing strings updated across 12 files (see below). Internal keys preserved. | ✅ |
| P0-2 | Citations | `CITATION.md` created; every threshold marked with hit-traceable status; unverifiable entries flagged `[Source verification required]`. | ✅ |
| P0-3 | Data provenance | `evaluation/provenance_manifest.json` created; `consent_status: unknown` recorded for `corrected_all_data/bowling/*` (2,559 clips) and the two real tabular datasets; no consent invented. | ✅ |
| P0-4 | Feature provenance | Per-feature `feature_provenance` (source + UNCALIBRATED visibility-based confidence) + per-sequence `landmark_source_summary` emitted from `feature_engineering`, threaded through `pipeline.AnalysisResult`, and exported in the JSON report. Manual (slider) entries are tagged `source: manual_entry`. | ✅ |
| P0-5 | DEMO vs RESEARCH separation | Result view now shows model badges (`DEMO MODELS • SYNTHETIC-TRAINED` vs `REAL-DATA-TRAINED MODELS`) and input-mode badges (`REAL VIDEO • FEATURES MEASURED` vs `MANUAL SLIDER INPUT`) with explanatory captions. | ✅ |
| P0-6 | Performance claim audit | "Pace Potential Index" → "Demonstration performance indicator (literature-informed demo score, not a validated measurement)"; coaching notes and report headings reworded; synthetic targets explicitly labeled self-referential demo targets. | ✅ |
| P0-7 | Preserve 67.5→80 | `evaluation/runs/before_summary.csv`, `after_summary.csv`, `validation_scores.json`, `evaluation/validation_report.md` **untouched**. Result is a *plausibility / bug-fix verification* against internal literature-informed target ranges, never described as accuracy. | ✅ |
| P0-8 | GT limitation documented | README "Important caveats" now leads with: no human per-delivery biomechanical ground truth exists; MAE/RMSE/ICC/Bland-Altman cannot yet be computed. | ✅ |
| P0-9 | Math naming | Display labels renamed (internal keys kept): `hip_rotation_deg` → **Pelvic Tilt (from horizontal)** (`app.py` FEATURE_LABELS + radar); `angular_velocity_deg_s` → **Shoulder-Rotation Speed** with honest note that it's a shoulder-rotation rate derived by finite differences (not "arm speed"); provenance notes say exactly what each is. | ✅ |
| P0-10 | Reproducibility | `REPRODUCIBILITY.md` created: verified Python 3.14.0 + pinned deps (numpy 2.5.1 / pandas 3.0.5 / sklearn 1.9.0 / joblib 1.5.3 / xgboost 3.3.0 / catboost 1.2.10 / shap 0.52.0 / numba 0.67.0), global seed `config.RANDOM_STATE=42` use throughout, model `metadata_*.json` per bundle, and an honest AppLocker note (prior `WinError 4551`; torch 2.13.0+cpu and ultralytics 8.4.115 import successfully in this session, but live end-to-end YOLO validation still unexecuted). | ✅ |
| P0-11 | This report | Created. | ✅ |

---

## Terminology mapping (P0-1) — user-facing strings fixed

| Was | Now |
|---|---|
| Injury risk / Injury-risk model / Injury Alert | **Biomechanical risk indicator** / risk-indicator model / biomechanical-risk driver (literature-informed screening; never "prediction of actual injury") |
| ICC legality / LEGAL / EXCEEDS ICC LIMIT | **ICC screening** / WITHIN LIMIT / ABOVE REFERENCE, always with "screening indicator, not an official ICC measurement" |
| Performance Rating / Pace Potential Index / Overall action quality | **Demonstration performance indicator / demo performance score** — "not a validated measurement" |
| Main injury driver | Main risk-indicator driver |
| Clinical incidence / clinical benchmarks tab | Reported incidence / **Literature Risk Thresholds** ("screening only — not a prediction of injury") |
| arm speed (angular velocity) | Shoulder-rotation speed (finite-difference of shoulder rotation) |
| Hip Rotation | Pelvic Tilt (from horizontal) |

Internal identifiers that are **kept** for compatibility (not renamed to avoid
breaking policies/history): `injury_risk`, `risk_level`, `injury_bundle`,
`INJURY_TARGET`, `inj_*` training scripts, `performance_score`,
`shap_contributions_injury`, `icc_legal` history key.

---

## Files affected by P0

| File | Change |
|---|---|
| `app.py` | Terminology, demo/research badges, provenance in JSON report, display-label renames, demoted words; CSS badges added |
| `src/coaching.py` | Rule docstring + driver + performance wording |
| `src/assistant.py` | Report headings/wording |
| `src/pipeline.py` | `feature_provenance` + `landmark_source_summary` in `AnalysisResult` (video + manual modes) |
| `src/feature_engineering.py` | Provenance helpers + per-feature provenance + landmark-source summary |
| `src/synthetic_data.py` | Terminology in docstring/comment |
| `src/explainability.py` | Wording in SHAP docstring |
| `src/injury_knowledge_base.py` | "Shoulder-rotation speed" label |
| `chat_assistant.py` | Prompt + helper wording |
| `src/auth_login.py` | Login subtitle wording |
| `assets/loading_overlay.html` | "Screening Lumbar Shear & Biomechanical Risk..." |
| `README.md` | Pipeline title, retail wording, probe of GT limitation caveat |
| `CITATION.md` | **new** |
| `REPRODUCIBILITY.md` | **new** |
| `evaluation/provenance_manifest.json` | **new** |
| `evaluation/research_p0_fixes.md` | **new** (this file) |
| `tests/test_assistant.py` | 3 assertions re-worded to the corrected terminology (not weakened — same strength of check, new strings) |

---

## Test status

- Before P0 (recorded): `python -m pytest tests/` → **184 passed**.
- After P0 (2026-09-10): `python -m pytest tests/` → **184 passed** (0 failures,
  0 errors).
- No test was weakened. The only edits to tests are 3 assertions in
  `tests/test_assistant.py` updated to assert the newly required wording
  (`WITHIN LIMIT`/`ABOVE REFERENCE` + "screening only..." disclaimer, and the
  `## Performance indicator model (demo)` heading).

---

## Claims audit — what is now claimed vs not claimed

**Now claimed (all true):**
- The system computes 10 biomechanical quantities from MediaPipe pose.
- Performance/risk predictions come from **demo models** trained on
  synthetic/literature-informed data; the performance target is self-referential
  (derived from the features) — metrics measure fit to the generator.
- Elbow extension is a **screening** check against the ≤15° ICC reference value;
  an official legality ruling needs lab-grade 3D motion capture.
- `hip_rotation_deg` is actually **pelvic tilt from horizontal**; the code and
  labels now say so (internal key unchanged).
- No human per-delivery biomechanical ground truth exists yet; MAE/RMSE/ICC/
  Bland-Altman are **not computed**.

**Not claimed (deliberately):** no accuracy/agreement numbers on real clips, no
official legality or clinical diagnosis, no predictions of actual injury, no
claim that 67.5→80.0 is validation.

---

## Remaining blockers (P0 open items)

| Blocker | Why it blocks | P-level |
|---|---|---|
| `consent_status: unknown` on `corrected_all_data/bowling` (2,559 clips), `multimodal_sports_injury_dataset`, `cricket_injury_dataset` | published/commercial use requires consent; research use requires ethical review | P1 (data) |
| No biomechanical ground truth annotations | MAE/RMSE/ICC/Bland-Altman impossible | P1 (annotation) |
| Original citations/URLs for two tabular datasets not recorded in-repo | reproducible provenance | P1 (paperwork) |
| Threshold sweeps in CITATION.md flagged `[Source verification required]` | exact numeric defensibility | P1 (literature) |
| Live end-to-end YOLO/ByteTrack run not yet executed & verified in-shell | no live-detection claim made | P1 (repro) |
| 100–300 delivery corpus + validation framework wiring not done | statistical validation (ICC, Bland-Altman, MAE/RMSE CI) | P1 (collection) |

## P0 stop-condition honored

P1/P2 tasks from the audit roadmap (ground-truth annotation, clip collection
targets, ablation/robustness, ICC/Bland-Altman/MAE-RMSE experiments, re-training
on real labels, clinical validation) were **not** started. No GT was invented, no
accuracy numbers were fabricated or adjusted, no thresholds were moved to make
results look better, and no demo metrics were relabeled as research results.