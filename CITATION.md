# PaceAI — Source Citations and Provenance

This file documents the scientific basis for every hard-coded threshold, benchmark,
and epidemiology figure used in the project, so that any researcher can audit
whether a number is traceable to literature or is an engineering estimate.

> **Audit status (2026-09-10):** This is a *verification-in-progress* list.
> Entries marked `[Source verification required]` could not be confirmed against
> the cited material during the P0 audit (no internet access in the sandbox);
> a full sweep against primary literature is required before the numbers are used
> for research-grade decisions. No entry here should be read as a clinical
> diagnosis or a validated measurement.

---

## 1. ICC elbow-extension screening reference

| Value | Where it lives | Source |
|---|---|---|
| Elbow extension limit ≤ 15° at the moment of release | `src/config.py::ICC_ELBOW_EXTENSION_LIMIT_DEG = 15.0` | ICC Playing Conditions / Regulations (bowling action law). The 15° tolerance is the widely published legal-threshold figure. [Source verification required] for the exact regulation clause, but the 15° figure is consistent with ICC's universally reported elbow-extension allowance. |

**How it is used:** `<15°` → "within limit" (screening), `>15°` → "above reference"
(screening). The app always labels this a *screening indicator*, never an official
legality ruling, because an official ruling requires lab-grade 3D motion capture per
ICC protocol.

## 2. Biomechanical risk thresholds (`data/cricket_injury_recovery_benchmarks.json`)

The JSON file bundles epidemiology incidence figures and per-injury biomechanical
triggers. Integrated from multiple fast-bowling scientific reviews during
development. Column/metric-level status:

| Metric / trigger | Value | Source status |
|---|---|---|
| Lumbar stress fracture (LBSI/spondylolysis) clinical incidence | 24–55% adults, up to 64% adolescents | Consistent with the widely reported "lumbar spine stress is the most common fast-bowling injury" literature. [Source verification required] for the exact numerators. |
| Shoulder counter-rotation > 30° → lumbar trigger | 30° | [Source verification required] |
| Lateral trunk flexion > 35° → lumbar + abdominal trigger | 35° | [Source verification required] |
| 7-day ball load > 234 balls → lumbar trigger | 234 balls | Tied to the cricket fast-bowling workload literature (spike load ~4–5x chronic). [Source verification required] for the exact 234 figure. |
| Abdominal side-strain incidence | 12–18% | [Source verification required] |
| Angular velocity > 800 °/s → abdominal trigger | 800 °/s | [Source verification required] |
| Hamstring strain incidence | 15–22% | [Source verification required] |
| Stride length > 1.2 × height → hamstring trigger | 1.2× | [Source verification required] |
| Rotator cuff / labral overload incidence | 12–15% | [Source verification required] |
| Shoulder abduction > 190° → RC trigger | 190° | [Source verification required] |
| > 8 continuous overs → RC workload trigger | 8 overs | Common fast-bowling workload guidance; [Source verification required]. |
| Patellar tendinopathy incidence | 10–14% | [Source verification required] |
| Knee angle at FFC < 150° → patellar trigger | 150° | [Source verification required] |
| ACWR sweet spot | 0.8–1.3 | Gabbett TWO (training-load literature); generally consistent with ACWR guidance. The specific 0.8–1.3 band is the commonly-cited "sweet spot" range. [Source verification required]. |
| ACWR > 1.5 → 2.5–3.3× injury multiplier | 1.5 / 2.5–3.3× | The "risk rises sharply for ACWR > 1.5" claim is the canonical well-cited result (e.g., Hulin, Gabbett and co-authors, 2014–2016, Australian football + cricket fast bowlers). [Source verification required] for the exact 2.5–3.3 and 1.5 figures here. |
| 7-day spike (234 balls) → 11× lumbar multiplier | 234 / 11× | [Source verification required] (lumbar-stress-context workload spike figures). |
| < 2 rest days between spells → 2.4× injury rate | 2.4× | "2 days' rest is protective" guidance; [Source verification required] for the exact multiplier. |

## 3. Coaching thresholds (`src/coaching.py::_RULES`)

These are *illustrative* bands, explicitly re-labeled in code as literature-informed
starting points to be recalibrated against a labeled target-population dataset.

| Feature | Band used | Source status |
|---|---|---|
| elbow flexion at release | 0–15° | ICC reference (see §1) |
| trunk lateral lean | 15–40° | Typical-range guidance; [Source verification required] |
| front-knee flexion at bracing | 5–30° | [Source verification required] |
| stride length | 0.6–1.3 × height | Typical-range guidance; [Source verification required] |
| shoulder counter-rotation | 20–60° | [Source verification required] |
| peak angular velocity | 300–1200 °/s | [Source verification required] |
| ground-contact time | 0.08–0.25 s | [Source verification required] |

## 4. Feature ranges in the synthetic demo generator (`src/synthetic_data.py::FEATURE_BOUNDS`)

These low/high bounds mirror the `FEATURE_LABELS` slider ranges in `app.py`.
They are engineering ranges chosen so the demo dataset covers plausible bowling
mechanics — **not** literature-extracted distributions. Do not treat the generated
distribution shape as empirical.

## 5. Demonstration performance-score weighting (`compute_performance_score`)

The 0–100 score combines weighted distances from the `ELITE_BENCHMARK` profile in
`app.py`. The benchmark values (e.g., stride 1.05×height, release angle 78°, FFC
0.11s) are **illustrative coaching targets**, not measured normative data.
[Source verification required] for each individual benchmark value.

## 6. Public datasets

| File | Source | Status |
|---|---|---|
| `data/multimodal_sports_injury_dataset.csv` | Kaggle "multimodal-sports-injury-dataset" (anjalibhegam) | Downloaded artifact; original citation to be added on verification. |
| `data/cricket_injury_dataset.csv` | Referenced as "Cricket Injury Dataset" in `src/cricket_injury_data.py` | Downstream artifact; original source URL not recorded in-repo. [Source verification required]. |
| `data/synthetic_bowling_dataset.csv` | Generated by `src/synthetic_data.py` | First-party demo data (synthetic). |
| `data/gt_clips/` | Generated clips by `scripts/gen_gt_clips.py` | First-party synthetic ball-tracking clips — not human-labeled biomechanics ground truth. |
| `data/cricket_recovery_benchmarks.csv` | Derived from `cricket_injury_recovery_benchmarks.json` | First-party derived table; see §2. |

## 7. Model artifacts

`models/*.joblib` are first-party bundles produced by
`scripts/train_demo_model.py` / `scripts/train_sports_injury_model.py`. Each bundle
carries `data_source` (`synthetic` or `real`), CV metrics, baseline comparison,
training config, and a tagged seed where applicable.

---

## How to update this file

1. Sweep every `[Source verification required]` entry against primary literature
   with full internet access; replace with a proper citation (author, year,
   journal, PMID/DOI) or delete the number.
2. After any change to `data/cricket_injury_recovery_benchmarks.json`,
   `src/coaching.py::_RULES`, `app.py::ELITE_BENCHMARK`, or
   `src/synthetic_data.py::FEATURE_BOUNDS`, update §2–§5 in the same change.