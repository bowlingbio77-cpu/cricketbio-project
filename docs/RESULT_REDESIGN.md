# PaceAI Result Experience — Audit & Redesign Specification

Status: design spec (Phases 1–5, 13–15). Implementation + test results appended in §7–§9.

---

## 0. The finding that determines the whole design

**`app.py:1308-1312` throws away the real `AnalysisResult` and renders a different, poorer one.**

```
analyze_video(run_ml=False)  ->  full result (30+ fields, CV artefacts, provenance, identity)
        |  app.py:1308-1312  return value DISCARDED
        v
~20 scalars copied field-by-field into st.session_state (app.py:469-498)
        |
        v
analyze_feature_vector(...)  ->  a SECOND, fresh result  (app.py:1505)
        v
the result screen renders THIS object
```

Consequences, all verified:

| Field | Appears in UI as | Because |
|---|---|---|
| `bowler_track_id`, `bowler_confidence` | always `None` | read at `app.py:1407-1408` off the ML-only result |
| `bowler_bboxes` | `None` in replay | replay receives the ML-only result |
| `key_moments` / `events` | permanently `[]` | **do not exist on the dataclass at all** (`app.py:481`, `1339-1398` read them via `getattr(..., None) or []`) |
| `feature_provenance` | never shown | only the video result has it; report JSON gets `None` |
| `landmark_source_summary` | empty | same |
| `scoring_blocked_reason` | never displayed | stored in session, put in JSON, no UI reads it — the UI re-derives its own single-reason text |
| `warnings` | session-copied, so OK | the one exception |

So Phase 5's "analysis completeness / feature provenance / landmark source / subject verification" sections
are **not** new features — the data already exists and is already correct. It is currently unreachable.

This is a plumbing fix, not a second pipeline. Fixing it is a prerequisite for Phase 13 compliance
("reuse the existing analysis outputs", "do not create a second source of truth").

---

## 1. CURRENT OUTPUT AUDIT

### 1.1 What the pipeline produces (`src/pipeline.py:26-67`)

37 dataclass fields. Grouped by what they actually *are*:

| Group | Fields | Nature |
|---|---|---|
| **Measurements** | `feature_vector` (10 features) | computed from MediaPipe landmarks |
| **Per-measurement trust** | `feature_provenance`, `landmark_source_summary` | measured vs 2D-fallback vs defaulted |
| **Model outputs** | `performance_score`, `injury_risk{risk_level,risk_index,probabilities}`, `shap_contributions_performance`, `shap_contributions_injury` | synthetic-trained RF |
| **Actions** | `coaching_notes` (`list[str]`) | deterministic threshold rules + model |
| **Validity gates** | `subject_verified`, `delivery_reliable`, `scoring_blocked_reason` | pipeline refusals |
| **Identity** | `bowler_track_id`, `bowler_confidence`, `bowler_confirmed`, `bowler_confirm_reason`, `bowler_candidates`, `identity_switch_count`, `player_roles`, `batting_stances`, `striker_track_id`, `non_striker_track_id` | tracking heuristics |
| **Artefacts** | `video_path`, `pose_video_path`, `reels_video_path`, `analysis_replay_path`, `bowler_bboxes`, `original_frame_dims` | files |
| **Ball** | `ball_stats` (13 keys) | YOLO + Kalman |
| **Diagnostics** | `stage_times`, `stage_backends`, `warnings` | instrumentation |
| **Context** | `bowling_arm`, `camera_view` | inputs (note: `camera_view` is functionally inert — no feature branches on it) |

Not on the dataclass, but computed and rendered: `prediction_interval_performance` (`ml_models.py:369`),
`out_of_distribution_warnings` (`ml_models.py:352`). Both are recomputed locally by the UI from
`feature_vector` + bundle, so any non-UI consumer gets no uncertainty at all.

### 1.2 What is measured vs derived — the trust ladder

This is the backbone of the new information hierarchy. It already exists in code:

| Rung | Signal | Source | How strong |
|---|---|---|---|
| **1. Truly measured** | `feature_provenance[f]["source"] == "world_3d"` | MediaPipe metric 3D landmarks | strongest |
| **2. Degraded** | `== "normalized_2d"` | 2D image coords, perspective-distorted angles | usable with a caveat |
| **3. Not measured** | `== "missing_xyz"` | config default substituted | **not a measurement** |
| — | `["confidence"]` ∈ high/medium/low + `mean_landmark_visibility` | visibility heuristic | coarse band, not a calibrated score |
| **4. Deterministic rule** | `injury_kb.assess_biomechanical_risks`, `coaching._RULES` | band check on measured features | explainable, but **every threshold is `[Source verification required]` per `CITATION.md:7-12`** |
| **5. Synthetic model** | `performance_score`, `injury_risk`, SHAP | RF trained on labels generated *from the features* | weakest — near-perfect metrics are meaningless |
| **6. Fabricated** | `risk_score` 22/58/88, `"low"` default, `elbow 10.0°`, `knee 0.0°` | hard-coded in app.py | **must be deleted** |

Rung 6 is the Phase 11 violation. Rungs 1–3 are currently invisible to the user, which is why a
`normalized_2d` measurement and a `missing_xyz` default look identical to a clean 3D measurement.

### 1.3 The three unrelated severity vocabularies

1. **ML model** → `low` / `moderate` / `high` (argmax, synthetic-trained)
2. **Knowledge base** → boolean *triggered injury list*, no severity at all
3. **Coaching rules** → two-sided in/out-of-band, no severity at all

These are displayed side by side with no reconciliation (`app.py:1611` vs `1781-1806` vs `1748`).
The new design must not merge them into one number — it must label each as what it is.

---

## 2. SEMANTIC AUDIT — observation / interpretation / action / limitation

| Output | Observation | Interpretation | Action | Hard limitation to preserve |
|---|---|---|---|---|
| `feature_vector` (world_3d) | measured angle/speed | — | compare to band | ±camera perspective, single camera |
| `feature_vector` (normalized_2d) | 2D-derived | — | treat as indicative | degrees are perspective-distorted |
| `feature_vector` (missing_xyz) | **none** | — | — | value is a config default, **not a measurement** |
| threshold trigger | measured value vs published band | "this crossed a reference" | change technique | reference unverified (`CITATION.md`) |
| `coaching_notes` | — | derived | the drill/technique change | model+threshold hybrid, not individualized |
| `performance_score` | — | RF on synthetic labels | — | **demo only; not validated**; 68% PI is a *model* band, not a measurement CI |
| `injury_risk` | — | RF argmax | — | **screening context, never diagnosis** |
| `bowler_confidence` | heuristic score | `1-exp(-3·score)` | — | **not a calibrated probability** |
| ICC elbow screen | measured elbow flexion | vs 15° reference | straighter arm path | **not an official ICC on-field measurement** |
| `ball_stats` | YOLO hits + Kalman fills | coverage % | review the replay | interpolated points are predicted, not seen |
| `stage_times` | wall clock | — | — | implementation metadata |

---

## 3. THE REAL "IMPORTANT RESULT" (Phase 3)

The performance score is **not** the most important result, and neither is the biggest number.

Priority is derived **only** from evidence already in the system, with transparent deterministic rules
(no invented importance score, no new model):

```
1. TRUST      subject_verified / scoring_blocked_reason / delivery_reliable
              + count of low-confidence & missing_xyz features
              + bowler_confirmed / identity_switch_count
              + landmark_source_summary
              + OOD features
2. TRIGGERED  literature threshold triggers (deterministic on measured features)
              + coaching band flags
3. ACTIONS    coaching_notes (all of them, not just [0])
4. MODEL      performance_score, injury_risk  (synthetic → visually subordinate, always labelled)
5. ADVANCED   SHAP, intervals, radar, provenance, timing, backends
```

Rationale: rungs 1–3 of the trust ladder are *evidence*; rungs 4–5 are *interpretation*. The current
UI leads with rung 5. The new UI leads with rung 1, then rung 2.

The logic is a pure function over `AnalysisResult` and is unit-testable — see
`tests/test_result_view.py`.

---

## 4. KEEP / MERGE / DEMOTE / REMOVE

| Output | Verdict | Where it goes |
|---|---|---|
| `subject_verified`, `scoring_blocked_reason`, `delivery_reliable` | **KEEP — raise** | §01 Delivery Status (currently buried / never shown) |
| `feature_provenance`, `landmark_source_summary` | **KEEP — expose** | §05 per-measurement source label; §01 completeness (currently report-JSON only) |
| Literature threshold triggers | **KEEP — raise to headline** | §03 Top Findings + §06 Evidence chain |
| `coaching_notes` (all) | **MERGE** | one Coaching Plan §07; today only `[0]` is shown in 3 places |
| Threshold cards + full benchmark table | **MERGE** | §09 cards; the duplicate dataframe → §10 |
| Replay (`analysis_replay_path`) | **KEEP — promote** | §04, large, above the fold in priority order |
| `performance_score` | **DEMOTE** | §08, small, with model/data context + interval |
| `injury_risk` + `probabilities` | **DEMOTE** | §09 as screening context |
| ICC elbow screen | **MERGE** | one instance in §03/§06; drop the 3 other restatements |
| `render_plain_language_summary` | **REPLACE** | superseded by §02 hero + §03 findings |
| 4 metric cards (perf/risk/ICC/knee) | **REPLACE** | superseded by §01 + §02 + §05 |
| `risk_score` 22/58/88, `"low"` default, `risk_pct` donut | **REMOVE** | fabricated; replaced by real `P(high)` or explicit "withheld" |
| `elbow_flexion_deg` default `10.0` | **REMOVE** | fabricates a *passing* legal screen |
| `knee_flexion_deg` default `0` | **REMOVE** | fabricates a measurement |
| Hard-coded joint-stress % bars (`app.py:1707-1719`) | **REMOVE from primary** | ratios of features with invented denominators; → §10 or drop |
| Static "Corrective Exercise Protocols" blocks (`app.py:1760-1773`) | **DEMOTE** → §10 | not produced by the pipeline for *this* delivery |
| `render_modern_gauge` ×2 on the result screen | **DEMOTE** → §08/§09 | duplicate of the metric cards |
| Radar | **DEMOTE** → §10 | |
| SHAP bars | **DEMOTE** → §10 | top-1 driver already in §03 |
| OOD warnings | **RAISE** | must precede derived interpretation (§04 state) |
| `stage_times` | **DEMOTE** → §10 | already was; keep collapsed |
| `render_model_quality_expander` | **KEEP once** | §10; the other ~10 copies of the disclaimer deleted |
| `feature_provenance` in report JSON | **KEEP** | unchanged |
| History / Save / Export | **KEEP, demote visually** | §11 |
| `key_moments` UI panel | **REMOVE** | field does not exist; permanently blank. (Data path fixed in §0 — if the pipeline ever populates it, it returns.) |
| `src/design_system.py`, `src/ui_components.py` | **LEAVE ALONE** | already dead code; do not widen the blast radius |

Nothing underlying is deleted. Every demoted/removed *presentation* keeps its data in the export JSON.

---

## 5. NEW INFORMATION ARCHITECTURE

```
01  DELIVERY STATUS        can I trust this?      gates everything below
02  WHAT THIS DELIVERY TELLS YOU
      main finding (1 sentence) / why it matters / what to work on
03  TOP FINDINGS           max 3, each: title / observed / why / action
04  ANALYSIS REPLAY        large; evidence for 03
05  KEY MEASUREMENTS       grouped release / lower body / trunk+pelvis / upper body
                           each: value · unit · context · SOURCE TYPE
06  WHY THIS WAS FLAGGED   finding -> measurement -> reference -> interpretation -> action
07  COACHING               priority / what / why / how to practise
08  PERFORMANCE INDICATOR  value + band + model context + interval   (small)
09  RISK INDICATORS        indicator / triggered factors / evidence / reference / limitation
10  ADVANCED ANALYSIS      collapsed: radar, SHAP, provenance, landmark source,
                           OOD detail, literature table, model quality, timing, backends
11  HISTORY / EXPORT       utilities only
```

### Adaptive states (Phase 4)

| State | Trigger (existing data only) | Layout |
|---|---|---|
| `UNRELIABLE` | `subject_verified is False` or `delivery_reliable is False` | §01 dominates; §02–03 reframed as "what could not be measured"; **no scores shown at all** |
| `SCORING_WITHHELD` | `scoring_blocked_reason` set | §08/§09 render an explicit withheld panel, not a zero |
| `OOD` | `out_of_distribution_warnings` non-empty | limitation banner **above** §02, before any derived interpretation |
| `PARTIAL` | any `missing_xyz`, or low-confidence count > 0 | §01 shows partial-completeness line; affected metrics flagged inline in §05 |
| `FLAGGED` | ≥1 threshold trigger | §02 lead = the top trigger; §06 expanded |
| `NORMAL` | none of the above | §02 lead = the strongest positive; coaching becomes refinement |

State is computed by one pure function, `assess_delivery()` in `src/result_view.py`.

### Source-type labels for §05

Derived from `feature_provenance[f]["source"]` — no new metadata invented:

| Label | Condition |
|---|---|
| `MEASURED` | `world_3d` |
| `MEASURED (2D FALLBACK)` | `normalized_2d` |
| `NOT MEASURED — DEFAULT USED` | `missing_xyz` |

---

## 6. COMPONENT STRUCTURE

New file `src/result_view.py` — a **presentation layer only**. It reads `AnalysisResult` (+ the two
bundles) and calls `st.*`. It never computes a new metric, never mutates the result, never imports
the CV pipeline.

```
assess_delivery(result, bundles) -> DeliveryStatus      # pure, testable, no st
build_findings(result, status)    -> [Finding]          # pure, testable, no st
build_evidence(result, findings)  -> [EvidenceChain]    # pure, testable, no st

render_delivery_status(status)                    # 01
render_insight_hero(status, finding)              # 02
render_top_findings(findings)                     # 03
render_replay(...)                                # 04  -> delegates to existing app.render_analysis_replay
render_key_measurements(result, status)           # 05
render_evidence_panel(chains)                     # 06
render_coaching_plan(result, status)              # 07
render_performance_indicator(result, bundle)      # 08
render_risk_indicators(result, status)            # 09
render_advanced(result, bundles, ...)             # 10
render_history_export(result)                     # 11
render_result_page(result, ...)                   # orchestrator, called from app.py
```

Theme tokens per Phase 10, scoped under `.pai-*` so they cannot collide with the existing
`.metric-card` / `.kin-*` / `.lab-*` stylesheets:

```
--pai-bg:#070A0F  --pai-surface:#0D121A  --pai-raised:#141B25
--pai-accent:#20D9FF  --pai-text:#F4F7FA  --pai-muted:#7F8B99
```

`src/design_system.py` and `src/kinetic_ui.py` are **not** modified — the new layer is additive.

---

## 7. FILES CHANGED

(filled in after implementation)

## 8. TEST RESULTS

(filled in after implementation)

## 9. BEFORE → AFTER

(filled in after implementation)
