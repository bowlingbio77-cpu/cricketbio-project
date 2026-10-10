# Bowler Identity Audit — PHASE 0

**Status:** complete. No source code modified to produce this report.
**Scope:** every code path that decides *who the bowler is*, and every downstream path that re-derives or silently assumes that identity.
**Repository:** `D:\cricket_biomech_ai`, branch `main`, base commit `21fc2da` + uncommitted UI work.
**Test-suite baseline at audit time:** 338 tests across 21 files, all passing. Zero tests open a real video file.

> **Remediation status.** This document is the *pre-change* record and its numbers
> are deliberately left unedited. Work completed since: the validation yardstick was
> rebuilt (§5) and `src/cricket_roles.py` was added as the authoritative role engine
> with 46 tests in `tests/test_cricket_roles.py`. Items 1-2 of §7 are addressed at
> the engine level; wiring the engine into `src/pipeline.py` and closing the pose
> divergence at `src/pose_estimation.py:32-47` remain outstanding. **No accuracy
> improvement has been measured yet** — §1 still describes current shipped behaviour.

---

## 1. Headline finding

The premise of this task is confirmed by measurement, not assumption.

The repository contains one per-frame comparison artifact against human-annotated broadcast
footage: `evaluation/single_video_benchmark/bowler_tracking_comparison.csv`.
Recomputed directly from that file (6,557 rows):

| Metric | Measured value |
|---|---|
| Total comparison rows | 6,557 |
| Rows with **both** a GT box and a predicted box | **1,423** |
| Rows where the predicted track matches the GT bowler (`match == True`) | **369** |
| → match rate over overlapping frames | **25.9 %** |
| → match rate over all rows | **5.63 %** |
| Median IoU, all rows | **0.0000** |
| Rows with IoU exactly 0.0 (all rows) | 5,986 (**91.3 %**) |
| Rows with IoU exactly 0.0 (of the 1,423 overlapping) | 852 (**59.9 %**) |
| Median IoU *on matched frames only* | **0.9117** |

The distribution is **bimodal, not noisy**. When the system locks onto the right person it is
excellent (median IoU 0.91). When it does not, the overlap is *exactly zero* — the signature of
selecting a wholly different person, not of a loose bounding box. This is a **discrete identity
error**, which is precisely the failure class a role engine with a locked track ID is designed to
remove. Tightening a bounding box cannot fix it.

**Documentation conflict (must be corrected).** `evaluation/single_video_benchmark/FINAL_REPORT.md`
§2 publishes mean IoU 0.6905, median 0.8616, 74.0 % above IoU 0.5, 10.8 % complete miss. **None of
those five numbers is derivable from the committed CSV**, and they contradict §8 of that same file,
which reports 5.8 % recall. `evaluation/REMAINING_WORK.md:17` repeats the incorrect figures while
items 2 and 5 of the same file give CSV-correct values. These claims must be retracted or
re-derived before any accuracy number is published. By contrast, `evaluation/results/validation_report.md`
and `evaluation/provenance_manifest.json` are rigorous and correctly refuse to report accuracy at
all; that is the standard this work should hold to.

---

## 2. Current identity architecture

There is **already** a non-trivial primary selector. This is a hardening and centralisation job,
not a greenfield build.

**Primary selector — `src/tracking.py`**
- A body-normalised, multi-signal candidate scorer (`_weighted_score`) ranks person tracks.
- Produces `bowler_candidates`, `bowler_track_id`, `bowler_confidence`, `bowling_arm`.
- Intended signals: bbox geometry, motion, pose, temporal continuity.

**Confirmation gate — `src/tracking.py` / `src/pipeline.py`**
- Emits `bowler_confirmed`, `bowler_confirm_reason`, `identity_switch_count`, `batting_stances`,
  `striker_track_id`, `non_striker_track_id`, `player_roles`.
- Introduced by commit `e57efde` ("body-normalized bowler selection, confirmation gate, and
  Players Detected roster").

**Manual override — `src/pipeline.py:328, 476-490`**
- `bowler_track_override` is accepted but has **no UI affordance**, so it is unreachable in practice.

**The confirmed weaknesses**
1. Scores are relative to other tracks in the same frame. With one dominant batsman and a small,
   distant bowler, "relatively best" is still the wrong person.
2. Bounding-box geometry is a prior, not evidence of the bowling action. `README.md:64-83`
   already states this: the "largest × most confident" heuristic "does not work for the bowler
   specifically."
3. `docs/RESULT_REDESIGN.md:105` correctly labels `bowler_confidence` a **heuristic score, not a
   calibrated probability**. It must never be presented as confidence in the probabilistic sense.
4. There is no bowling-action evidence. Nothing in the pipeline observes the
   run-up → delivery → release → follow-through sequence that uniquely identifies a bowler.

---

## 3. Accidental second identity decisions

These are the divergence points. Each independently re-derives *who* is being analysed.

| # | Location | Decision | Why it can diverge |
|---|---|---|---|
| 1 | `src/pose_estimation.py:32-47` `_primary_person` | Picks the person with the **largest landmark bounding box**. | **The most serious defect.** Ignores `bowler_track_id` entirely. On broadcast side-on footage the batsman, who is nearest the camera and frequently larger, wins. This is very likely the direct cause of the 59.9 % exact-zero IoU measured in §1. |
| 2 | `src/pipeline.py:577` → `pose_estimation.process_frame` | Passes no subject constraint. | Because of #1, the pose target is chosen independently of the primary bowler decision. Identity can be `LOCKED` while the pose is extracted from a different human. |
| 3 | `src/result_view.py:445-506` | Gates the TRUST state on `subject_verified` **alone**; never reads `bowler_confirmed` or `identity_switch_count`. | `docs/RESULT_REDESIGN.md:119-123` explicitly requires both on the top trust rung. A run whose bowler was *rejected* by the pipeline still renders as trusted. Two already-populated fields would close this; no new API needed. |
| 4 | `src/pipeline.py:643-647` | `all_frames` fallback path. | Silently inherits `subject_verified = True` set at `:618` instead of failing safe. An empty/absent confirmation degrades into a trusted analysis. |
| 5 | `app.py:1377` | `getattr(result, "bowler_role", None) or "bowler"` | `AnalysisResult` has no `bowler_role` field, so this is a hardcoded constant. A wrong-person result is still labelled "bowler" with full confidence. |
| 6 | `evaluation/single_video_benchmark/run_validation.py:155-156` | Ground-truth bowler chosen by `if "bowler" in raw_lower` — substring match on free text. | Defines the yardstick for every reported recall number. Also matches `non_striker`-style strings; a label-parse slip silently caps published accuracy. |
| 7 | `app.py:1462-1495` `_merge_video_result` | `bowler_candidates` is dropped when `== []` (`:1490-1491`). | A rejected run loses its candidate list entirely, so a future manual picker would have nothing to offer in video mode. |

**Propagation note.** `bowler_confirmed`, `bowler_candidates`, `bowler_confirm_reason`,
`identity_switch_count`, `batting_stances`, `striker_track_id`, `non_striker_track_id` are
*propagated* onto the analysis result by `_merge_video_result`, but **no code branches on any of
them** except at the producer in `src/pipeline.py`. Confirmed by repo-wide grep: the only other
occurrences are the mirrored field-name lists in `tools/verify_video_path.py:21-23`,
`tests/test_result_view.py:24-26`, and `docs/RESULT_REDESIGN.md`. The gate exists but is inert.

---

## 4. Root cause of the measured failure

Ranked by expected contribution:

1. **Pose is extracted from the wrong person** (#1, #2). Even a correct primary bowler ID cannot
   save the biomechanics, because the pose engine re-chooses by pixel area. Given that matched
   frames score IoU 0.91 while mismatched frames score exactly 0.00, the system is behaving as two
   distinct identities rather than one sloppy one.
2. **No bowling-action evidence.** Selection rests on geometry and proximity, so a large,
   confident, near-camera batsman outranks a small, distant, side-on bowler — the default
   configuration of broadcast cricket footage, which is what the benchmark uses.
3. **The confirmation gate is not enforced** (#3, #4). The system's own safety net does not gate
   the user-facing output.
4. **No identity lock.** Nothing prevents a mid-delivery switch to another track, and
   `identity_switch_count` is recorded but never checked.

---

## 5. Validation reality check

This section is included because it determines what may honestly be claimed.

**Usable ground truth for bowler identity: yes — 26 annotated broadcast videos.**
`evaluation/single_video_benchmark/dataset/data/<video_id>_2overs.json` contains per-frame
`person` annotations with `{id, role, box}` and an independent `event` stream whose
`event == "is bowling"` frames delimit each delivery. This is human annotation and is
sufficient to measure bowler-identification accuracy directly.

**Corrected during this audit.** The previously extracted footage covered only 3 of the 10
annotated videos. The full archive `dataset/cricket-bowlrelease-dataset.zip` (3.06 GB) contains
**40 videos**, and 23 further videos matching the 26 annotation files were extracted, giving a
**26-video paired validation set**. The remaining 7 videos have no annotations and cannot be used.

`evaluation/build_bowler_gt.py` rebuilds ground truth from the raw JSON, replacing the substring
yardstick. Two measured properties:

| Property | Whole-video `DOMINANT_ID` | `PER_DELIVERY` (adopted) |
|---|---:|---:|
| Bowler-role frames scored | 82,718 | 23,482 |
| Frames excluded as fragmented | 142,493 | **674** |
| Distinct bowlers resolved per video | 1 | 3–10 |

Whole-video dominant identity fails because a video contains **two overs bowled by two different
players**, so no single id can be right for the whole clip. Resolving identity *inside each
`is bowling` window* drops fragmentation from 142,493 frames to 674 (2.8% of the window), at the
cost of scoring only the action window rather than every frame the bowler is visible. Scoring the
action window is the semantically correct unit for "did the system pick the bowler during his
delivery"; the fraction of all bowler-role frames this represents (~10%) is a sampling rate, not a
defect, because the remainder covers the bowler standing at the mark or walking back.

Result: **26 videos, 282 deliveries, 23,482 scored frames, 674 excluded** — a conservative
denominator in which excluded frames are never silently scored as misses.

`tests/test_bowler_gt.py` (18 tests) guards this yardstick, including a regression test that a
`non_striker` label can never satisfy a bowler query — the exact defect of the old substring test.

Residual limitations, stated rather than hidden:
- Annotation identity still fragments within a delivery window; 674 frames are excluded rather than
  scored, which lowers the measured denominator and makes the resulting accuracy conservative.
- Videos are one competition format (Sportradar 2-over broadcast). No cross-format generalisation
  can be claimed from them.

**No other ground truth exists.**
- `evaluation/ground_truth/ground_truth.csv` is a 316-byte header. `evaluation/metadata.csv` is an
  85-byte header. `annotation_status.md` records Annotated: **0**.
- There are **zero** human per-delivery biomechanical annotations anywhere. `evaluation/biomech_ground_truth.json`
  does not exist despite `REMAINING_WORK.md` item 3 being marked COMPLETE.
- `data/cricket_ball_dataset` labels are generated by `scripts/auto_label.py`, which runs the YOLO
  detector itself. Evaluating a detector on its own output is circular. `provenance_manifest.json`
  currently flags this dataset `research_grade_gt: true`, which is **wrong** and must be corrected.
- The 12 real `data/gt_clips/*` directories are self-flagged `DRAFT_UNVERIFIED.txt`, are tracker
  output rather than human annotation, and **contain no video file**.

**Therefore:** bowler *identification* accuracy can be measured for real against 26 videos. 
Downstream biomechanical accuracy cannot, and must not be reported, until human labels exist.

---

## 6. Additional data-integrity defect

`corrected_all_data/bowling/` contains **410 colliding filenames** caused by two zero-pad widths
for the same clip number:

```
fast_left_00000001.avi  (1,254,472 B)   vs   fast_left_0000001.avi  (1,022,478 B)
```

All 410 pairs were MD5-compared: **0 are byte-identical** — they are different clips sharing an
ambiguous `video_id`. 820 files (32 % of the corpus) therefore have a non-unique natural key, and
any join on `video_id` across ground truth, predictions, and inventory is unsafe. This must be
resolved before any dataset-scale metric is published.

---

## 7. Scope of the required change

Ordered by value per unit of risk. Phases 1-12 of the task specification address items 1-4 below;
items 5-6 are prerequisites for trustworthy reporting.

1. **`CricketRoleEngine` in `src/cricket_roles.py` as the single source of truth.** Every heuristic
   in `src/tracking.py` that ranks person identity moves behind this interface. Downstream modules
   consume a locked `bowler_track_id` and must not re-derive identity.
2. **Close the pose divergence (#1, #2).** Add an explicit subject/ROI argument to
   `pose_estimation.process_frame` and pass the locked bowler bbox from `src/pipeline.py:577`, so
   `_primary_person` degrades to a tie-break *inside the bowler's own region* rather than a
   global decision.
3. **Enforce the existing gate (#3, #4).** Add `bowler_confirmed` and `identity_switch_count` to the
   TRUST decision in `src/result_view.py`; make the `all_frames` fallback fail safe. These two
   fixes require no new fields and close a documented spec violation.
4. **Add real bowling-action evidence** (run-up → approach → gather → delivery stride → release →
   follow-through) as the dominant identity signal, with per-phase interpretable contributions.
   The lower-half prior (`LOWER_HALF_THRESHOLD = 0.50`) must be a weak temporal prior only and
   never an assignment rule.
5. **Rebuild ground truth from raw Sportradar JSON** with an explicit, documented dominant-bowler-ID
   policy, replacing the `"bowler" in raw_lower` substring test.
6. **Retract the unreproducible numbers** in `FINAL_REPORT.md` §2 and `REMAINING_WORK.md:17`, and
   correct the `research_grade_gt` flag in `provenance_manifest.json`.

Convert `scripts/debug_bowler_selection.py` into asserting tests before any threshold is changed.
It already encodes the right adversarial cases (broadcast side view, occlusion fragmentation,
static large distractor) but `run_case()` only prints, so a regression in `_weighted_score`
would currently be silent.

---

## 8. What must not be claimed

- No accuracy, recall, IoU, or confidence figure beyond what a committed, reproducible artifact supports.
- No biomechanical accuracy claim — no human ground truth exists for it.
- No injury or clinical claim. The `models/injury_*` and `performance_*` bundles are trained on
  synthetic, self-referential targets; `models/sports_injury_*` and `cricket_injury_*` use real
  data but are not biomechanics validation.
- No real-world generalisation claim from 10 broadcast videos of one competition format.
- `bowler_confidence` is a heuristic score, not a calibrated probability.
