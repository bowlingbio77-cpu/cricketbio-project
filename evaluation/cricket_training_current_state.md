# PHASE 0 — Current State Audit: Cricket Understanding Training

**Repository:** `D:\reserve\cricket_biomech_ai` (git HEAD `1781510`)
**Audited:** 2026-09-29
**Auditor scope:** read-only. No behavior was changed during this audit.

> **Path note.** The task brief names `D:\cricket_biomech_ai`. That path is a
> *container* directory holding two things: a `clone/` folder and a
> `cricket_biomech_ai/` folder that is a **different, older copy** of the repo
> (it has `venv312/`, `bowling vids/`, `yolo_test_output/`, and a root-level
> `yolo11n.pt`, and its most recent commit set does not include the
> Streamlit/PaceAI app). The actual active project — the one this session's
> working directory points at, and the one whose git history contains the
> bowler/batsman role-detector work — is `D:\reserve\cricket_biomech_ai`.
> **All findings below refer to `D:\reserve\cricket_biomech_ai`.** The
> 2,573 videos in `D:\cricket_biomech_ai\cricket_biomech_ai\bowling vids` are
> outside this repository and are not counted anywhere in this document.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Does a human-labelled cricket dataset exist? | **No.** `evaluation/ground_truth/ground_truth.csv` is a header with **0 data rows**. |
| How many real videos exist in-repo? | **2,572** (2,559 in `corrected_all_data/bowling/`, 13 in `data/gt_clips/`). |
| How many are human-annotated for cricket understanding? | **0** deliveries, **0** roles, **0** phases. |
| Is the trained role model used at runtime? | **No.** `models/bowler_batsman_yolo11n.pt` is never loaded by `src/`. |
| Is a bowling-phase taxonomy implemented anywhere? | **No.** Only `release_frame_idx` and `front_foot_contact_frame` exist. |
| Is ByteTrack used? | Yes — Ultralytics' built-in `bytetrack.yaml` via `model.track()`. |
| Can the test suite run? | Partially: **244 passed, 6 modules fail to collect** (pre-existing catboost/Py3.14 issue). |
| Can YOLO run in this environment? | **Yes, in the project venv.** `venv\Scripts\python.exe` imports `torch 2.13.0+cpu` and `ultralytics 8.4.115`. A different system interpreter is blocked by AppLocker -- see the W8 correction below. |

**Bottom line:** the repository has a strong *heuristic* cricket pipeline and an
honest, empty ground-truth store. It has **no cricket training data at all**.
Everything in Phases 1–15 must therefore be built so that the *first* real
measurement can be produced, without disturbing the working heuristic path.

---

## 2. Current pipeline (as actually executed)

```
app.py (Streamlit)
  └─ src/pipeline.py:analyze_video()            src/pipeline.py:315
       ├─ src/preprocessing.py                  decode → TARGET_FPS=20, RESIZE_DIM=(640,360)
       ├─ src/pipeline.py:_precheck_cricket()   src/pipeline.py:286
       ├─ src/tracking.py:BowlerTracker         src/tracking.py:45
       │    └─ ultralytics YOLO .track(tracker="bytetrack.yaml")
       ├─ src/tracking.py:select_bowler_track_with_meta()   src/tracking.py:663
       ├─ src/tracking.py:classify_player_roles()          src/tracking.py:877
       ├─ src/tracking.py:classify_batting_stances()       src/tracking.py:952
       ├─ src/pipeline.py:_crop_frames_to_bowler()         src/pipeline.py:90
       ├─ src/pose_estimation.py:PoseEstimator             src/pose_estimation.py:50
       ├─ src/feature_engineering.py:analyze_delivery_phases()  :440
       ├─ src/ml_models.py            (injury / performance)
       ├─ src/explainability.py + src/coaching.py
       └─ src/analysis_replay.py:render_analysis_replay()  src/analysis_replay.py:381
```

### 2.1 Detection

| Property | Value | Source |
|---|---|---|
| Library | Ultralytics YOLO | `src/detection.py:19` |
| Weights | `models/yolo11n.pt` — **stock COCO** | `src/config.py:21` |
| Resolver | configured path → project root → bare `yolo11n.pt` → auto-download | `src/detection.py:25-32` |
| Classes | `classes=[0]` = COCO `person` | `src/detection.py:73`, `src/config.py:23` |
| Confidence | `0.25` | `src/config.py:22` |
| Fallback | `cv2.HOGDescriptor` (people SVM), threshold `2× conf` | `src/detection.py:62-64, 85-95` |
| Fallback is recorded | Yes, `self.fallback_reason` | `src/detection.py:57-60` |

`src/detection.py:97` `select_primary_bowler()` (largest `conf × area`) exists but
is **not** on the live pipeline path — the real selection is in `tracking.py`.

### 2.2 Tracking

| Property | Value | Source |
|---|---|---|
| Tracker | Ultralytics **ByteTrack**, `tracker="bytetrack.yaml"` | `src/config.py:24`, `src/tracking.py:115` |
| Persistence | `persist=True`, `stream=True` | `src/tracking.py:109-131` |
| Track record | `Track(track_id, frames: List[int], bboxes: List[tuple])` | `src/tracking.py:35-42` |
| Fallback | custom greedy IoU tracker, ids from `self._next_id` | `src/tracking.py:78, 152-248` |

**Audit finding:** ByteTrack configuration parameters are **not** set by this
project — the shipped ultralytics defaults are used unmodified. There is no
`track_buffer`, `track_thresh`, or `match_thresh` override anywhere in `src/`.
This is a direct conflict with Phase 5/12, where track continuity on a bowler
with a brief occlusion is a required, measurable property.

### 2.3 Bowler selection (the piece to be learned)

`select_bowler_track_with_meta()` (`src/tracking.py:663`) ranks every track by a
**hand-tuned weighted sum of 12 normalised components**
(`_score_components`, `src/tracking.py:313`; weights `_weighted_score`, `:429`):

| Component | Weight | Constant | Meaning |
|---|---|---|---|
| `motion` | 2.0 | `BOWLER_W_MOTION` :190 | mean step / **own bbox diagonal** per frame |
| `relative` | 2.0 | `BOWLER_W_RELATIVE` :191 | motion after subtracting shared camera drift |
| `active` | 1.5 | `BOWLER_W_ACTIVE` :192 | fraction of steps above `BOWLER_ACTIVE_BODY_MIN=0.05` (:210) |
| `growth` | 1.0 | `BOWLER_W_GROWTH` :193 | monotonic apparent-size change (temporal, not absolute) |
| `span` | 0.5 | `BOWLER_W_SPAN` :194 | temporal coverage of clip |
| `straight` | 0.8 | `BOWLER_W_STRAIGHT` :195 | net/path straightness |
| `range` | 0.3 | `BOWLER_W_RANGE` :196 | spatial coverage / frame diagonal |
| `band` | 0.3 | `BOWLER_W_BAND` :197 | fraction of frames in central 15–85% x-band |
| `decel` | 0.4 | `BOWLER_W_DECEL` :198 | late slowdown (plant) |
| `directed` | 0.6 | `BOWLER_W_DIRECTED` :199 | direction persistence |
| `delivery` | 0.4 | `BOWLER_W_DELIVERY` :200 | late-stride "plant" peak signature |
| `vert` | 0.15 | `BOWLER_W_VERT` :201 | legacy vertical-motion term (near-zero weight) |

Gating constants:

- `BOWLER_MIN_MOTION_FRACTION = 0.08`, `BOWLER_MIN_ACTIVE_FRACTION = 0.05` (:215-216) → returns `(None, None)`; no bowler is invented.
- `BOWLER_CONFIRM_MIN_SCORE = 0.35` (:223) — top-1 must reach this.
- `BOWLER_CONFIRM_MIN_MARGIN = 0.06` (:224) — top-1 must beat top-2 by this.

Identity handling (already close to the Phase 12 target):
- `_stitch_continuations()` (`src/tracking.py:545`) re-attaches continuations of
  the same physical person after an occlusion, gated by an appearance check
  (`_appearance_match`, `:530`, HSV histogram correlation) when a
  `frame_provider` is supplied.
- `count_bowler_identity_switches()` (`:638`) re-scores two half-clip windows and
  counts how often the top candidate differs from the final lock.
- `select_bowler_track_with_meta` never substitutes a different person; the
  docstring at `src/tracking.py:14-17` states this explicitly.

### 2.4 Role classification (heuristic, and it does **not** match the target schema)

Two independent heuristic passes exist, with **inconsistent label vocabularies**:

`classify_player_roles()` — `src/tracking.py:877`, labels at `:761-765`:

```
"batsman", "wicketkeeper", "umpire", "fielder", "unknown"
```

`classify_batting_stances()` — `src/tracking.py:952`, labels at `:948-949`:

```
"striker", "non_striker"
```

Signals used: bbox aspect ratio (`_KEEPER_MAX_ASPECT=1.6`,
`_BATSMAN_MIN_ASPECT=1.4`, `_UMPIRE_MIN_ASPECT=2.0`, :768-770), motion
(`_BATSMAN_MAX_MOTION=0.15`, `_KEEPER_MAX_MOTION=0.20`, :773-774), vertical
centre fraction (`_KEEPER_MIN_Y_FRAC=0.45`, `_BATSMAN_MAX_Y_FRAC=0.55`, :778-779),
relative bbox area.

**Conflict with the new schema.** `cricket_understanding_v1` requires
`BOWLER | STRIKER | NON_STRIKER | WICKETKEEPER | FIELDER | UNKNOWN`. The existing
code has `batsman` (≈ striker) **and** `striker` **and** `umpire` (a role with no
home in the new schema). A single classifier must be introduced; the two
existing passes can remain as a legacy fallback but must not be *trained* data.

### 2.5 Pose extraction

| Property | Value | Source |
|---|---|---|
| Library | MediaPipe Tasks `PoseLandmarker` | `src/pose_estimation.py:17-19` |
| Model | `models/pose_landmarker_heavy.task` (30.7 MB) | `src/config.py:31` |
| Mode | `RunningMode.VIDEO` (timestamp-keyed) | `src/pose_estimation.py:62` |
| Max people | `POSE_MAX_PEOPLE = 2` | `src/config.py:34`, `pose_estimation.py:63` |
| Min det / tracking conf | `0.5 / 0.5` | `src/config.py:32-33` |
| Output | `landmarks (33,4)` = x,y,z,visibility; `world_landmarks (33,3)` | `src/pose_estimation.py:83-88` |
| Person choice | **largest landmark bbox** (`_primary_person`, `:32`) | `src/pose_estimation.py:81` |

**Critical gap for Phase 8.** When MediaPipe returns no pose for a frame,
`process_frame` returns `None` (`src/pose_estimation.py:77-78`) and the frame is
**silently dropped** from the sequence. The *relative* frame index is therefore
lost, and there is no explicit "no pose observed here" record. Phase 8 requires
an explicit missing-value representation; the pose module must be wrapped, not
silently reused as-is.

Landmark visibility *is* present per-landmark (column 3), so low-confidence
observations can be represented without changing MediaPipe.

### 2.6 Delivery / phase detection

The only delivery-timing code is `analyze_delivery_phases()`
(`src/feature_engineering.py:440`), which returns:

```python
{"frame_features", "release_frame_idx", "front_foot_contact_frame",
 "front_leg", "bowling_arm", "reliable", "reliability_reason", "n_frames"}
```

- `front_foot_contact_frame()` — `src/feature_engineering.py:396`
- `find_release_frame()` — `src/feature_engineering.py:406`; bounded to
  `[contact, contact+15]` frames (:460) so a straight-arm follow-through is not
  mistaken for release.
- Quality gate `reliable`: `contact is None`, `release_idx <= 1`,
  `release_idx >= n-2`, or `release_idx - contact > 20` ⇒ unreliable (:465-475).

**There is no phase taxonomy anywhere in the codebase.** No `RUN_UP`, `GATHER`,
`DELIVERY_STRIDE`, or `FOLLOW_THROUGH` label exists. `src/analysis_ui.py` uses
the word `phase` only for UI run state (`"idle"|"running"|"complete"|"error"`,
`:103`). Model C therefore has **no baseline** to improve on.

### 2.7 Replay identity

`render_analysis_replay()` (`src/analysis_replay.py:381`) accepts
`bowler_track_id` (`:386` region) and threads it into
`_draw_bowler_box(..., bowler_track_id=...)` (`:217`) and
`_draw_header(..., bowler_track_id=...)` (`:264`).
`src/pipeline.py:831` passes the locked id. **Replay already reuses the locked
identity** — Phase 12 rule 5 is satisfied today and must be preserved.

---

## 3. Current inputs / outputs

### 3.1 Inputs

| Type | Count on disk | Notes |
|---|---|---|
| Real videos (in-repo) | **2,572** | 2,559 in `corrected_all_data/bowling/`, 13 in `data/gt_clips/` |
| Synthetic videos | (within the 13) | 8 of the 13 `data/gt_clips/` are synthetic stress clips; see `evaluation/provenance_manifest.json:73-80` |
| Pose model | 1 | `models/pose_landmarker_heavy.task` |
| YOLO weights | 2 | `models/yolo11n.pt` (used), `models/bowler_batsman_yolo11n.pt` (**unused by `src/`**) |
| Tabular datasets | 4 CSV | 2 are **synthetic** (`data/synthetic_bowling_dataset.csv` self-referential) |

Video breakdown by camera view prefix (`corrected_all_data/bowling/`):

| Prefix | Files | Container |
|---|---|---|
| `fast_left` | 481 | 481 avi |
| `fast_right` | 525 | 525 avi |
| `leg_left` | 350 | 161 avi / 189 mp4 |
| `leg_right` | 324 | 138 avi / 186 mp4 |
| `off_left` | 443 | 215 avi / 228 mp4 |
| `off_right` | 436 | 208 avi / 228 mp4 |
| **total** | **2,559** | 1,728 avi / 831 mp4 |

**Data-integrity defect:** filenames use two zero-padding widths (2,149 files at
7 digits, 410 at 8), producing **410 ambiguous `video_id`s** — e.g. both
`fast_left_00000001.avi` and `fast_left_0000001.avi` exist and are different
sizes. Any dataset keyed on a filename-derived id will silently collide. This
must be handled in Phase 5/6 by keying on a content hash.

### 3.2 Outputs

`AnalysisResult` (`src/pipeline.py:27`) fields relevant to this project:

| Field | Line | Meaning |
|---|---|---|
| `bowler_track_id` | :45 | locked identity — **already exists** |
| `bowler_confirmed` | :57 | confirmation gate outcome |
| `subject_verified` | :49 | `False` ⇒ ML/coaching refused |
| `player_roles` | :53 | `track_id → {role, confidence, scores}` |
| — | — | **no per-stage confidence breakdown exists** (Phase 13) |

### 3.3 What is NOT produced

- No `role_confidence`, `action_confidence`, `delivery_confidence`,
  `measurement_confidence` (only a single `_confidence(score)` heuristic,
  `src/tracking.py:448`).
- No per-delivery annotation export.
- No cricket-annotated JSON/JSONL at all.

---

## 4. Current weaknesses (evidence-backed)

| # | Severity | Weakness | Evidence |
|---|---|---|---|
| W1 | **Critical** | **Zero human ground truth.** No cricket role, phase, or delivery label exists anywhere. | `evaluation/ground_truth/ground_truth.csv` = 1 header line, 0 rows |
| W2 | **Critical** | **Trained role model is dead weight.** `models/bowler_batsman_yolo11n.pt` is never referenced by `src/`. | written at `scripts/train_bowler_batsman.py:128`; script only *prints* an instruction at :177; runtime uses `config.YOLO_WEIGHTS` (`src/config.py:21`) in `src/detection.py:44`, `src/tracking.py:46`, `src/ball_tracking.py:120`, `src/ball_tracking_v2.py:388` |
| W3 | **Critical** | **No phase taxonomy.** Model C has no baseline and no label source. | `src/feature_engineering.py:440-486` returns only release + contact |
| W4 | **Critical** | **All existing detection "labels" are model-derived.** `data/bowler_batsman_dataset/auto_labeled/` (558), `data/cricket_ball_dataset/auto_labeled/` (1,254) and both `yolo_dataset/` trees were produced by OpenCV heuristics + a YOLO model, not a human. | `scripts/prepare_bowler_batsman_dataset.py`, `scripts/auto_label.py`; no `annotator_id` in either metadata file |
| W5 | **High** | **Heuristic thresholds were tuned against the same clips they are reported on.** `training_summary.json` reports `precision 0.0067` with `recall 1.0` on a 119/27 split that does not match the 387/171 split on disk. | `data/bowler_batsman_dataset/training_summary.json:2-14` vs on-disk counts |
| W6 | **High** | **410 ambiguous video ids** (dual zero-padding). | `corrected_all_data/bowling/*` |
| W7 | **High** | **Stale "184 tests passed" claim.** Real state: 244 passed, 6 collection errors. | measured 2026-09-29; errors from `src/ml_models.py:51` `import catboost` on Python 3.14.2 |
| W8 | ~~**High**~~ -> **CORRECTED, see note** | **`torch` is blocked by AppLocker** in this environment, so YOLO/ByteTrack cannot be executed here at all. | **SUPERSEDED.** The block applies to the system interpreter, not the project venv. `venv\Scripts\python.exe -c "import torch, ultralytics"` succeeds (`torch 2.13.0+cpu`, `ultralytics 8.4.115`), and the pipeline runs on real clips with `detection=bytetrack` / `tracking=bytetrack` on 8/8 clips. Use the venv interpreter for all work. |
| W9 | **High** | **Ball YOLO `data.yaml` points at the wrong project root** (`D:\cricket_biomech_ai\...` instead of `D:\reserve\cricket_biomech_ai\...`). | `data/cricket_ball_dataset/yolo_dataset/data.yaml:1-2` |
| W10 | **Medium** | **Pose frames are silently dropped** when MediaPipe finds no person. | `src/pose_estimation.py:77-78` |
| W11 | **Medium** | **Primary-person selection is "largest bbox", not the locked track.** Only safe because crops are pre-applied. | `src/pose_estimation.py:32-47` |
| W12 | **Medium** | **Two inconsistent role vocabularies** (`batsman`/`striker`/`umpire` vs the new 6-class set). | `src/tracking.py:761-765` vs `:948-949` |
| W13 | **Medium** | **Single collapsed confidence number.** | `src/tracking.py:448` |
| W14 | **Medium** | **ByteTrack params are ultralytics defaults**, not tuned or logged. | no override in `src/` |
| W15 | **Low** | Doc/code contradiction on detection threshold (doc says 0.4, code is 0.25). | `PACEAI_COMPLETE_SYSTEM_AUDIT.md:255` vs `src/config.py:22` |
| W16 | **Low** | Single-video Sportradar benchmark is **not reproducible** (source videos absent). | `evaluation/REMAINING_WORK.md:72-81`; `evaluation/single_video_benchmark/` has derived artifacts only |

### 4.1 What the repo gets *right* (must be preserved)

- `evaluation/annotator.py` is a genuine **blind** annotation GUI: it never reads
  `evaluation/predictions/`, requires `--annotator_id` (`:436`, `:455-456`), and
  upserts on `(video_id, annotator_id, delivery_id)` (`:115-166`) so multiple
  independent raters are supported. **Phase 3 should follow this discipline.**
- `evaluation/results/*` and `agreement_metrics.json` report **honest emptiness**
  (`"paired_samples_available": false`) rather than invented numbers.
- `ground_truth_validation.py` implements ICC(2,1) / Bland–Altman with explicit
  minimum-sample gates and refuses to emit statistics below them.
- The bowler-lock discipline the brief asks for in Phase 12 is **already largely
  implemented** (`src/tracking.py:14-17`, `:545`, `:663`; `src/pipeline.py:501`,
  `:620-629`).

---

## 5. Reusable components (do not rewrite)

| Component | Location | Reuse for |
|---|---|---|
| `BowlerTracker` / ByteTrack wrapper | `src/tracking.py:45` | Phase 5/7 track extraction, Phase 12 candidate generation |
| `Track` dataclass | `src/tracking.py:35` | Phase 2 `track_id` semantics, Phase 5 leakage checks |
| `select_bowler_track_with_meta` + `candidates` + `identity_switch_count` | `src/tracking.py:663` | **Phase 10 baseline**, Phase 11 experiment A |
| `count_bowler_identity_switches` | `src/tracking.py:638` | Phase 10 metric 6, Phase 14 tests |
| `score_bowler_tracks` (ranked components) | `src/tracking.py:586` | Phase 11 experiment B features |
| `PoseEstimator` | `src/pose_estimation.py:50` | **Phase 8** (wrap, do not modify) |
| `analyze_delivery_phases` | `src/feature_engineering.py:440` | Phase 10 release-frame baseline |
| `render_analysis_replay` | `src/analysis_replay.py:381` | Phase 12 rule 5 |
| `AnalysisResult.bowler_track_id` / `.bowler_confirmed` | `src/pipeline.py:45,57` | Phase 12 lock carrier |
| Blind annotator CLI/GUI pattern | `evaluation/annotator.py` | Phase 3 GUI design |
| CSV upsert + adjudicator pattern | `evaluation/annotator.py:115-166`, `ground_truth_validation.py:367-377` | Phase 2 multi-rater schema |
| Ultralytics training scaffold | `scripts/train_yolo.py:168`, `scripts/train_bowler_batsman.py:103` | Phase 9 dry-run harness (no training yet) |

---

## 6. Conflicts with the new training architecture

| ID | Conflict | Resolution (do not break existing code) |
|---|---|---|
| C1 | Role vocabularies differ (`batsman`/`umpire` vs `STRIKER`/`NON_STRIKER`). | New 6-class vocabulary lives **only** in `src/cricket_understanding/`. Existing `classify_player_roles` stays as the legacy fallback until Model B is validated. |
| C2 | No phase taxonomy exists. | Phase vocabulary is new. Model C has **no baseline** — Phase 10 records it as `NOT MEASURED`. |
| C3 | ~~`torch` blocked here ⇒ no YOLO/ByteTrack execution~~ **RESOLVED** — `torch` and `ultralytics` import fine in `venv\Scripts\python.exe`; the AppLocker block affects a different interpreter. | Generators still support `--dry-run` with no torch so the code stays runnable elsewhere. **Model A is runnable here** with the venv interpreter. |
| C4 | 6 test modules cannot import (catboost on Py3.14). | New tests must not import `src.pipeline`/`src.ml_models`. Existing failures are recorded, not "fixed" by deletion. |
| C5 | Auto-labelled YOLO datasets are model-derived. | They must **not** be loaded as ground truth. New generators read only `data/cricket_understanding/annotations/`. |
| C6 | 410 ambiguous `video_id`s. | New `video_id` = `sha1(file bytes)[:12]` + stored filename. Never derive id from the stem. |
| C7 | Existing `data/` trees are large and pre-existing. | New data lives in `data/cricket_understanding/`; videos are **referenced by path**, never copied. |
| C8 | Existing `config.YOLO_WEIGHTS` is global. | New weights are resolved through a new `src/cricket_understanding/` settings module; `src/config.py` is left untouched. |
| C9 | `_primary_person` picks the largest pose, not the locked track. | Phase 8 wraps the estimator and records which crop/subject source was used per frame. |
| C10 | Test suite takes ~7 minutes. | New tests must be fast and torch-free. |

---

## 7. Phase 0 exit criteria — MET

- [x] Complete repository inspected (177 tracked files; `src/`, `scripts/`, `tests/`, `tools/`, `evaluation/`, `docs/`, `models/`, `data/`).
- [x] Actual models identified from code, not documentation.
- [x] Detection / tracking / pose / bowler-selection / delivery / replay logic mapped to `file:line`.
- [x] Datasets counted (2,572 real videos; **0** human cricket annotations).
- [x] Evaluation code, annotation tools, configs, tests identified and executed.
- [x] No behaviour changed.

### Verified facts that constrain everything downstream

1. **There is no cricket training data.** Nothing may be trained or evaluated until Phase 3 produces human labels.
2. **`models/bowler_batsman_yolo11n.pt` is not in the inference path.**
3. **There is no bowling-phase baseline to beat.**
4. **244 tests pass; 6 modules cannot import (pre-existing).**
5. ~~**torch is blocked in this environment**, so Model A is not runnable here.~~
   **CORRECTED:** `venv\Scripts\python.exe` imports `torch 2.13.0+cpu` and
   `ultralytics 8.4.115`; the pipeline runs on real clips with the ByteTrack
   backend on 8/8 clips tested. Model A is runnable here with the venv
   interpreter. The test-suite state above is also stale — the suite now passes
   in full (574 passed), see `evaluation/cricket_understanding_report.md`.
