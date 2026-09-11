# PaceAI P1 — Ground-Truth Annotation Protocol

**Document version:** 1.1

**Research phase:** P1 — Measurement validation (NOT clinical injury prediction)

**Status:** Protocol defined; no real annotations entered yet.

**Working research question:**

> Can consumer-grade monocular side-view cricket bowling video estimate
> meaningful fast-bowling delivery kinematics with sufficient agreement to
> human/reference measurements?

This document is the formal rulebook that a *human* annotator must follow to
turn the 8 real pilot clips into **independently measured reference values**.
It defines what can and cannot be measured from the available footage, how each
measurement is made, which coordinate conventions are used, and how uncertainty
and missing data are recorded.

---

## 1. Scope and degrees of freedom

The current real dataset is **8 clips** at **1280x720**, 25 FPS (one clip at
59.94 FPS), each 2.0–2.5 s long. Only one delivery per clip is expected.

Measurements are restricted to what a trained annotator can *realistically*
establish from ordinary side monocular video:

| Measurement | Annotatable from this footage? | Why |
|---|---|---|
| `release_frame` | **Yes — reliable** | Ball visually leaves the bowling hand. |
| `front_contact_frame` | **Yes — reliable** | Front (non-bowling) foot visibly plants. |
| `front_knee_angle_deg` | **Yes — with uncertainty** | Hip–knee–ankle of the front leg visible at release in most clips; planar estimate. |
| `elbow_flexion_deg` | **Yes — with uncertainty** | Bowling-arm shoulder–elbow–wrist visible at release. |
| `trunk_lean_deg` | **Yes — with uncertainty** | Hip-centre to shoulder-centre vs vertical; sensitive to camera tilt. |
| `stride_length_px` | **Yes — planar only** | Pixel distance between foot-ground contacts; no physical scale. |
| `release_angle_deg` | **Conditional** | Requires ball clearly visible on ≥2–3 frames after release. |
| `release_speed_mps` | **NO** | Requires pixel-per-metre calibration (known dimension in scene). Not established for this dataset. |

> **Rule:** no measurement is entered for a feature the annotator cannot
> establish from the footage. That field is left blank (missing) and the
> visibility/missing-value rule below applies.

---

## 2. Frame and coordinate conventions

- **Frame indexing:** 0-indexed frames **of the preprocessed video**
  (after `src.preprocessing.preprocess_video` — 20 FPS, 640x360 by default),
  matching the coordinate space PaceAI's pipeline runs in.
  The annotation UI shows the preprocessed frame index. To convert to the
  original-clip timestamp, use `frame / preprocessed_fps`.
- **Pixel coordinates:** image coordinates, `x` right, `y` down, origin at
  top-left, measured in the displayed (preprocessed) frame.
- **Angled units:** degrees.
- **Angle convention:**
  - *Knee / elbow flexion:* internal angle at the joint vertex between the two
    adjacent segments. **180° = fully extended**, smaller = more flexed.
    - elbow: apex = elbow, arms = shoulder and wrist.
    - knee: apex = knee, arms = hip and ankle.
  - *Trunk lean:* angle between the segment **shoulder-centre → hip-centre**
    and the **upward vertical** vector `(0, −1)` in image coordinates. **0° =
    upright**, 90° = horizontal trunk, increasing positive = forward lean.
  - *Release angle:* angle of the ball velocity vector (from the last visible
    movement just after release) above/below the horizontal in the image plane,
    positive = downward. Requires a straight segment through ≥2 ball positions.

---

## 3. Per-measurement definition

### 3.1 `release_frame`

- **Definition:** the first frame in which the ball has visibly left the
  bowling hand (no hand-ball contact; wrist/ball separation occurs).
- **Frame-selection rule:** the earliest frame where separation is unambiguous;
  if ambiguous between two adjacent frames, choose the earlier and record
  uncertainty in `notes`.
- **Visibility requirement:** ball and release hand visible on the frames of
  separation.
- **Occlusion handling:** if the ball or hand is occluded across the suspected
  release window, mark `visibility_release=occluded` and leave
  `release_frame` blank.
- **Acceptable uncertainty:** ±1 frame (±50 ms at 20 FPS).
- **Annotator instruction:** step frame-by-frame several times across the
  suspected window; do not skip directly to one frame.

### 3.2 `front_contact_frame`

- **Definition:** first frame in which the front (non-bowling) foot has
  clearly made ground contact at the front-foot landing.
- **Frame-selection rule:** earliest frame showing foot-ground contact with no
  visible further lowering of the foot in the next frame.
- **Visibility requirement:** front foot and ground visible at landing.
- **Occlusion handling:** blank if occluded; set `visibility_release` (shared
  contact-window flag) or note in `notes`.
- **Acceptable uncertainty:** ±2 frames (±100 ms at 20 FPS) — grass/artificial
  surfaces make exact contact harder than release.

### 3.3 `front_knee_angle_deg` (measured at `release_frame`)

- **Definition:** internal angle at the front (non-bowling) leg's knee,
  hip–knee–ankle, measured on the frame identified as `release_frame`.
- **Anatomical points:** front hip, front knee, front ankle (same side as the
  front/non-bowling leg).
- **Frame-selection rule:** use `release_frame`.
- **Coordinate convention:** 2D pixel positions; angle per Section 2.
- **Visibility requirement:** hip, knee, ankle of the front leg all visible and
  approximate occlusion-free in that frame.
- **Occlusion handling:** if any of the three points is occluded or self-shadowed,
  leave blank and set `visibility_knee=occluded`.
- **Missing-value rule:** blank (never a filled "guessed" number).
- **Acceptable uncertainty:** ±5° (planar estimate; perspective foreshortening
  is not corrected).
- **Annotator instruction:** click the three points explicitly on the frame; do
  not eyeball the number.

### 3.4 `elbow_flexion_deg` (measured at `release_frame`)

- **Definition:** internal angle at the bowling-arm elbow,
  shoulder–elbow–wrist, on `release_frame`. 180° = straight arm (typical
  near-release posture is 150–170°).
- **Anatomical points:** bowling shoulder, bowling elbow, bowling wrist.
- **Frame-selection rule:** use `release_frame`.
- **Visibility requirement:** bowling shoulder, elbow, wrist visible in that
  frame. In several pilot clips the bowling elbow/wrist visibility is low
  (e.g. pose-completeness fraction < 0.2) — blank if not clearly visible.
- **Occlusion handling:** blank + `visibility_elbow=occluded` if the wrist or
  elbow is behind the trunk or off-frame.
- **Missing-value rule:** blank.
- **Acceptable uncertainty:** ±5°.
- **Annotator instruction:** click shoulder → elbow → wrist in that order.

### 3.5 `trunk_lean_deg` (measured at `release_frame`)

- **Definition:** forward deviation of the trunk from vertical, computed from
  shoulder-centre and hip-centre on `release_frame` (see Section 2).
- **Anatomical points:** centre of the two shoulders, centre of the two hips.
- **Frame-selection rule:** use `release_frame`.
- **Visibility requirement:** both shoulders and both hips visible.
- **Occlusion handling:** if one side is fully occluded, blank +
  `visibility_trunk=occluded` (no single-side extrapolation without documented
  justification).
- **Missing-value rule:** blank.
- **Acceptable uncertainty:** ±6–8° (sensitive to camera tilt / horizon
  assumption). Record horizon/tilt observation in `notes` if relevant.

### 3.6 `stride_length_px`

- **Definition:** planar pixel distance between the front-foot ground-contact
  point and the back-foot ground-contact point at `front_contact_frame`.
- **Anatomical points:** supporting point under the front foot, supporting
  point under the back foot.
- **Frame-selection rule:** use `front_contact_frame`.
- **Coordinate convention:** absolute pixel distance (no physical scale).
- **Visibility requirement:** both feet visible and on the ground in that frame.
- **Occlusion handling / note:** because PaceAI's predicative `stride_length`
  is *normalized* (not pixel-based), a pixel-value comparison is **not directly
  validatable**; the evaluator therefore excludes `stride_length_px` from
  feature agreement until a common normalized unit is agreed in the protocol.
- **Acceptable uncertainty:** ±8 px.

### 3.7 `release_angle_deg` (conditional)

- **Definition:** ball velocity direction just after release relative to
  horizontal (positive downward) in the image plane.
- **Method:** click the estimated ball centre on ≥2 frames immediately after
  release; the tool computes the segment angle.
- **Visibility requirement:** ball clearly distinguishable from background on
  those frames. In many pilot clips the ball is small/blurred → blank.
- **Missing-value rule:** blank.
- **Acceptable uncertainty:** ±4°.

### 3.8 `release_speed_mps`

- **Not annotatable** from uncalibrated footage. **Blank always** for this
  dataset. Recording fabricates a quantity that cannot be measured here.

---

## 4. Missing-value and consensus rules

- Any field the annotator cannot establish → **blank** (never "0", never a
  guess).
- The matching `visibility_*` field records why: `good`, `partial`, `occluded`,
  `out_of_frame`, `not_established`.
- `annotation_confidence`: integer 1–5 cumulative confidence for the row
  (5 = fully confident each entered value was measurable).
- `notes`: free text (occlusion detail, camera tilt, ambiguity, edge).

---

## 4a. Annotator identity (required)

- Every annotation row **must** carry an explicit `annotator_id` (the tool
  refuses to save without one). This is required for future inter-rater
  reliability (ICC between independent annotators).
- Use a code, e.g. **`ANNOTATOR_A`**, **`ANNOTATOR_B`**. Do not record personal
  names unless the researcher intentionally chooses to.
- Two or more annotators may independently annotate the same clip; each pass is
  a separate row (same `video_id`, same `delivery_id`, different
  `annotator_id`).
- After independent passes, a third party may add a **single adjudicated row**
  with `annotator_id=adjudicator`; the evaluator then uses only adjudicator
  rows as the reference for that clip.

---

## 4b. Data validation (report, never silently fix)

The evaluator (`evaluation/ground_truth_validation.py`) validates every
ground-truth row and **reports** issues; it never edits the CSV:

- `release_frame` / `front_contact_frame` must be non-negative and below the
  clip's known `frame_count` (from `video_inventory.csv`).
- every clip must have positive `fps` metadata.
- numeric measurements within physically meaningful bounds:
  knee/elbow flexion 0–180°, trunk lean 0–90°, release angle 0–90°,
  stride length ≥ 0 px.
- no duplicate rows for the same `(video_id, delivery_id, annotator_id)`.
- `annotator_id` present on every row.
- `visibility_*` values in `{good, partial, occluded, out_of_frame,
  not_established}`; `annotation_confidence` in 1–5.

Invalid values are listed in `evaluation/results/validation_report.md`
(and never "corrected").

---

## 5. Annotator independence

- **Ground truth must be independent of PaceAI.** Annotators annotate from the
  video only. The annotation tool does **not** load or display PaceAI
  predictions (by default or otherwise). Should a future tool ever show
  predictions for convenience, every such overlay must be labelled
  `MODEL PREDICTION — DO NOT USE AS GROUND TRUTH`, must not be displayed during
  primary annotation, and must be independently toggleable.
- Every row stores `annotator_id`. Two (or more) annotators may annotate the
  same clip independently; each gets its own `delivery_id` row.
- **Inter-rater agreement** can then be computed per feature (see Section 6
  of the protocol and `evaluation/ground_truth_validation.py`).
- **Adjudication:** after independent annotation, a third-party (or the study
  lead) may produce a single adjudicated reference value per clip for the
  primary feature comparisons, recorded with `annotator_id=adjudicator`.

---

## 5a. Human video-class review (inventory)

`evaluation/ground_truth/video_inventory.csv` is filled in two stages:

1. **Automatic probing** (`scripts/build_video_inventory.py`) records only
   objective video metadata (fps, frame_count, resolution, duration) and marks
   `view_type` / `usable_for_annotation` as `UNKNOWN` — it never classifies.
2. **Human review** (`scripts/review_video_inventory.py --show` then
   interactive `--all` or by-clip) sets for each clip:

   - `view_type`: `side_view` / `near_side_view` / `other` / `unusable`
   - `usable_for_annotation`: `yes` / `no`
   - `notes` + `review_date`: why the decision was made.

Rule: a clip is only counted as **reviewed** and **usable** after this human
decision exists. Classification is never delegated to PaceAI.

---

## 6. Validation design (held-out; pilot only)

- The 8 real clips are the **initial pilot validation set**.
- Future split strategy, once deliveries are collected:
  - **Development/training set** (never used for final reporting)
  - **Validation set** (model selection)
  - **Held-out test set** (final agreement reporting)
- The pilot set must **not** be used to tune PaceAI; any parameter change that
  arises from studying these clips must be documented and re-validated on a
  separate split.
- Results from the 8 clips are labelled **PILOT VALIDATION** — they are not a
  population-level generalization claim.

---

## 7. Claim-protection language (mandatory)

Any report produced by the validation evaluator must state:

> Ground-truth validation is performed independently of the PaceAI prediction
> pipeline.

> The current 8-clip dataset represents pilot validation and is insufficient by
> itself to establish population-level generalization.

> The validation evaluator is implemented, but no real accuracy statistics are
> reported until independent annotations are available.

The 67.5 → 80.0 value already reported is a **plausibility-vs-published-ranges**
score and a bug-fix verification, **not** measurement accuracy and **not**
validation.

---

## 8. Reproducibility record

Each annotation row records (per ANNOTATION_PROTOCOL_VERSION = 1.1):

| Field | Source |
|---|---|
| `annotation_protocol_version` | stamped on save by `evaluation/annotator.py` |
| `annotation_date` | stamped on save by `evaluation/annotator.py` |
| `annotator_id` | entered by the annotator (required) |
| `video_id` | clip identifier |
| filename, fps, frame_count | `video_inventory.csv` (joined by `video_id`) |
| PaceAI version/commit | `evaluation/provenance_manifest.json` |
| prediction artifact version | `evaluation/predictions/<video_id>_pred.json` (built from `evaluation/runs/after/<clip>.json`) |

Do not record unnecessary personal information.

---

## 9. No model tuning on ground truth

The pilot ground-truth set is a **validation** instrument, not a training set.
No pipeline component (YOLO/ByteTrack/MediaPipe configuration, feature
formulas, release-detection thresholds, biomechanics thresholds, Random Forest,
model weights) may be changed because of ground-truth results. If a genuine bug
is discovered, document it separately and re-validate under an explicit
version scheme, e.g. `VERSION A → VALIDATION`, `VERSION B → CORRECTED
VALIDATION`. Never present a result produced by a corrected model as the
original validation result.