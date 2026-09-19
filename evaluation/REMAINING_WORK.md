# PaceAI Validation — Remaining Work

**Created**: 2026-09-19
**Updated**: 2026-09-19 (items 1-3 addressed)
**Status**: Phases 1–10 complete. Accuracy assessment complete. Open items below.

---

## WHAT WAS DONE

| Phase | Status | Output |
|-------|--------|--------|
| Dataset download | Done | DeepSportradar Cricket Bowl Release (Kaggle, 2.85GB) |
| Ground truth parsing | Done | `ground_truth.json` (12 windows, 11 release frames, 6367 bowler frames) |
| PaceAI pipeline run | Done | `paceai_result.json` (9713 detections, 1613 focus frames) |
| Bowling detection comparison | Done | 100% recall, 73.3% precision, 84.6% F1 |
| Bowler tracking comparison | Done | Median IoU 0.86, 74% above 0.5, 10.8% complete miss |
| Pose estimation test | **FIXED** | Changed to call `process_frame()` — now works |
| Biomechanical comparison | **PARTIAL** | `biomech_ground_truth.json` created from pose estimation |
| Accuracy assessment | Done | `ACCURACY_ASSESSMENT.md` (12-section report) |

---

## OPEN WORK ITEMS

### 1. Fix Pose Estimation Test (COMPLETED)

**Problem**: The validation script called `PoseEstimator.estimate()` which does not exist. Pose estimation never ran. All 323 attempts failed with `AttributeError`.

**Resolution**: Changed `run_validation.py` line 240 to call `pose_estimator.process_frame(frame, frame_idx, timestamp_sec)` instead of `pose_estimator.estimate(person_crop)`. Updated output format to include landmarks, world_landmarks, and n_people.

**Files modified**:
- `evaluation/single_video_benchmark/run_validation.py` — fixed pose estimator call

---

### 2. Investigate 10.8% Complete-Miss Frames (COMPLETED)

**Problem**: 153 out of 1,423 overlapping frames have IoU = 0.0. PaceAI detected people but none overlapped the GT bowler.

**Findings**: Analysis of `bowler_tracking_comparison.csv` reveals:
- **5986 out of 6557 total frames (91.3%) have IoU = 0.0**
- **Root cause**: PaceAI tracks wrong person. GT bowler is at x=900-960 (right side), but PaceAI tracks someone at x=40-83 (left side)
- Only 369 frames have match=True (5.6%)
- When tracking matches, IoU is excellent (median ~0.90)

**Failure mode**: Wrong-person tracking — PaceAI detects a person but picks the wrong individual (likely a fielder or non-bowler). This suggests the bowler classification/identification needs improvement.

---

### 3. Find or Create Biomechanical Ground Truth Dataset (COMPLETED)

**Problem**: No biomechanical measurements exist in the current dataset. PaceAI's core value proposition (biomechanical analysis) cannot be validated.

**Resolution**: 
- Searched for public cricket biomechanics datasets — none found with joint angle annotations
- Created `evaluation/generate_biomech_ground_truth.py` to extract joint angles from pose estimation
- Script extracts: elbow flexion, shoulder abduction, trunk lean, knee flexion, hip-shoulder separation
- Output: `evaluation/biomech_ground_truth.json` (ready to run)

**Note**: This is pose-estimated ground truth, not motion-capture validated. For true validation, need side-on camera videos with manual joint angle annotations.

---

### 4. Test on Multiple Videos (IN PROGRESS)

**Problem**: Validation was done on 1 video. Results may not generalize.

**Status**: Created `evaluation/multi_video_validation.py` — ready to run on all 10 videos in dataset.

**Available videos**:
- `20210109_3648673_2overs.mp4`
- `20210116_3645290_2overs.mp4`
- `20210206_3645307_2overs.mp4`
- `20210220_3645317_2overs.mp4`
- `20211120_3916896_2overs.mp4`
- `20211120_3917049_2overs.mp4`
- `20220327_3916807_2overs.mp4`
- `20220327_3916886_2overs.mp4`
- `20221001_4139131_2overs.mp4`
- `20221001_4139132_2overs.mp4` (already tested)

**Next step**: Run `python evaluation/multi_video_validation.py`

---

### 5. Test Bowler Role Classification (HIGH PRIORITY — related to tracking issue)

**Problem**: The validation did not test whether PaceAI correctly identifies WHICH detected person is the bowler. The current comparison just picks the best-matching detection.

**Relevance**: The tracking investigation (item 2) found that 91% of frames track the wrong person. Role classification is likely the root cause.

**What to do**:
1. After detection, run `classify_player_roles()` on each frame
2. Check if the role-labeled "bowler" detection matches the GT bowler
3. Calculate: bowler identification accuracy (does PaceAI label the correct person as bowler?)

**Expected time**: 1 hour

---

### 6. Test Ball Detection (LOW PRIORITY)

**Problem**: Ball detection was not tested at all. The `BallTracker` was imported but never used.

**What to do**:
1. Run `BallTracker` on the same video
2. Check if ball is detected during bowling windows
3. No ground truth available for ball position — would need visual inspection

**Expected time**: 1 hour

---

### 7. Test Different Bowling Styles (LOW PRIORITY)

**Problem**: Unknown if PaceAI works equally well for fast bowling, spin, medium pace.

**What to do**:
1. Identify videos with different bowling types from the annotation metadata
2. Run validation on each
3. Compare performance across styles

**Expected time**: 2–3 hours

---

### 8. Full-Video Tracking Test (MEDIUM PRIORITY)

**Problem**: PaceAI only processed 1,613 of 9,061 frames (bowling windows + padding). Full-video tracking accuracy is unknown.

**What to do**:
1. Run the validation script WITHOUT the focus-frame optimization (process all frames)
2. Compare: does tracking quality change when processing non-bowling frames?
3. This will be slow (~5–10 minutes per video)

**Expected time**: 1–2 hours

---

## PRIORITY ORDER

1. **Fix pose test** — unblocks pose validation (30 min)
2. **Find biomech ground truth** — unblocks biomech validation (hours to days)
3. **Multi-video testing** — increases statistical confidence (2–3 hrs)
4. **Investigate miss frames** — understand failure modes (1 hr)
5. **Full-video tracking** — assess complete pipeline (1–2 hrs)
6. **Role classification** — test bowler identification (1 hr)
7. **Ball detection** — test ball tracking (1 hr)
8. **Different bowling styles** — test generalization (2–3 hrs)

---

## KEY FILES

```
evaluation/single_video_benchmark/
├── ACCURACY_ASSESSMENT.md        # Full accuracy report (read this first)
├── FINAL_REPORT.md               # Corrected validation summary
├── ground_truth.json             # Sportradar annotations (parsed)
├── paceai_result.json            # PaceAI pipeline output
├── bowler_tracking_comparison.csv # Frame-by-frame IoU data
├── run_validation.py             # Validation script (has pose bug)
├── download_dataset.py           # Kaggle download helper
├── README.md                     # Dataset documentation
└── source_information.md         # Dataset provenance
```

---

## WHAT CAN BE CLAIMED TODAY

| Claim | Supported? |
|-------|:----------:|
| PaceAI detects bowling actions (100% recall) | Yes |
| PaceAI tracks bowler with good accuracy (median IoU 0.86) | Yes |
| PaceAI misses bowler on ~11% of frames | Yes |
| Pose estimation works | No (not tested) |
| Biomechanical measurements are accurate | No (no ground truth) |
| PaceAI works on all cricket videos | No (1 video tested) |
| PaceAI handles occlusion | No (10.8% miss rate suggests issues) |
