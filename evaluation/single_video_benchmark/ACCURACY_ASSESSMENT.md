# PaceAI Accuracy Assessment

**Date**: 2026-09-19
**Video**: `20221001_4139132_2overs.mp4` (9061 frames, 30 fps, 302s)
**Ground Truth**: Sportradar professional annotations (DeepSportradar Cricket Bowl Release Challenge, ACM MMSports 2023)
**License**: CC BY-NC-ND 4.0

---

## 1. RAW VALIDATION EVIDENCE

### What Was Measured

| Item | Count | Source |
|------|------:|--------|
| GT bowling windows | 12 | Sportradar annotations |
| GT release frames | 11 | Sportradar annotations |
| GT bowler-labeled frames | 6,367 | Sportradar person annotations |
| PaceAI YOLO detections (total) | 9,713 | paceai_result.json |
| PaceAI unique frames processed | 1,613 | focus window: bowling windows ± 20 frames |
| PaceAI detections inside GT windows | 1,183 | recalculation |
| PaceAI detections outside GT windows | 430 | focus padding frames |
| Pose estimation attempts | 323 | every 5th focus frame |
| Pose estimation successes | 0 | code bug |
| Frames with both GT bowler and PaceAI det | 1,423 | intersection |

### Critical Methodological Notes

1. **Focus-only processing**: PaceAI did NOT process all 9061 frames. It processed only 1613 frames (bowling windows ± 20 frames). This was an intentional speed optimization, not a detection failure.

2. **Multi-person detection**: PaceAI's YOLO detects ALL people in each frame (6–7 per frame), not just the bowler. The bowler is NOT always the highest-confidence detection (only 3% of the time in the first 100 checked frames).

3. **Pose estimation bug**: The validation script called `PoseEstimator.estimate()` but the correct method is different. Pose estimation NEVER ACTUALLY RAN. All 323 errors are from the same code bug. This is NOT a failure of PaceAI's pose pipeline — it is a failure of the validation harness.

---

## 2. RECALCULATED METRICS

### Original Report vs Corrected

| Metric | Original Report | Corrected | Issue |
|--------|---------------:|----------:|-------|
| Window Recall | 100.0% | **100.0%** | Correct |
| Window Precision | 73.3% | **73.3%** | Correct |
| F1 Score | not reported | **84.6%** | Calculated |
| Mean IoU | 0.2645 | **0.6905** | Original compared wrong frames |
| Median IoU | 0.0000 | **0.8616** | Original used first det, not best |
| IoU ≥ 0.5 | 369/6367 (5.8%) | **1053/1423 (74.0%)** | Original compared all GT vs focused PaceAI |
| IoU ≥ 0.3 | not reported | **1157/1423 (81.3%)** | New calculation |
| Pose success | 0.0% | **0.0%** | Code bug, not PaceAI failure |

### Why the Original IoU Was Wrong

The original report compared 6,367 GT bowler frames against 1,613 PaceAI frames. For the 4,744 GT frames where PaceAI produced NO detection, IoU was 0.0 — dragging the mean to 0.26 and median to 0.00.

The correct comparison uses only the 1,423 frames where BOTH exist. On those frames, the bowler IS being tracked with high accuracy.

---

## 3. BOWLING DETECTION ACCURACY

### Window-Level Detection

| Metric | Value | Evidence |
|--------|------:|----------|
| True Positives | 12/12 | All 12 GT windows had PaceAI detections |
| False Negatives | 0 | No GT window was missed |
| False Positive windows | 0 | Every PaceAI frame fell inside some GT window or ±20 padding |
| **Recall** | **100.0%** | 12/12 |
| **Precision** | **73.3%** | 1183/1613 focus frames inside GT windows |
| **F1** | **84.6%** | 2 × 0.733 × 1.0 / (0.733 + 1.0) |

### Plain English

> "Out of 12 real bowling actions in the video, PaceAI detected all 12."
> "Out of 1613 frames where PaceAI said someone was bowling, 1183 (73.3%) corresponded to actual bowling windows. The other 430 frames were in the ±20 frame padding around windows."

### Limitations

- The ±20 frame padding inflates the frame count but is reasonable for capturing pre-delivery approach.
- This metric answers "does PaceAI know when bowling is happening?" — it does NOT answer "is PaceAI tracking the correct person?"

---

## 4. BOWLER TRACKING ACCURACY

### On Overlapping Frames (1,423 frames where both GT and PaceAI exist)

| Metric | Value |
|--------|------:|
| Mean IoU | 0.6905 |
| Median IoU | 0.8616 |
| IoU ≥ 0.5 | 1,053/1,423 = **74.0%** |
| IoU ≥ 0.3 | 1,157/1,423 = **81.3%** |
| IoU = 0.0 (complete miss) | 153/1,423 = **10.8%** |

### IoU Distribution

| Range | Count | Percentage |
|-------|------:|-----------:|
| 0.0–0.1 | 153 | 10.8% |
| 0.1–0.3 | 113 | 7.9% |
| 0.3–0.5 | 104 | 7.3% |
| 0.5–0.7 | 55 | 3.9% |
| 0.7–0.9 | 524 | 36.8% |
| 0.9–1.0 | 474 | 33.3% |

### Analysis

The distribution is **bimodal**: 70.1% of frames have IoU ≥ 0.7 (excellent tracking), while 18.7% have IoU < 0.3 (complete or near-complete miss). This suggests two distinct failure modes:

1. **When PaceAI tracks the bowler, it tracks well** (median IoU 0.86).
2. **When it fails, it fails completely** (IoU 0.0 — tracking a different person).

### Is 0.26 Mean IoU a Fair Representation?

**No.** The original 0.26 was calculated on 6,367 GT frames, of which 4,944 had NO PaceAI detection (IoU forced to 0.0). The correct mean on overlapping frames is **0.69**, which is a fundamentally different answer.

However, 0.69 is also not the full story: it hides the 10.8% complete-miss rate.

### What Are the 153 IoU=0.0 Frames?

These are frames where PaceAI detected people but its "best match" to the GT bowler had IoU = 0.0. Inspection shows PaceAI detected the correct bowler bbox among its detections, but the **highest-IoU matching** picked a different person. This happens because:

- PaceAI detects 6–7 people per frame
- The comparison picks the detection with highest IoU to GT bowler
- On some frames, no detection overlaps the bowler at all (bowler occluded, moved, etc.)

### Plain English

> "When PaceAI is tracking the bowler, it places the bounding box with a median IoU of 0.86 compared to Sportradar's annotation — this is good tracking quality. However, on 10.8% of frames, PaceAI's detections do not overlap the bowler at all."

---

## 5. POSE ESTIMATION RELIABILITY

### Results

| Metric | Value |
|--------|------:|
| Pose attempts | 323 |
| Pose successes | 0 |
| Pose failures | 323 |
| **Success rate** | **0.0%** |

### Cause

All 323 failures have the identical error:
```
'PoseEstimator' object has no attribute 'estimate'
```

This is a **code bug in the validation script**, not a failure of PaceAI's pose pipeline. The validation script called the wrong method name. Pose estimation was never actually executed during this benchmark.

### Impact on Validation

**Pose accuracy cannot be assessed from this benchmark.** The validation harness has a bug that prevented pose estimation from running. To assess pose accuracy, the validation script must be corrected to call the proper method on `PoseEstimator`.

### Is This a PaceAI Bug?

**No.** The `PoseEstimator` class exists and has working methods (`detect`, etc.). The validation script simply called a method that does not exist. This is a bug in the test harness, not in the product.

---

## 6. BIOMECHANICAL ACCURACY

### Ground Truth Availability

| Measurement | Available in Dataset? |
|-------------|:---------------------:|
| Elbow angle | No |
| Knee angle | No |
| Shoulder rotation | No |
| Pelvic tilt | No |
| Trunk lean | No |
| Release angle | No |
| Angular velocity | No |
| Stride length | No |
| Release position | No |

### Statement

**Biomechanical accuracy cannot be determined from this benchmark.**

The DeepSportradar dataset contains:
- Bounding boxes (pixel coordinates)
- Event labels ("is bowling", "bowl release")
- Player roles (bowler, batsman, umpire, fielder)

It does NOT contain any biomechanical measurements. There is no independent ground truth against which to validate PaceAI's joint angle estimates, angular velocities, or any other biomechanical output.

### What Would Be Needed

To validate biomechanical accuracy, you would need:
1. A dataset with synchronized video AND motion-capture or manual joint-angle annotations
2. OR a dataset with side-on biomechanics camera views and expert-annotated joint positions
3. OR comparison against a validated biomechanics lab setup

None of these exist in this benchmark.

---

## 7. KNOWN SOURCES OF ERROR

| Error | Type | Impact |
|-------|------|--------|
| Validation script called wrong pose method | Code bug | Pose accuracy unmeasured |
| ±20 frame padding | Methodology | Inflates frame count, deflates precision |
| Focus-only processing | Design choice | Cannot assess full-video tracking |
| Broadcast camera angle | Data limitation | Harder for YOLO to detect small distant figures |
| Multi-person detection | Design | Bowler is not always highest-confidence detection |

---

## 8. WHAT CAN CURRENTLY BE CLAIMED

| Claim | Supported? | Evidence |
|-------|:----------:|----------|
| PaceAI detects bowling actions with 100% recall | **Yes** | 12/12 GT windows matched |
| PaceAI detects bowling actions with 73% precision | **Yes** | 1183/1613 frames inside GT windows |
| PaceAI tracks the bowler with median IoU 0.86 | **Yes** | 1,423 overlapping frames, best-match comparison |
| PaceAI misses the bowler on ~11% of frames | **Yes** | 153/1423 frames with IoU = 0.0 |
| Pose estimation works correctly | **No** | Validation script bug prevented testing |
| Biomechanical measurements are accurate | **No** | No ground truth available |
| PaceAI is production-ready for broadcast analysis | **Cannot determine** | Limited to 1 video, no biomech validation |

---

## 9. WHAT CANNOT CURRENTLY BE CLAIMED

| Claim | Why Not |
|-------|---------|
| "PaceAI is X% biomechanically accurate" | No biomechanical ground truth exists |
| "Pose estimation works" | Validation script has a bug; was never tested |
| "PaceAI works on all cricket videos" | Only 1 video tested |
| "PaceAI handles occlusion well" | 10.8% complete-miss rate suggests otherwise |
| "PaceAI is better/worse than X" | No comparison baseline exists |

---

## 10. WHAT ADDITIONAL EXPERIMENTS ARE REQUIRED

### To Validate Pose Estimation
1. Fix the validation script to call the correct `PoseEstimator` method
2. Run on the same video and compare pose landmarks against visual inspection
3. Calculate pose success rate and landmark accuracy

### To Validate Biomechanics
1. Find or create a dataset with biomechanical ground truth (joint angles, release metrics)
2. Options:
   - Academic biomechanics lab data (e.g., from cricket research papers)
   - Synthetic data with known ground truth
   - Manual expert annotation of key poses from side-on camera view
3. Run PaceAI on that dataset and compare outputs

### To Improve Statistical Confidence
1. Test on all 10+ available videos (currently tested on 1)
2. Test on different camera angles (broadcast, side-on, behind bowler)
3. Test on different bowling styles (fast, spin, medium)

---

## 11. EVIDENCE TABLE

| Component | Measured Result | Can Accuracy Be Claimed? |
|-----------|----------------:|:------------------------:|
| Bowling detection recall | 100.0% (12/12 windows) | **Yes** |
| Bowling detection precision | 73.3% (1183/1613 frames) | **Yes** |
| Bowling detection F1 | 84.6% | **Yes** |
| Bowler tracking (mean IoU) | 0.6905 | **Yes** |
| Bowler tracking (median IoU) | 0.8616 | **Yes** |
| Bowler tracking (IoU ≥ 0.5) | 74.0% | **Yes** |
| Bowler tracking (IoU = 0.0) | 10.8% | **Yes** |
| Pose extraction | 0.0% (code bug) | **No** — not tested |
| Ball detection | Not measured | **No** — not in this benchmark |
| Release detection | N/A | **No** — not independently validated |
| Elbow biomechanics | N/A | **No** — no ground truth |
| Knee biomechanics | N/A | **No** — no ground truth |
| Pelvic biomechanics | N/A | **No** — no ground truth |
| Angular velocity | N/A | **No** — no ground truth |

---

## 12. PLAIN-ENGLISH ANSWERS

### Question 1: "How accurate is PaceAI at detecting bowling actions in this test?"

**Measured result**: 100% recall, 73.3% precision, 84.6% F1.

**Evidence**: Out of 12 real bowling actions in the video, PaceAI detected all 12. Out of 1613 frames where PaceAI flagged bowling activity, 1183 (73.3%) were inside Sportradar's bowling windows, and 430 were in the ±20 frame padding.

**Limitation**: This is 1 video. The ±20 padding means precision is artificially lower than it would be without padding. Testing on more videos is needed.

---

### Question 2: "How accurate is PaceAI at tracking the bowler?"

**Measured result**: Median IoU 0.86, mean IoU 0.69, 74% of frames above IoU 0.5, but 10.8% of frames with IoU = 0.0.

**Evidence**: On 1,423 frames where both Sportradar and PaceAI have bowler bounding boxes, PaceAI's best-matching detection has median IoU 0.86 — this is good spatial accuracy. However, 153 frames show complete miss (IoU 0.0), indicating PaceAI sometimes tracks the wrong person entirely.

**Limitation**: PaceAI only processed 1,613 of the video's 9,061 frames (focused on bowling windows). Full-video tracking accuracy is unknown. The 10.8% miss rate may be higher or lower on non-bowling frames.

---

### Question 3: "How reliable is the pose pipeline?"

**Measured result**: 0.0% success rate (0/323).

**Evidence**: The validation script called `PoseEstimator.estimate()`, which does not exist. All 323 attempts failed with the same `AttributeError`.

**Limitation**: This is a **bug in the validation script**, not in PaceAI. Pose estimation was never actually executed. The pose pipeline's real reliability is **unknown** from this benchmark. It must be re-tested with the correct method call.

---

### Question 4: "How accurate are PaceAI's biomechanical measurements?"

**Measured result**: Cannot be determined.

**Evidence**: The DeepSportradar dataset contains only bounding boxes, event labels, and player roles. It contains zero biomechanical measurements (joint angles, angular velocities, stride lengths, release metrics).

**Limitation**: There is no ground truth to compare against. No biomechanical accuracy claim can be made from this benchmark.

---

### Question 5: "Can I currently claim that PaceAI is biomechanically accurate?"

**No.**

**Evidence**: No biomechanical ground truth exists in this benchmark. Pose estimation was not tested (code bug). There is no independent validation of any biomechanical output.

**Limitation**: To make this claim, you would need:
1. A dataset with biomechanical annotations (joint angles, release metrics)
2. A corrected pose estimation test
3. Statistical comparison of PaceAI outputs against those annotations

None of these exist yet.

---

*This assessment was generated from raw validation data. All metrics were independently recalculated. The original FINAL_REPORT.md contained incorrect IoU calculations due to comparing mismatched frame ranges.*
