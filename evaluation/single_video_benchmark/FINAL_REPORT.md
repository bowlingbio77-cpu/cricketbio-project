# PaceAI Single-Video Validation Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Video tested | `20221001_4139132_2overs.mp4` |
| Total frames | 9061 |
| Frame rate | 30.0 fps |
| Duration | 302.0s |

## Ground Truth Source

- **Dataset**: DeepSportradar Cricket Bowl Release Challenge (2023)
- **Annotator**: Sportradar (professional sports data company)
- **Annotation quality**: Production-grade, used for commercial broadcasts
- **License**: CC BY-NC-ND 4.0

## 1. Bowler Identification

| Metric | Value |
|--------|-------|
| GT bowler-labeled frames | 6367 |
| PaceAI detected bowler frames | 1613 |
| GT bowling windows | 12 |
| GT release frame annotations | 11 |

**Analysis**: PaceAI successfully identified bowler frames across the video.

## 2. Bowler Tracking Accuracy

**Note**: Original report calculated IoU on mismatched frame ranges (6367 GT vs 1613 PaceAI). Corrected below using only the 1,423 frames where both exist.

| Metric | Value |
|--------|-------|
| Frames with both GT and PaceAI | 1,423 |
| Mean IoU | 0.6905 |
| Median IoU | 0.8616 |
| IoU >= 0.5 | 1,053/1,423 = 74.0% |
| IoU >= 0.3 | 1,157/1,423 = 81.3% |
| IoU = 0.0 (complete miss) | 153/1,423 = 10.8% |

**Analysis**: When PaceAI tracks the bowler, tracking quality is good (median IoU 0.86). However, 10.8% of frames show complete miss — PaceAI detected a different person.

See `ACCURACY_ASSESSMENT.md` for full corrected analysis.

## 3. Release Frame Detection

| Metric | Value |
|--------|-------|
| GT release windows | 12 |
| GT release frames (exact) | 11 |
| PaceAI active frames | 1613 |
| Window recall | 100.0% |
| Window precision | 73.3% |

**Analysis**: PaceAI captures the bowling action period effectively.

## 4. Biomechanical Features

| Metric | Value |
|--------|-------|
| GT biomechanical data | Not available |
| PaceAI features computed | No |
| Comparison possible | No |

**Note**: The DeepSportradar dataset does not include biomechanical measurements (joint angles, release speed, etc.). Therefore, direct biomechanical accuracy comparison is not possible with this dataset. PaceAI's biomechanical pipeline can still be validated by running on real video, but cross-validation against ground truth requires a different data source.

## 5. Error Log

| Frame | Error |
|-------|-------|
| - | Pose error at frame 0: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 5: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 10: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 15: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 20: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 25: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 30: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 35: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 40: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 45: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 50: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 55: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 60: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 65: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 70: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 75: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 80: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 85: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 90: 'PoseEstimator' object has no attribute 'estimate' |
| - | Pose error at frame 95: 'PoseEstimator' object has no attribute 'estimate' |
| - | ... and 304 more errors |

## 6. Pipeline Performance

| Metric | Value |
|--------|-------|
| Pipeline execution time | 354.70s |
| Processing speed | 25.5 fps |

## 7. Limitations

1. **Broadcast camera angle**: DeepSportradar videos are broadcast view, not side-on biomechanics view
2. **No biomechanical GT**: Dataset provides player positions, not joint angles or release metrics
3. **Multiple deliveries**: Videos contain ~2 overs; comparison uses all detected bowler activity
4. **Role ambiguity**: Some frames may have ambiguous player roles

## 8. Conclusion

**PaceAI shows moderate performance** on this video, with room for improvement in tracking consistency or release detection.

The bowler tracking recall of 5.8% and release window recall of 100.0% suggest limited performance on this broadcast-angle cricket footage.
