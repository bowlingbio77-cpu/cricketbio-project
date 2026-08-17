# PaceAI Evaluation Framework

## Status: Evaluation dataset not yet available

No real bowling videos with ground-truth annotations have been collected.
The framework below defines the annotation format, metrics, and evaluation
procedure so that when data becomes available, evaluation can begin immediately.

## Annotation Format

### Frame-level ball bounding boxes (`annotations/<video_id>.csv`)

```
frame_number,x1,y1,x2,y2,confidence
```

- `frame_number`: 0-indexed frame number in the preprocessed video (after FPS reduction and resize)
- `x1,y1,x2,y2`: Top-left and bottom-right pixel coordinates of the ball bounding box
- `confidence`: Annotation confidence (1.0 = certain, lower = uncertain)

### Release frame annotation (in `metadata.csv`)

- `release_frame_annotated`: 0-indexed frame number where the ball is released from the hand

### Pose landmark validation (`annotations/<video_id>_pose.csv`)

```
frame_number,landmark_name,x,y,confidence
```

- `landmark_name`: MediaPipe landmark name (e.g., `right_wrist`, `right_elbow`, `right_shoulder`)
- `x,y`: Pixel coordinates in the preprocessed frame
- `confidence`: Annotator confidence

## Metrics

### 1. Ball Detection mAP50

- IoU threshold: 0.50
- Confidence threshold: configurable (default 0.1, matching production)
- Reports: Precision, Recall, mAP@0.50 per video and averaged

### 2. Tracking Consistency

- Track continuity: fraction of frames with a ball detection/prediction
- ID switches: number of times the winning track changes identity
- Coverage: detected frames / total frames in the delivery window

### 3. Release Frame Accuracy

- Absolute frame error: |predicted_release - annotated_release|
- Reports: mean, median, max error across all annotated deliveries
- Equivalent time error: frame_error / FPS

### 4. Pose Accuracy

- Pixel RMSE for wrist, elbow, shoulder landmarks
- Normalized RMSE (divided by image diagonal)
- Reports per-landmark and overall

### 5. Wrist-Proxy Reliability

Quality flags based on:
- HIGH: wrist landmark visibility > 0.8 AND release frame detected reliably
- MEDIUM: wrist landmark visibility 0.5-0.8 OR delivery marked unreliable
- LOW: wrist landmark visibility < 0.5 OR no release frame detected

## Running Evaluation

```bash
cd D:\cricket_biomech_ai
python evaluation/evaluate.py
```

Requires real annotated data in `evaluation/videos/` and `evaluation/annotations/`.
Without data, the script reports **NOT MEASURED** for all metrics.
