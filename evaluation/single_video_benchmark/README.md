# PaceAI Single-Video Validation Benchmark

## Dataset: DeepSportradar Cricket Bowl Release Challenge (2023)

**Source**: https://github.com/DeepSportradar/cricket-bowl-release-challenge  
**Kaggle**: https://www.kaggle.com/datasets/dzambrano/cricket-bowlrelease-dataset  
**License**: CC BY-NC-ND 4.0 (research purposes only)  
**Annotator**: Sportradar (professional sports data company)

## Why This Dataset

- **Real match footage**: Videos extracted from actual cricket matches (not synthetic)
- **Professional annotations**: Bounding boxes with player role labels (bowler, batsman, etc.)
- **Bowl release events**: Frame-level annotations of bowling action windows
- **Publicly available**: Can be downloaded from Kaggle
- **Academic benchmark**: Used in ACM MMSports 2023 challenge

## What the Annotations Provide

Per frame:
- `person`: Bounding boxes of all detected players with role labels
  - `"person_role"`: "bowler", "batsman", "wicketkeeper", "umpire", "fielder"
  - `"bounding_box"`: `[x1, y1, x2, y2]` pixel coordinates
- `event`: Bowling action events with start/end frame indices

## Setup

### 1. Get Kaggle API Key

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`

### 2. Download Dataset

```bash
python evaluation/single_video_benchmark/download_dataset.py
```

### 3. Run Validation

```bash
python evaluation/single_video_benchmark/run_validation.py
```

## Files

| File | Purpose |
|------|---------|
| `download_dataset.py` | Downloads the dataset from Kaggle |
| `run_validation.py` | Runs PaceAI on selected video and compares |
| `ground_truth.json` | Ground-truth annotations for the selected video |
| `paceai_result.json` | PaceAI output for the selected video |
| `bowler_tracking_comparison.csv` | Frame-by-frame tracking comparison |
| `FINAL_REPORT.md` | Complete validation report |
| `README.md` | This file |
| `source_information.md` | Detailed dataset source information |
