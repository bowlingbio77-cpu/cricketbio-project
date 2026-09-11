# Reproducibility

How to reproduce PaceAI results, and what is / is not guaranteed across runs.

## Verified environment (2026-09-10)

| Component | Version | Notes |
|---|---|---|
| OS | Windows (win32), PowerShell 5.1 | Console code page used for tests: UTF-8 via `$env:PYTHONIOENCODING='utf-8'` |
| Python | 3.14.0 | |
| numpy | 2.5.1 | pinned in `requirements.txt` |
| pandas | 3.0.5 | pinned |
| scikit-learn | 1.9.0 | pinned |
| joblib | 1.5.3 | pinned |
| opencv-python-headless | 5.0.0 | pinned (`cv2 5.0.0`) |
| mediapipe | present | `import mediapipe` OK; pose model `models/pose_landmarker_heavy.task` needed for video mode |
| xgboost / catboost / shap | 3.3.0 / 1.2.10 / 0.52.0 | pinned (bundle pickles require matching versions) |
| numba | 0.67.0 | required for shap import on numpy>=2.5 |
| torch | 2.13.0+cpu | importable in this environment; NOT in `requirements.txt` (optional for CNN-LSTM / Transformer sequence models) |
| ultralytics | 8.4.115 | importable; `yolo11n.pt` present in `models/` |

> **AppLocker caveat (updated 2026-09-10):** an earlier session hit `OSError:
> [WinError 4551]` on a torch/ultralytics operation. In this session both
> `import torch` and `import ultralytics` succeed. Live YOLO/ByteTrack end-to-end
> video validation has still **not** been executed/verified from this shell, so
> no live-detection accuracy claims are made.

## Seeds

- `src/config.py::RANDOM_STATE = 42` is the single global seed.
- All sklearn estimators pass `random_state=config.RANDOM_STATE` (see `src/ml_models.py`).
- Sequence models seed torch explicitly (`torch.manual_seed`, generator seed —
  `src/ml_models.py`).
- Synthetic data generators accept `seed=config.RANDOM_STATE` (`src/synthetic_data.py`).
- CV splits use `train_test_split(..., random_state=random_state)` and
  `cross_validate(..., cv=KFold/StratifiedGroupKFold, random_state=...)`.

To reproduce a full re-train with identical random draws, the data files must be
the committed ones (`data/*.csv`) and the versions in `requirements.txt` must hold
(joblib pickles are not portable across sklearn/xgboost/catboost major bumps).

## Model provenance metadata

Every exported bundle is accompanied by a `metadata_*.json` in `models/`
containing `data_source` (`synthetic` / `real`), fold-wise CV scores, baseline
comparison, label map, feature list, and training config. The provenance manifest
(`evaluation/provenance_manifest.json`) maps each artifact to its origin.

## Known non-determinism

- `ball_tracking_v2.*` and pose overlay rendering are deterministic given sorted
  inputs (no RNG); ball detection depends on YOLO weights (`yolo11n.pt`, fixed).
- Excel/pandas `sample` is seeded; SHAP background sampling uses `random_state`.
- Streaming/GPU-dependent stages (YOlo inference) may vary by CUDA/thread schedule
  on non-CPU environments; results here are CPU-only.