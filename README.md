# cricketbio

Cricket Bowling Biomechanics AI — an end-to-end implementation of the pipeline:

```
Video → Preprocessing → Bowler Detection (YOLOv11) → Tracking (ByteTrack)
      → Pose Estimation (MediaPipe, 33 landmarks) → Biomechanical Feature
      Engineering → ML (Performance / Injury Risk) → SHAP → Streamlit
      Dashboard → Coaching Recommendations
```

## What's real vs. what's a fallback

This was built in a sandbox with **no internet access**, so it ships with:

| Stage | Library used | Status here |
|---|---|---|
| Preprocessing | OpenCV | ✅ fully working, tested |
| Detection | YOLOv11 (`ultralytics`) | code correct; **needs `pip install ultralytics`** — falls back to an OpenCV HOG person-detector if not installed |
| Tracking | ByteTrack (via `ultralytics`) | code correct; falls back to a built-in greedy IoU tracker |
| Pose estimation | MediaPipe Tasks `PoseLandmarker` | ✅ library installed & API verified; **needs one model file download** (see below) |
| Feature engineering | NumPy | ✅ fully working, unit-tested with synthetic landmarks |
| ML models | scikit-learn always; XGBoost/CatBoost/PyTorch if installed | ✅ trained and validated end-to-end (Random Forest) |
| Explainability | SHAP if installed, else permutation/sensitivity fallback | ✅ fallback tested end-to-end |
| Dashboard | Streamlit | code correct; **needs `pip install streamlit plotly`** |
| Coaching engine | Rule-based, SHAP-aware | ✅ fully working, tested |

Everything is wired so the app runs today with just feature-vector sliders (no
video/GPU needed), and upgrades automatically as you install the optional
heavier libraries — no code changes required.

## Setup

```bash
pip install -r requirements.txt

# One-time: download the MediaPipe pose model (needed only for video upload mode)
wget -O models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# Train demo models (synthetic data) so the dashboard has something to load
python train_demo_model.py --model random_forest
```

## Setting up YOLO11 detection + ByteTrack tracking specifically

These two stages need `ultralytics`, which wasn't installable in the dev
sandbox (no internet), so this was written and API-verified against
Ultralytics' current docs, but not executed end-to-end here. On your own
machine:

```bash
bash scripts/setup_yolo.sh
```

This installs `ultralytics`, downloads the YOLO11-nano weights (`yolo11n.pt`,
~5MB, cached automatically), and confirms `bytetrack.yaml` is present. Then
verify it against one of your own clips:

```bash
python scripts/test_yolo_detection.py path/to/your_clip.mp4
```

This saves annotated frames to `./yolo_test_output/` — the bowler should get
a thick **green** box (the "primary bowler" heuristic: largest × most
confident detection), other people get thin orange boxes. It also runs full
ByteTrack tracking over the clip and reports which track ID was selected as
the bowler.

**Notes:**
- `config.BOWLER_CLASS_ID = 0` is COCO's "person" class — YOLO11 pretrained
  on COCO detects any person, not specifically "a bowler." The
  `select_primary_bowler()` / `select_bowler_track()` heuristics (largest,
  most-confident, longest-lived detection) are what narrow it down to the
  bowler specifically — check these hold for your camera angle; a very wide
  shot with a big crowd may need a tighter heuristic (e.g. restrict to a
  region of interest around the bowling crease).
- `yolo11n.pt` (nano) is the fastest/least accurate size. If detection misses
  the bowler often, try swapping in `yolo11s.pt` or `yolo11m.pt` in
  `config.YOLO_WEIGHTS` — larger model, same API, no code changes needed.
- If you'd rather train a model specifically on "bowler" as its own class
  (instead of relying on generic "person" + heuristics), you'd need a
  labeled dataset of bowler bounding boxes and would fine-tune with
  `model.train(data=your_data.yaml, ...)` — a meaningfully bigger project;
  ask if you want help setting that up.

## Run the dashboard

```bash
streamlit run app.py
```

Two modes in the sidebar:
- **Manual feature entry** — move sliders for the 10 biomechanical features and see live predictions. No video or GPU needed.
- **Upload bowling video** — runs the full CV pipeline (detection → tracking → pose → features).

After any analysis, use **Save this result to history** (section 7) to persist the
delivery into a local SQLite database (`data/bowling_history.db`).

Two pages in the sidebar:
- **Analyze** — run a new delivery (manual entry or video upload).
- **History & Compare** — browse every saved session, track the performance trend
  over time, compare sessions side-by-side (performance, injury risk, and a
  feature-by-feature table/delta), inspect full details of any session, and
  delete/clear history.

## Project layout

```
cricket_biomech_ai/
├── app.py                    # Streamlit dashboard
├── train_demo_model.py       # trains & saves performance/injury models
├── requirements.txt
├── src/
│   ├── config.py             # paths, thresholds, feature list, landmark names
│   ├── preprocessing.py      # frame extraction, resize, denoise
│   ├── detection.py          # YOLOv11 bowler detection (+ HOG fallback)
│   ├── tracking.py           # ByteTrack (+ IoU fallback)
│   ├── pose_estimation.py    # MediaPipe 33-landmark pose extraction
│   ├── feature_engineering.py# 10 biomechanical features from landmarks
│   ├── ml_models.py          # RF / XGBoost / CatBoost / CNN-LSTM / Transformer
│   ├── explainability.py     # SHAP (+ permutation-importance fallback)
│   ├── history_db.py         # SQLite history DB for saved analyses (stdlib sqlite3)
│   ├── coaching.py           # rule-based coaching recommendation engine
│   ├── synthetic_data.py     # generates a plausible demo dataset
│   └── pipeline.py           # orchestrates the full video→coaching flow
├── models/                   # trained model bundles (.joblib) + pose model (.task)
├── data/                     # synthetic_bowling_dataset.csv (generated)
└── scripts/
    ├── setup_yolo.sh          # installs ultralytics, downloads yolo11n.pt, checks bytetrack.yaml
    └── test_yolo_detection.py # verifies detection+tracking on a real clip, saves annotated frames
```

## Using real data

`synthetic_data.py` exists only so the app works before you have labeled data.
Once you have real (features → performance_score, injury_risk) rows —
ideally scored by a coach/biomechanist and a sports-medicine team —
retrain with:

```bash
python train_demo_model.py --data your_labeled_dataset.csv --model xgboost
```

CSV columns required: the 10 names in `src/config.FEATURE_NAMES`, plus
`performance_score` (0–100) and `injury_risk` (0=low, 1=moderate, 2=high).

### ML validity safeguards

The ML layer reports its own honesty, so nobody is fooled by demo metrics:

- **Honest evaluation**: models are scored with 5-fold cross-validation
  (scaler refit per fold, no leakage) instead of a single train/test split.
- **Baseline comparison**: every metric is printed next to a trivial baseline
  ("always predict the mean" / "always predict the majority class") so you can
  see whether the model actually adds predictive value.
- **Data provenance**: bundles are tagged `synthetic` or `real`; the dashboard
  shows a loud warning when predictions come from synthetic demo models.
- **Input validation**: `train_demo_model.py` rejects missing columns, bad
  `injury_risk` values, non-numeric features, and warns on implausible ranges.
- **Out-of-distribution detection**: the dashboard warns when a delivery's
  features fall outside the training range (predictions would extrapolate).
- **Prediction uncertainty**: the dashboard shows a ~68% prediction interval
  for the performance score (spread of random-forest trees).
- **Circular-label warning**: synthetic labels are generated *from* the
  features, so near-perfect metrics there only measure fit to the demo
  generator — this is called out explicitly rather than presented as accuracy.

## Important caveats

- **Elbow flexion / ICC legality**: the feature is a good *screening* signal,
  not a certified throwing test. Official illegal-action rulings require
  lab-grade 3D motion capture per ICC protocol.
- **Coaching thresholds** in `coaching.py` are illustrative, drawn from
  published fast-bowling biomechanics ranges — calibrate them (and better,
  replace the rule engine's role with your trained model's SHAP output) against
  your own population before using for real training-load or medical decisions.
- **This is a decision-support tool**, not a substitute for a qualified coach,
  biomechanist, or sports physician.
