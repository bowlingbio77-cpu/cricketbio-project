# PACEAI — COMPLETE SYSTEM AUDIT

**Project root:** `D:\cricket_biomech_ai`
**Audit date:** 2026-08-27
**Mode:** READ-ONLY (no application files modified during this audit)
**Evidence legend:** 🟢 VERIFIED (inspected code / ran it) · 🟡 INFERRED (strong code evidence, not executed) · 🔴 COULD NOT VERIFY / MISSING

> Note on timestamps: this audit reflects the repository as it exists on the audit date. Several files were legitimately modified during development immediately before this audit (video-encoder fix and unified Analysis Replay wiring); those changes are part of the current state and are described as such.

---

## 1. PROJECT OVERVIEW

PACEAI is an AI-powered cricket fast-bowling biomechanics analysis platform. A user uploads a bowling delivery video (or dials in 10 kinematic sliders manually); the system runs computer vision (YOLOv11 person + sports-ball detection, ByteTrack multi-object tracking, MediaPipe Pose Landmarker) to extract the bowler's movement, derives 10 biomechanical features, and then uses trained ML models to produce a **performance score (0–100)**, an **injury-risk classification (low/moderate/high)**, **SHAP explainability**, **rule-based coaching recommendations**, and an **ICC arm-legality check**. Intended users are cricket coaches, players, sports physiotherapists, and (for a hackathon/judge audience) technical evaluators of an AI sports-science pipeline. The main workflow is: upload/preview → staged analysis → biomechanics → ML prediction → explainability → coaching → history. It is differentiated by combining a full CV pipeline with biomechanical feature engineering, transparent SHAP explanations, injury-risk modeling, and coaching — all in one interactive dashboard. Maturity: a functional, well-engineered **prototype/demo** (models are trained on synthetic data for the core performance/injury pair), with partial real-data secondary models and 167 passing tests.

---

## 2. COMPLETE PROJECT STRUCTURE

| FILE / DIRECTORY | PURPOSE | IMPORTANCE |
|---|---|---|
| `app.py` (92 KB) | Main Streamlit application (entry point, UI, all pages) | **CRITICAL** |
| `chat_assistant.py` | Sidebar Ollama chat widget (HTTP client) | MODERATE |
| `start_app.bat` | Idempotent launcher → `python -m streamlit run app.py` on port 8501 | HIGH |
| `requirements.txt` | Pinned dependency manifest | HIGH |
| `pyproject.toml` | Python packaging / tool config | LOW–MODERATE |
| `test_synthetic.mp4` | Sample video for demos | LOW |
| `assets/loading_overlay.html` | Loading overlay template | MODERATE |
| `src/pipeline.py` | End-to-end orchestration `analyze_video()` → `AnalysisResult` | **CRITICAL** |
| `src/preprocessing.py` | Video decode → frames (`preprocess_video`) | HIGH |
| `src/detection.py` | Bowler detection (YOLOv11, HOG fallback) | HIGH |
| `src/tracking.py` | Bowler tracking (ByteTrack, IoU fallback) | HIGH |
| `src/ball_tracking_v2.py` | Ball detection+tracking, video writing, reels, validate_video (63 KB) | **CRITICAL** |
| `src/ball_tracking.py` | Legacy v1 ball tracking (heuristic) | LOW (superseded) |
| `src/pose_estimation.py` | MediaPipe Pose Landmarker → 33 landmarks `PoseFrame` | **CRITICAL** |
| `src/feature_engineering.py` | 10 biomechanical features from pose | **CRITICAL** |
| `src/video_validity.py` | Cricket-scene pre-check + OOD/reliability | HIGH |
| `src/ml_models.py` | ML training/predict (RF/XGB/CatBoost/CNN-LSTM/Transformer) | **CRITICAL** |
| `src/explainability.py` | SHAP (+ permutation fallback) | HIGH |
| `src/coaching.py` | Rule-based coaching recommendations | HIGH |
| `src/synthetic_data.py` | Synthetic demo dataset generator | HIGH |
| `src/sports_injury_data.py` | Multimodal sports-injury dataset loader | MODERATE |
| `src/cricket_injury_data.py` | Cricket injury dataset loader | MODERATE |
| `src/injury_knowledge_base.py` | Clinical injury benchmarks KB | MODERATE |
| `src/history_db.py` | SQLite history persistence | HIGH |
| `src/analysis_replay.py` | Unified "Analysis Replay" video renderer (hero video) | HIGH |
| `src/auth_login.py` | Login screen + `is_authenticated` (not active) | LOW (disabled) |
| `src/batch_analysis.py`, `comparative_experiments.py`, `assistant.py` | Batch / experiments / behind-the-scenes assistant | LOW |
| `src/cricket_ball_detection.py` | Cricket-ball fine-tune **config only** (no trained model) | LOW |
| `tests/` (10 files, 167 tests) | Unit + integration test suite | **HIGH** |
| `scripts/` | Train/eval/label/smoke scripts (`train_demo_model.py`, `train_yolo.py`, `gen_gt_clips.py`, etc.) | MEDIUM |
| `data/` | Datasets + SQLite history (`bowling_history.db`) | HIGH |
| `models/` | Pre-trained bundles (RF/XGB/CatBoost/CNN-LSTM/Transformer + yolo11n.pt + pose task) | **CRITICAL** |
| `evaluation/` | `evaluate.py` (27 KB), `metadata.csv` (empty), videos/annotations/predictions (empty) | MODERATE |
| `corrected_all_data/bowling/` | 2,562 real .avi clips | HIGH (real data) |
| `data/gt_clips/` | 25 synthetic ground-truth clip dirs (each ~3 items) | MEDIUM |
| `data/cricket_ball_dataset/` | `yolo_dataset/` (train/val/data.yaml) + `auto_labeled/` | MEDIUM |
| `docs/screenshots/` | Screenshots | LOW |
| `README.md` | Documentation | MEDIUM |

Irrelevant caches (`__pycache__`, `.git`) omitted. `venv/` and `runs/` are ignored/absent.

---

## 3. TECHNOLOGY STACK

**Frontend / UI**
- **Streamlit 1.60.0** (verified) — entire UI in `app.py`.
- **Plotly 6.9.0** (verified) — gauges, radar, SHAP bars, trajectory charts.
- Custom HTML/CSS (dark theme, metric cards, badges, hero banner, preloader, loading overlay).

**Backend / Application**
- **Python 3.14** (global interpreter used by `start_app.bat`), modules in `src/`.
- threading? No dedicated backend server; Streamlit is the app server. Ollama chat via `requests`.

**Computer Vision**
- **OpenCV 5.0.0** (`opencv-python-headless==5.0.0.93`, verified) — frame I/O, drawing, video writer fallback.
- **NumPy 2.5.1**, **pandas 3.0.5** — array/table processing.

**Object Detection**
- **Ultralytics YOLOv11** (`ultralytics==8.4.115`, verified; `yolo11n.pt` present) — person + sports-ball.
- Fallback: OpenCV HOG person detector.

**Tracking**
- **ByteTrack** (via ultralytics `model.track`) — bowler.
- **Custom multi-hypothesis Kalman filter** (`ball_tracking_v2.py`) — ball.

**Pose Estimation**
- **MediaPipe 1.0.0** Tasks API PoseLandmarker **heavy** (`pose_landmarker_heavy.task`, ~30 MB, present) — 33 landmarks.

**Machine Learning**
- **scikit-learn 1.9.0** (verified) — RandomForest/GBM/MLP fallback.
- **XGBoost 3.3.0** (verified), **CatBoost 1.2.10** (verified).
- **PyTorch 2.13.0+cpu** (verified installed; **not** in requirements.txt — commented).
- **joblib 1.5.3** — model serialization.

**Explainability**
- **SHAP 0.52.0** (verified) + **numba 0.67.0**; fallback = sklearn permutation importance.

**Database**
- **SQLite** via `src/history_db.py` (`data/bowling_history.db`).

**Video Processing**
- **OpenCV** for read/write.
- **imageio-ffmpeg 0.6.0** (verified) — bundles static ffmpeg **7.1** (libx264) for browser-compatible H.264 writes.

**Visualization**
- **Plotly 6.9.0**; native Streamlit metrics/captions/progress.

**Testing**
- **pytest** (167 tests across 10 files; all pass — verified this audit).

**Deployment**
- **`start_app.bat`** idempotent launcher; hardcoded absolute Python path; no Docker/K8s; no CI config found.

**Other**
- **requests 2.34.2** — Ollama chat HTTP.
- **imblearn?** — NOT verified (not in straight import probe; MLP/backends are sklearn).

Version of OLLAMA server: not part of repo (external service on port 11434, model `llama3.2:1b` per earlier session usage).

---

## 4. APPLICATION ENTRY POINT

**Entry:** `start_app.bat` → runs `python -m streamlit run app.py --server.headless true` on port **8501**. The batch is **idempotent**: it checks for a listener on 8501 and does not spawn a duplicate server/tab (verified — lines 9–24).

**Main pages / views** (Streamlit sidebar `page` selector):
- Main analysis page (default).
- **History & Compare** (`render_history_page`) — saved SQLite results, with `_confirm_clear_history`.
- Sidebar chat widget (`render_chat_widget`, `chat_assistant.py` — Ollama).

**Navigation:** Streamlit sidebar with `page` radio; sidebar also holds model choice, bowling arm, camera view, input mode, and analysis settings (target FPS, resolution, denoise, reels sliders). A "System Status" block reports backend availability.

**Session state keys** (grep-verified): `video_output_path`, `pose_video_path`, `reels_video_path`, `analysis_replay_path` (added in dev), `ball_stats`, `video_stage_times`, `video_upload_time`, `last_warnings`, `features`, `performance_score`, `injury_risk`, `shap_values`, `recommendations`, plus simulator inputs, history filters, and login keys (`authenticated`, etc.).

**Authentication:** Implemented (`src/auth_login.py`, `is_authenticated`) but **NOT active** — the gate and logout are commented out in `app.py` (lines 55–56, 1267–1269). Confirmed: authentication is disabled in the running app.

**Configuration:** `src/config.py` — TARGET_FPS=20, RESIZE_DIM=(640,360), DENOISE=False, YOLO conf 0.4, POSE conf 0.5, ICC elbow limit 15°, feature list, model dir, dataset paths, injury risk thresholds (low 0.33 / high 0.66).

**Environment variables:** `PACEAI_LOGIN_PASSWORD_HASH` / `PACEAI_LOGIN_USER` referenced in `auth_login.py` (only relevant if auth were enabled). No other `.env` usage found.

**Major app functions:** `load_or_train_models`, `render_loader`, `render_fullscreen_splash`, `analysis_panel_html`, `render_analysis_pipeline`, `render_modern_gauge`, `render_radar_comparison`, `render_shap_bar`, `render_timings`, `render_model_quality_expander`, `render_ood_warnings`, `render_plain_language_summary`, `render_features_guide`, `render_history_page`, `_stage_cb` (real pipeline progress callback).

**Flow:** import modules → load/train models → hero banner → input (simulator sliders OR video upload) → optional feature guide (collapsed) → on video: `_stage_cb` drives a live analysis panel → `pipeline.analyze_video(...)` → store outputs in session state → render results (metric cards, summary, deep-dive expander, model-quality, timings, save-to-history) → sidebar chat.

---

## 5. COMPLETE USER JOURNEY

Primary **video** path (from `app.py` lines 1332–1415 + pipeline):

1. **Select input mode** — sidebar: "📹 Upload Video" or "🎛️ Interactive Bio-Simulator".
2. **Upload** — `st.file_uploader` (`mp4/mov/avi`); bytes written to `tempfile.NamedTemporaryFile` (`video_path`). No pre-upload video preview metadata card found (a gap).
3. **Settings** — FPS, resolution, denoise, reels sliders, bowling arm, camera view.
4. **Click Analyze** — the app auto-runs on upload (no separate button in the video path as currently coded). **System:** `render_loader` + `analysis_panel_html` progress; `_stage_cb(done,total,label)` mirrors real pipeline stages.
5. **Analysis** — `pipeline.analyze_video(video_path, ..., run_ml=False, progress_cb=_stage_cb)`; ML is re-run afterward via `analyze_feature_vector`.
6. **Results** — metric cards, plain-language summary, deep-dive expander (6 tabs), model-quality, disclaimer, run timing, **for video: ball/pose/reels video section**, save-to-history.
7. **Cleanup** — `finally: os.remove(video_path)` removes the **uploaded temp file only** (line 1409). Output MP4s are kept in session state and survive reruns.

**Secondary path — Interactive Bio-Simulator:** user drags 10 feature sliders (preset loadable), then `analyze_feature_vector` runs directly (no CV). No video produced.

**History:** user can enter name/label/tags and "Sava this result" → `history_db.save_analysis` (idempotent).

**FILES/DATA CREATED:** uploaded temp MP4 (deleted), ball-track MP4, pose MP4, reels MP4, analysis-replay MP4 (all in OS temp), SQLite history row, session-state dicts.

**NEXT STEPS** in the flow are linear: upload → analyze → results → (optional) history/chat.

---

## 6. VIDEO PIPELINE — EXTREMELY DETAILED

Order of operations is exactly as coded in `pipeline.analyze_video` (lines 253–). **Preliminary note:** `video_precheck` cricket-validation runs BEFORE ball tracking (a "fail fast" ordering), and ball tracking runs on the FULL frames before the bowler crop.

1. **Preprocess** — `preprocessing.preprocess_video(video_path, target_fps, resize_dim, denoise)` → list of `(idx, ts, BGR ndarray)`. VERIFIED. Output: `frames`.
2. **Cricket pre-check (hard gate)** — `video_validity._precheck_cricket(frames, bowling_arm)` → `CricketPrecheckError` if not a bowling scene. FAST sub-sampled pose pass. VERIFIED (see `video_validity.py`).
3. **Ball detection + tracking (full frames)** — `ball_tracking_v2.track_ball(frames)` → `(track, track_stats)`; `annotate_frames` + `write_mp4` → `video_path`. Only YOLO detections ≥ SEED_MIN_CONF may originate a track. VERIFIED.
4. **Detect/track bowler (full frames)** — `tracking.BowlerTracker().track_frames(...)` + `select_bowler_track` → paddeds bboxes `bowler_bboxes`; then **`frames` is REPLACED** with bowler-cropped frames (`_crop_frames_to_bowler`) unless no stable track. VERIFIED.
5. **Pose estimation** — `pose_estimation.PoseEstimator().process_video_frames(frames)` → `pose_sequence` (`PoseFrame`, 33×4 landmarks). VERIFIED.
6. **Feature engineering** — `feature_engineering.analyze_delivery(pose_sequence, bowling_arm, camera_view)` → `(feature_vector[10], diagnostics)`. VERIFIED.
7. **Wrist-proxy ball augmentation** — `_extract_wrist_pixel_positions` + `_augment_trajectory_with_wrist_proxy` (crop→full mapping). VERIFIED.
8. **Pose-overlay video** — `pose_estimation.draw_skeleton` on cropped frames → `pose_video_path`. VERIFIED.
9. **ML** (second call, `run_ml=False` first time) — `ml_models.predict(perf_bundle, feature_vector)` + `explainability.explain_prediction` ×2. VERIFIED.
10. **Coaching** — `coaching.generate_recommendations(...)`. VERIFIED.
11. **Reels** — `ball_tracking_v2.render_slowmo_zoom(video_path, reels_video_path, traj, ...)`. VERIFIED.
12. **Analysis Replay (unified hero video)** — `analysis_replay.render_analysis_replay(frames_full, display_track, pose_sequence, bowler_bboxes, release_frame, ...)` → `analysis_replay_path`. VERIFIED as wired (dev change immediately before audit).

**Failure conditions:** empty frames → `CricketPrecheckError`; <3 pose frames → `RuntimeError("Not enough frames...")`; ball not detected → warnings only; tracker exceptions → caught+warnings; per-stage try/except convert hard failures to warnings so ML can still run.

**Used by the UI?** `video_path`, `pose_video_path`, `reels_video_path` all surfaced in app.py video section (lines 1414–1507); `analysis_replay_path` is newly wired into session state/display during this dev cycle.

---

## 7. VIDEO PLAYBACK AUDIT

**Chain:** uploaded file → temp `video_path` → `preprocess_video` decodes frames with OpenCV → renderers build frame lists → `write_mp4` → output MP4 → `os.path.exists()` guard → `st.video(path)` → browser.

**Writer (current state, `ball_tracking_v2.write_mp4`):** Primary path pipes raw BGR frames to imageio-ffmpeg's bundled ffmpeg (`libx264 -pix_fmt yuv420p -movflags +faststart`); on failure it writes intermediate via OpenCV `mp4v` then **transcodes to H.264** via ffmpeg. Frames are normalized (gray→BGR, RGBA→BGR, mixed sizes resized) and validated before return (`validate_video`). This is the result of a fix made in this dev cycle; the original code fell back to **OpenCV MPEG-4 Part 2 (`mp4v`)**, which browsers cannot play.

**FPS handling:** `target_fps` (default 20) passed to writer; invalid FPS falls back to 20 (fixed). OpenCV CAP gets FPS from the writer's configured param, not from the file container in all cases.

**Resolution/color:** fixed per-video (from first frame); `write_mp4` normalizes dimensions; code writes BGR rawvideo (correct for ffmpeg with `-pix_fmt bgr24`).

**Output paths:** `ball_tracking.make_output_path(prefix)` → `tempfile.mkstemp` in OS temp (e.g. `ball_track_*.mp4`, `pose*.mp4`, `reels*.mp4`, `analysis_replay*.mp4`).

**Cleanup:** the **uploaded temp** is removed in `app.py finally`; generated MP4s are **not** deleted (they persist for the session). Risk: orphaned temp MP4s accumulate across runs (minor).

**st.video usage:** `st.video(_ball_vid)`, `st.video(_pose_vid)`, `st.video(_reels_vid)` (app.py lines 1431/1496/1507) — each guarded by `os.path.exists(...)`. The new Analysis Replay is displayed via `st.video(analysis_replay_path)`.

**Verified generated-video evidence (probed with ffmpeg + OpenCV this audit):**
- `ball_track_iu9ser0y.mp4` 🟢: **H.264 (avc1, High), yuv420p, MP4, 640×360, 20 fps, 60 frames, 3.0 s, 236 KB**; first/mid/last frames decode. Browser-playable.
- `posegemzsyz9.mp4` 🟢 (as generated before the writer fix): **MPEG-4 Part 2 (`mp4v`) Simple Profile, 60×124, 20 fps, 1 frame, 3 KB** — **NOT browser-playable** and effectively broken (1 frame). Root cause of the "video doesn't play" symptom when the pose fallback was displayed.
- `tmpn94kwkv1.mp4` (an uploaded input): **MPEG-4 (`mp4v/FMP4`) + mp3 audio, 1280×720, 25 fps, 60 frames**. Non-H.264 input — if the app ever previews the raw upload via `st.video`, that would also be browser-unplayable.
- After writer fix: `mixed.mp4`/`forced_fallback.mp4`/`analysis_replay.mp4` all 🟢 **H.264 (avc1)**, valid dims/FPS/frames.

**Conclusion (root cause, now fixed):** outputs previously fell back to OpenCV `mp4v` because the imageio streaming path could deadlock (unread `stderr` pipe), silently producing browser-unplayable MPEG-4 Part 2 files. The pose output was additionally only 1 frame (a data/render limitation, separate from encoding).

**Outstanding risk:** the **pose-overlay** output can still be a very small number of frames if pose is detected in few frames; and the **raw upload preview** is not H.264. Neither currently blocks the main hero (ball-track/analysis-replay are H.264).

---

## 8. VIDEO OUTPUTS

| Name | Created by | Purpose | Codec (verified) | Resolution | FPS | Audio | Displayed where | Temp/Persistent | Used by UI | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Ball-track MP4 (`video_path`) | `ball_tracking_v2.write_mp4` | Ball box + trajectory overlay | H.264 avc1 | (resized, e.g. 640×360) | target (20) | none | app.py "Ball Tracking Visualization" | temp-file persisted for session | ✅ | ✅ H.264 |
| Pose-overlay MP4 (`pose_video_path`) | `write_mp4` on `draw_skeleton` frames | Pose skeleton overlay / ball fallback | was mp4v (now H.264) | bowler-crop (e.g. 60×124) | target | none | app.py "Pose Skeleton Overlay" | temp persisted | ✅ | ⚠️ can be 1 frame |
| Reels MP4 (`reels_video_path`) | `render_slowmo_zoom` | Slow-mo + zoom highlight | H.264 (uses write_mp4 path) | from ball vid | target | none | app.py "Reels" | temp persisted | ✅ | ✅ |
| Analysis Replay MP4 (`analysis_replay_path`) | `analysis_replay.render_analysis_replay` | ONE unified hero video (ball+traj+pose+release) | H.264 avc1 (verified) | full frame (e.g. 640×360) | target | none | app.py hero (new) | temp persisted | ✅ | ✅ |
| Uploaded temp input | `st.file_uploader` bytes | decode source | depends on upload (mmpeg4/FMP4 seen) | as uploaded | any | may have | n/a (not displayed raw) | deleted in `finally` | — | — |

**Duplicate/redundant outputs:** The **pose**, **ball-track**, and **reels** videos overlap with the unified Analysis Replay (which already contains ball box, trajectory, and pose skeleton). Per the intended product direction these three are candidates for de-emphasis in favor of ONE hero replay (they remain internal/intermediate).

---

## 9. COMPUTER VISION PIPELINE

- **Bowler detection:** YOLOv11-nano (COCO `person`, class 0), conf ≥ 0.4 (`detection.py`); fallback HOG. VERIFIED.
- **Bowler tracking:** ByteTrack via ultralytics `model.track`; "bowler = longest continuous run"; IoU-greedy fallback (`tracking.py`). VERIFIED.
- **Ball detection:** YOLOv11 COCO class 32 `sports ball` (conf ~0.05–0.35 at delivery-cam res; **not** cricket-fine-tuned — see `cricket_ball_detection.py`: "STATUS: NOT MEASURED"). Motion blobs as secondary candidates. VERIFIED.
- **Ball tracking:** custom multi-hypothesis Kalman (`ball_tracking_v2.py`): constant-velocity model, Mahalanobis gating, ballistic (constant-accel) consistency, track-validity rejection, segment splitting, impact-guard. VERIFIED.
- **Pose estimation:** MediaPipe PoseLandmarker heavy (`pose_estimation.py`), 33 landmarks, primary-person by largest bbox, min vis 0.5. VERIFIED.
- **Trajectory reconstruction:** merged `display_track` (detected + interpolated + wrist-proxy), clipped at impact. VERIFIED.
- **Release detection:** derived in `feature_engineering` (`diagnostics["release_frame_idx"]`) + `track_stats["release_idx"]`. VERIFIED usage.
- **Visualization:** `annotate_frames`, `draw_skeleton`, `analysis_replay` overlays. VERIFIED.
- **Used in final results?** Pose→features→ML: yes. Ball tracking→ball_stats/trajectory + video: yes. Release: yes. Wrist-proxy: yes (augments pre-release trajectory).

---

## 10. BALL DETECTION

- **Model/classes:** YOLOv11-nano pretrained on COCO, class **32 "sports ball"** (generic soccer/basketball/tennis…). Not cricket-specific. VERIFIED (`cricket_ball_detection.py`).
- **Training source:** COCO-pretrained weights `yolo11n.pt`; no fine-tuning on cricket balls (explicitly "NOT MEASURED"). VERIFIED.
- **Input resolution:** frames at `resize_dim` (default 640×360). VERIFIED.
- **Confidence threshold:** config `DETECTION_CONF_THRESHOLD=0.4` for bowler; ball uses its own seed conf (`SEED_MIN_CONF`) in `ball_tracking_v2`. A **cricket dataset exists** (`data/cricket_ball_dataset/auto_labeled` + `yolo_dataset/` train/val/data.yaml) but no trained ball model is referenced.
- **NMS:** ultralytics default NMS. (INFERRED — standard for the library.)
- **Detection output:** boxes/conf → associated into Kalman tracks.
- **Weaknesses (evidence-based):** generic "sports ball" class → false positives on round objects (stumps, helmets) and low confidence on a small fast ball; documented in `cricket_ball_detection.py`.

---

## 11. BALL TRACKING

Multi-hypothesis constant-velocity Kalman (VERIFIED in `ball_tracking_v2.py` and its docstring):
- **A. Ball detected:** YOLO candidate (≥ seed conf) may originate/update a track with tight covariance; assignment order favors YOLO over motion.
- **B. Ball missed:** Kalman prediction bridges short gaps; motion blobs may extend a track **only** while it still has recent YOLO support (`YOLO_LIVENESS_FRAMES`) and is ball-sized.
- **C. Confidence drops:** weaker candidates get loose measurement covariance; Mahalanobis gate vs predicted uncertainty decides association.
- **D. Multiple candidates:** several tracks run in parallel (bounded `MAX_ACTIVE_TRACKS`); ballistic-consistency ranking selects winner at end; assignment-order rule prevents a closer blob stealing a good track.
- **E. Ball disappears:** trailing off-frame trimming; track may be rejected if it degrades (pinned/fixed, or slow-growing person).
- **F. Ball reappears:** segment splitting on implausible teleport; new hypothesis evaluated.

---

## 12. POSE ESTIMATION

- **Model:** MediaPipe Pose Landmarker **heavy** (`pose_landmarker_heavy.task`, ~30 MB present), Tasks API, `mediapipe==1.0.0`.
- **Output:** `PoseFrame(frame_idx, timestamp_sec, landmarks[33,4], world_landmarks[33,3])` — normalized x,y,z,visibility + metric world coords. VERIFIED.
- **Filters/conf:** min detection/tracking confidence 0.5; primary person = largest landmark bbox. VERIFIED.
- **Missing landmarks/smoothing:** visibility thresholding in renderers (`min_visibility`), no explicit temporal smoothing found (INFERRED none).
- **Handling left/right arm:** `bowling_arm` selects MediaPipe anatomical side; wrist landmark (15/16) used for wrist-proxy; front foot chosen from motion. VERIFIED.
- **Pose → biomechanics:** `feature_engineering.analyze_delivery` converts landmark sequence → 10 features using **world landmarks** (metric) to avoid perspective bias, with 2D fallback. VERIFIED.

---

## 13. BIOMECHANICS

Features (from `config.FEATURE_NAMES` + `feature_engineering.py`) and grounding:

| Feature | Source | Units | Used by ML? | Shown to user? | Validation |
|---|---|---|---|---|---|
| shoulder_rotation_deg | shoulder/hip landmarks | deg | yes | yes (labels/report) | FEATURE_LABELS range |
| elbow_flexion_deg | shoulder/elbow/wrist | deg | yes | yes (ICC legality) | ICC 15° threshold |
| wrist_angle_deg | elbow/wrist/hand | deg | yes | in report | ranges |
| hip_rotation_deg | hips | deg | yes | in report | ranges |
| knee_flexion_deg | hip/knee/ankle | deg | yes | yes (metric card) | ranges |
| trunk_lean_deg | shoulders/hips | deg | yes | yes (coaching) | 15–40 band |
| stride_length_norm | ankles/hips | ratio | yes | in report | ranges |
| release_angle_deg | delivery vector | deg | yes | in report | ranges |
| angular_velocity_deg_s | joint change over time | deg/s | yes | stress gauge | <1200 |
| ground_contact_time_s | front-foot contact window | s | yes | in report | ranges |

Formulas are implemented in `feature_engineering.py` (world-landmark based); I did not transcribe each formula (explicitly not guessing). Validation = unit tests + `FEATURE_LABELS` benchmark ranges + a `reliable` diagnostic flag.

---

## 14. FEATURE ENGINEERING

- **Where:** `feature_engineering.analyze_delivery` (pipeline stage 6).
- **Features:** the 10 above (`config.FEATURE_NAMES`).
- **Transformations:** angle computation from world landmarks; front/back-foot detection via ankle-motion; release-frame detection; temporal deltas for angular velocity / stride / contact.
- **Normalization/scaling:** ML bundles use a `StandardScaler` (fit per model bundle — `bundle.scaler`). Feature engineering itself does NOT scale; scaling happens inside the model pipeline. VERIFIED.
- **Missing-value handling:** `SimpleImputer` inside `ml_models`. VERIFIED.
- **Clipping:** some stress indexes clamp to 100 in the UI (UI-level, not feature-level). VERIFIED in app.py.
- **Output:** `feature_vector` dict (10 keys) + `diagnostics` (n_frames, reliable, release_frame_idx, etc.).
- Delivery-quality diagnostics gate coaching/ML downstream.

---

## 15. MACHINE LEARNING

Two pipeline tasks (performance + injury) each with model families: **RandomForest, XGBoost, CatBoost, CNN-LSTM, Transformer**. Runtime picks backend by availability (RF always; XGB/CatBoost if installed; CNN-LSTM/Transformer need torch → else sklearn MLP fallback). VERIFIED (`ml_models.py`).

- **Performance model:** regression → 0–100 pace score. Bundles: `performance_{rf,xgboost,catboost,cnn_lstm,transformer}.joblib`. Trained on **synthetic** data (`synthetic_data.py`, `data/synthetic_bowling_dataset.csv`).
- **Injury model:** classification → {low, moderate, high}; risk_score = P(injury) × severity (ordinal 0–3) with low/high thresholds 0.33/0.66 (config). Bundles: `injury_*.joblib` (synthetic).
- **Secondary injury models (REAL data):** `sports_injury_*.joblib` trained on `data/multimodal_sports_injury_dataset.csv` (Kaggle: 15,420 session rows, 156 athletes, 3-class); `cricket_injury_*.joblib` on `data/cricket_injury_dataset.csv`; `cricket_severity_*.joblib` (ordinal severity). These are additional models; are they the ones the live app uses? The app's `load_or_train_models(model_choice)` trains/loads the per-model-choice **performance/injury** pair (perf_bundle/injury_bundle) — evidence points to synthetic performance/injury being the ones used for the primary display, with clinical/secondary models exposed in the deep-dive Clinical tab.
- **Inference:** `ml_models.predict(bundle, feature_vector)`; **confidence:** injury uses calibrated probabilities; performance uses prediction interval (`prediction_interval_performance`).
- **Training code:** `scripts/train_demo_model.py` (+ `train_sports_injury_model.py`).
- **Synthetic data involved:** YES for performance/injury pair. Real data for sports/cricket-injury models.

---

## 16. MODEL VALIDATION

**IMPLEMENTED (VERIFIED):**
- K-fold / Stratified / StratifiedGroup KFold cross-validation in `ml_models.py` training path.
- Metrics computed: MAE, R², accuracy, F1, recall, precision, ROC-AUC, MSE, classification_report.
- Prediction-interval for performance.
- 167 unit/integration tests including pipeline, feature-engineering, validation-framework, ball-tracking, video-output.
- `evaluation/evaluate.py` — detection/tracking/release/reels metrics via real (cached) predictions; `--predict` CLI to populate prediction cache.

**NOT IMPLEMENTED / MISSING:**
- No external/held-out **real cricket** validation of the performance/injury pair (those are synthetic).
- `evaluation/metadata.csv` is an **empty header** (85 B); evaluation/videos/annotations/predictions dirs are **empty** → no completed evaluation runs visible.
- No mAP for ball detection, no pose landmark error metric, no calibration curves output for the primary models in the repo evidence.
- No dedicated UI/end-to-end browser tests.

---

## 17. SHAP / EXPLAINABILITY

`explainability.explain_prediction(bundle, feature_vector)` returns `{feature: contribution}`; positive = pushes prediction up (higher score / higher risk). Uses SHAP when available (`SHAP_AVAILABLE=True`, verified), else permutation importance. Applies bundle scaler transform. `render_shap_bar` visualizes performance & injury contributors per tab. User-facing interpretation is a caption + colored bars. **Limitations:** SHAP on synthetic-data models explains synthetic-model behavior, not real physiology; contributions are per-feature, positive/negative.

---

## 18. INJURY-RISK SYSTEM

**Implementation (VERIFIED, config.py lines 64–79 + ml_models):** Injury risk is a **classification/scoring** of *long-term repetitive-exposure* risk, trained on an **ordinal target** `injury_severity` (0=no,1=minor,2=moderate,3=severe) that is **simulated** per-row over a long-term exposure (in synthetic data). The model reports `P(injury)` for a chosen exposure (`INJURY_EXPOSURE_DELIVERIES=500` default), expected severity, and combined `risk_score = P(injury) × severity` ∈ [0,1]; `low` <0.33, `high` >0.66, else `moderate`. `injury_risk` dict holds `risk_level`, `probabilities`.
- **Wording to users:** "INJURY RISK LEVEL … LOW/MODERATE/HIGH RISK", P(High).
- **Disclaimer:** prominent `st.warning` "Model trained on synthetic data … not a medical diagnosis" (video mode) + footer caption.
- **Nature:** it is a **prediction/classification with an explicit long-term-exposure assumption**, NOT a clinical diagnosis. The code does not support medical claims; the app explicitly disclaims them.
- **Real-data cross-check:** separate clinical benchmark KB (`injury_knowledge_base` + `cricket_injury_recovery_benchmarks.json`) surfaces literature thresholds in the Clinical tab — distinct from the ML risk score.

---

## 19. COACHING SYSTEM

**Rule-based** (`coaching.py`, VERIFIED): maps `(feature_vector, performance_score, injury_risk, SHAP)` → plain-English notes via `_RULES` thresholds grounded in published fast-bowling literature (ICC elbow 15°, trunk-lean 15–40, knee 5–30, etc.). Simple LinearSHAP-based personalization: SHAP contributions steer which feature's rule is emphasized. Output: list of `coaching_notes`. **Not ML-prioritized beyond rule order/SHAP steering**; no 15-item overwhelm currently (list is short). Displayed as biomechanics notes + separate canned corrective-drill protocols.

---

## 20. UI / UX AUDIT

- **Landing/hero:** banner with title, subtitle, "AI Engine Ready" / "YOLOv11 + MediaPipe" badges. Purpose: brand/what+tech.
- **Navigation:** sidebar radio (Main / History). Simple.
- **Upload screen:** file_uploader + info line describing the CV stages. **No pre-analysis video preview/metadata card** — user uploads, analysis auto-runs immediately (no explicit "Analyze" button in video mode). UX gap.
- **Processing:** live staged panel via `_stage_cb` (real progress). VERIFIED. Premium panel exists.
- **Results:** 4 metric cards (Performance/Injury/ICC/Front-Knee) + plain-language summary + deep-dive expander (6 tabs) + model-quality + disclaimer + timings + save-to-history. Rich but dense; the main video section (3 separate videos with verbose technical captions) sits above results — potential overload/duplication relative to the intended ONE-hero-video direction.

---

## 21. USER INTERACTIONS

| Control | Action | Backend | Expected | Possible bug |
|---|---|---|---|---|
| file_uploader | select video | writes temp; auto-analyze | runs CV | no separate Analyze button; auto-run |
| sliders (simulator) | adjust 10 feats | analyze_feature_vector | update result | n/a |
| model_choice select | switch family | load_or_train_models | reloads bundle | retraining/torch latency possible |
| bowling_arm / camera_view | set handedness/view | passed to pipeline | correct features | left-arm correctness depends on `_extract_wrist` |
| FPS/res/denoise | analysis settings | pipeline | speed/accuracy tradeoff | higher res slower |
| reels sliders | slow/zoom | render_slowmo_zoom | regenerate reels | only on next run |
| Analyze (video) | — | auto | — | (no explicit control) |
| "Save this result" | persist | history_db | idempotent row | — |
| Clear History | confirm | history_db.clear | wipe | confirm gate present |
| Sidebar chat | chat | chat_assistant (Ollama) | assistant reply | external service dependency |
| expanders/tabs | expand | render helpers | reveal detail | — |
| download (report JSON) | download | on-click | JSON file | — |
| login (if enabled) | auth | auth_login (disabled) | gate app | disabled |

---

## 22. CURRENT VIDEO ANALYSIS UI

**Strengths:** real-staged progress; metric cards; clear disclaimers; rich deep-dive; chart variety.
**Problems (evidence-based):**
- **Multiple videos compete** (ball track + pose + reels) instead of one hero — overlaps with the Analysis Replay.
- **Technical captions shown prominently** (YOLO/ByteTrack/MediaPipe, dashed-vs-solid box explanations) — good for judges, but noisy for coaches at the top.
- **No upload preview/metadata** before analysis.
- **Auto-run on upload** with no explicit Analyze button can feel abrupt.
- **Duplicate storylines:** trajectory chart + ball video + pose video + reels all narrate "what CV saw".
- Left-arm mirroring currently uses a **hardcoded `640` frame width** in app.py (line 1470) — noted as a correctness risk for non-640 resolutions (matches the user's flagged concern).
- "Save to history" and chat are useful but peripheral to the core story.

---

## 23. DATA FLOW

```
Video -> preprocess_video -> frames [(idx,ts,BGR)]
  -> _precheck_cricket (validity) 
  -> track_ball (full frames) -> track/display_track + ball_stats
  -> BowlerTracker -> bowler_bboxes -> frames := cropped
  -> PoseEstimator.process_video_frames -> pose_sequence (33x4)
  -> feature_engineering.analyze_delivery -> feature_vector[10] + diagnostics
  -> wrist-proxy augment trajectory
  -> (ML) predict(perf_bundle) -> performance_score ; predict(injury_bundle) -> injury_risk
  -> explain_prediction x2 -> shap_perf / shap_injury
  -> coaching.generate_recommendations -> coaching_notes
  -> write_mp4 x ball/pose ; render_slowmo_zoom -> reels ; analysis_replay.render -> hero
  -> AnalysisResult -> session_state -> UI
```
Formats: frames = ndarray; BallPoint = dataclass; PoseFrame = dataclass; feature_vector = dict[str,float]; bundles = TrainedBundle(joblib); outputs = mp4 paths + dicts.

---

## 24. STATE MANAGEMENT

- **session_state** holds all outputs (video paths, ball_stats, feature_vector, scores, SHAP, notes). On rerun, video-path keys are read with `os.path.exists` guard → stale paths become None safely.
- **caches:** `@st.cache` / model caching via `load_or_train_models` (INFERRED caching; returns bundles). 
- **Temporary files:** uploaded temp deleted; generated MP4s persist (orphan risk across runs).
- **Stale/deleted-file risks:** low due to exists-guards, but generated temp MP4s are never cleaned.
- **Duplicate processing:** video runs `analyze_video(run_ml=False)` + then `analyze_feature_vector` re-runs ML on the same features (intentional to avoid double CV, but is a second ML pass by design).
- **Inconsistent state:** if a video run's ML stage is skipped, results still render from simulator path — guarded by `if feature_vector:`.

---

## 25. ERROR HANDLING

- try/except with warning propagation at most pipeline stages (ball, tracker, pose overlay, reels, analysis replay) — **silent downgrade** pattern (stage skipped, warning appended).
- Hard gates: `CricketPrecheckError`, "Not enough frames", `write_mp4`/`validate_video` RuntimeErrors.
- Video failure in app: `st.error` + traceback expander + analysis-panel error state.
- Missing pose model → friendly `st.warning` (app.py line 1345).
- DB errors: `history_db` wrapped (INFERRED). Missing-file models → `load_or_train_models` retrains (slow).
- **Silent failures to note:** reels/analysis-replay failures only append warnings (UI may still show nothing); pose overlay can be 1-frame without loud error.

---

## 26. SECURITY

- **File uploads:** arbitrary mp4/mov/avi bytes written to a temp file — no size/format verification before decode (OpenCV handles invalid gracefully). No MIME sniffing. Medium risk for a public deployment.
- **Unsafe HTML:** extensive `unsafe_allow_html=True` with `_esc()` escaping of user/model text (good), but sliders/captions are mostly trusted. Plausible XSS surface if any user-controlled string bypasses `_esc`.
- **Paths:** temp paths via `tempfile`; no user-supplied paths.
- **Secrets/env:** `PACEAI_LOGIN_PASSWORD_HASH` (disabled auth). No API keys in repo evidence (Ollama is local).
- **Auth:** disabled by default; demo login exists if enabled.
- **Data storage:** local SQLite history; no PII beyond user-entered names.

---

## 27. PERFORMANCE

**Bottlenecks (code-evidenced):**
- **YOLO inference per frame** (bowler + ball) and **MediaPipe heavy pose** per frame are the compute-heavy CV stages; heavy model on many frames.
- **Video re-rendering:** ball track write + pose write + reels (decode+slowmo re-encode) + analysis replay = 3–4 full encode passes over the clip → redundant CPU.
- **`analyze_video` + `analyze_feature_vector`** = model load + ML twice (video path).
- Real clip run was observed to **time out** (>~5 min) on ball tracking for a 1.2 MB AVI in an earlier session — YOLO-on-full-res + multi-hypothesis is a likely culprit. (🔥 real-data finding.)
- No GPU use evident (OpenCV/CPU).

Measurable timings for a real delivery were not captured in this audit (only the timeout observation); no invented numbers.

---

## 28. TESTING

**167 tests across 10 files, all PASSING (verified this audit; `pytest tests/` exit 0):**
- test_assistant (18), test_ball_tracking (20), test_ball_tracking_v2 (18), test_feature_engineering (14), test_history_db (7), test_pipeline (10), test_pose_person (2), test_scoring (5), test_sports_injury (6), test_validation_framework (63), test_video_output (4, added in dev).
- Coverage areas: feature engineering, ball tracking (v1+v2), validation/OOD framework, pipeline, scoring, sports injury, history DB, assistant, video-output encoding (new).
- **Gaps:** no UI/Streamlit tests, no end-to-end real-video test in the suite, no YOLO detection test that reaches the model (script exists: `scripts/test_yolo_detection.py`), no ML model-quality regression test.

---

## 29. RESEARCH PAPER READINESS

| Aspect | STATUS | EVIDENCE |
|---|---|---|
| Research problem | PARTIAL | Fast-bowling biomechanics + injury-risk framing; synthetic-data caveat in synthetic_data.py |
| Methodology | PARTIAL | Documented pipeline; no methods section/paper |
| Dataset | PARTIAL | Synthetic (features→labels) + real injury CSVs; no cricket biomechanics real dataset |
| Ground truth | PARTIAL | `data/gt_clips` (synthetic) + `corrected_all_data` real .avi (unlabeled) |
| Annotation | PARTIAL | `scripts/label_ball.py`, `gen_gt_clips.py`, `auto_labeled/` exist; scale limited |
| Experimental setup | PARTIAL | train scripts + evaluation/ framework |
| Baseline | MISSING | no comparison baselines in repo evidence |
| Evaluation metrics | PARTIAL | sklearn metrics + evaluation/evaluate.py |
| Quantitative results | MISSING | no results artifacts; evaluation dirs empty |
| Ablation | MISSING | none |
| Limitations | PARTIAL | disclaimers in app + module docstrings |
| Reproducibility | PARTIAL | pinned requirements + seeds (RANDOM_STATE=42); heavy models need network downloads |

Overall research-readiness: **prototype level**, not yet a paper.

---

## 30. HACKATHON READINESS

| Score | Notes |
|---|---|
| Idea **8/10** | Strong, differentiated concept (full CV→biomech→ML→coach). |
| Technical implementation **8/10** | Deep, well-modularized, 167 tests pass. |
| CV/ML **6/10** | Impressive breadth but core performance/injury models are synthetic-data based; ball detection not cricket-tuned; real-video ball tracking slow/unreliable. |
| UI/UX **6/10** | Rich but cluttered; multiple videos; no upload preview; auto-run. (Improving with unified replay.) |
| Demo **6/10** | Real-stage progress is compelling; but real-video run can time out, and pose/ball reliability on arbitrary clips is uncertain. |
| Innovation **8/10** | Combining CV + biomech + SHAP + coaching + injury in one dashboard is genuinely novel. |
| Research potential **7/10** | Pipeline is a solid skeleton for a real-data study. |
| Reliability **5/10** | Fast synthetic demo is reliable; real-video path is not (slow, pose/ball flakiness; disclaimers). |
| **Overall = 6.8/10 (≈68/100)** | Great concept + code; biggest risk = real-video reliability + synthetic-data credibility. |

---

## 31. CRITICAL BUGS

- **P1 — Pose-overlay video can be 1-frame / tiny (browser-unplayable).** `posegemzsyz9.mp4` = 1 frame, 60×124, MPEG-4 Part 2. Location: `pose_video_path` generation (pipeline stage 6c). Impact: fallback/clip visibly broken. Reproduction: pose detected in very few frames / tiny crop. (Encoder side fixed; data side remains.)
- **P1 — Real-video ball tracking is very slow / can time out** on `corrected_all_data/bowling/*.avi`. Location: `track_ball` (YOLO on full frames). Impact: no result on real clips in default budget.
- **P2 — Raw uploaded videos that are MPEG-4 (`mp4v/FMP4`) are not browser-playable** if ever previewed directly; also some MOVs. Location: upload/preview path (no transcode on input).
- **P2 — Left-arm trajectory mirror hardcodes width 640** (app.py:1470) instead of the actual frame width → wrong for 960/1280 resolutions.
- **P2 — Outdated README references** (reorganization script paths in README lines ~157–199) don't match current layout.
- **P3 — Orphaned temp MP4s** accumulate across runs (generated outputs never cleaned).
- **P3 — Auth gate disabled** (intended, but if interface shows login-related UI it could confuse).

---

## 32. CRITICAL GAPS

**Technical gaps:** no input-video transcode to H.264; no upload metadata preview; redundant encode passes slow the pipeline; no timing capture in CI.
**ML/CV gaps:** core performance/injury models trained on **synthetic** data → realism/credibility; ball detector not cricket-fine-tuned; real-video pose/ball reliability unproven; no confidence calibration for performance.
**Data gaps:** no real labeled cricket-biomechanics dataset; 2,562 real AVI clips are unlabeled; evaluation metadata empty.
**Validation gaps:** no external/holdout real evaluation; no mAP/pose-error; evaluation dir empty.
**UI/UX gaps:** no single hero video experience in the shipped default (multiple videos); no upload preview; auto-run; hardcoded width mirror; info density.
**Research gaps:** no baselines/ablations/quantitative results.
**Hackathon gaps:** real-video reliability must be demonstrated (slow ball tracking); need a crisp 60–90 s demo narrative.

---

## 33. REDUNDANT / UNUSED CODE

- **`src/ball_tracking.py` (v1)** — superseded by `ball_tracking_v2` for the live pipeline; still tested.
- **Redundant video outputs** — ball-track + pose + reels overlap the unified Analysis Replay (product direction to consolidate).
- **`cricket_ball_detection.py`** — investigation/config only, not used at inference.
- **Commented-out auth gate + logout** in app.py (dead-ish UI flow).
- **`data/test_synthetic.mp4`**, `tmpn94kwkv1.mp4`/tmp files — sample/test artifacts.
- Duplicate ML rerun (`analyze_video(run_ml=False)` then `analyze_feature_vector`) — by design but redundant-looking.
- Some legacy `FEATURE_LABELS`/reports may reference v1 features (INFERRED).

---

## 34. ARCHITECTURE DIAGRAM

```
USER
 ↓
Streamlit UI (app.py)
 ├ sidebar: page / model / arm / view / settings / reels / chat
 ├ input: simulator sliders  OR  video upload
 ↓
pipeline.analyze_video (src/pipeline.py)
 ├ preprocessing.preprocess_video             -> frames
 ├ video_validity._precheck_cricket           -> hard gate
 ├ ball_tracking_v2.track_ball (YOLO+Kalman)  -> track, ball_stats, ball video
 ├ tracking.BowlerTracker (ByteTrack)         -> bowler_bboxes, crop
 ├ pose_estimation.PoseEstimator              -> PoseFrame[33]
 ├ feature_engineering.analyze_delivery       -> feature_vector[10]+diagnostics
 ├ wrist-proxy augment
 ├ pose-overlay video ; reels ; analysis_replay (write_mp4 -> H.264 MP4)
 ↓
feature_vector
 ├ ml_models.predict (performance) -> 0-100
 ├ ml_models.predict (injury)      -> low/mod/high
 ├ explainability.explain → SHAP
 ├ coaching.generate_recommendations
 ↓
AnalysisResult -> session_state -> UI renders (cards, video hero, tabs, disclaimers)
 ↓
history_db.save_analysis (SQLite)
```

---

## 35. COMPLETE FEATURE INVENTORY

| FEATURE | IMPLEMENTED? | WHERE | REAL DATA? | USER VISIBLE? | TESTED? | STATUS |
|---|---|---|---|---|---|---|
| Video upload | ✅ | app.py | yes | yes | — | ✅ |
| Interactive simulator | ✅ | app.py | n/a | yes | — | ✅ |
| Ball detection | ✅ | ball_tracking_v2 | real frames | via overlay | yes (v2) | ✅ (generic sports ball) |
| Bowler detection | ✅ | detection.py | real | warn | — | ✅ |
| Bowler tracking | ✅ | tracking.py | real | warn | — | ✅ |
| Pose estimation | ✅ | pose_estimation | real | overlay | yes | ✅ heavy |
| Biomech features (10) | ✅ | feature_engineering | real | yes | yes | ✅ |
| Performance score | ✅ | ml_models | synthetic | yes | yes | ⚠️ synthetic model |
| Injury risk | ✅ | ml_models | synthetic + real | yes | yes | ⚠️ long-term-exposure assumption |
| ICC legality | ✅ | app (elbow) | real | yes | yes | ✅ |
| SHAP explainability | ✅ | explainability | model | yes | — | ✅ |
| Coaching | ✅ | coaching.py | real | yes | — | ✅ (rule-based) |
| Clinical injury KB | ✅ | injury_knowledge_base | literary | deep-dive | yes | ✅ |
| Sports-injury model | ✅ | sports_injury_data | real Kaggle | clinical tab | yes | ✅ real |
| History/compare | ✅ | history_db | yes | yes | yes | ✅ |
| Ollama chat | ✅ | chat_assistant | yes | sidebar | yes | ✅ |
| Unified Analysis Replay | ✅ (new) | analysis_replay | real | hero | yes | ✅ H.264 |
| Reels slow-mo | ✅ | ball_tracking_v2 | real | yes | ✅ | ✅ |

---

## 36. WHAT IS ACTUALLY WORKING?

**🟢 VERIFIED WORKING** — Simulator + full ML/SHAP/coaching results; feature-engineering; sports/cricket-injury real-data models; history DB; sidebar chat; 167 tests pass; ball-track & analysis-replay H.264 video output (post-fix); staged real progress; disclaimers; OOD warnings; reels (smoke-tested).

**🟡 PARTIALLY WORKING / UNCERTAIN** — Real-video end-to-end (ball tracking slow→timeout on some AVI; pose reliability on arbitrary clips uncertain); pose-overlay 1-frame edge case; uploaded non-H.264 preview if displayed; left-arm mirror width; multi-model consistency in deep-dive.

**🔴 BROKEN / MISSING** — (post-fix) pose-overlay MPEG-4/1-frame **data** edge can still produce a broken-feeling clip; evaluation pipeline has no populated runs (empty dirs); no baselines/quant results; README stale references; auth disabled (if a login UI is ever expected).

---

## 37. WHAT SHOULD NOT BE CHANGED

- The **feature-engineering** and **biomechanics** core (world-landmark, front-foot-from-motion) — well-tested, correct.
- The **ML module** (`ml_models.py`) interface/bundles — stable, fallback chain, tested.
- **Ball-tracking v2 safeguards** (validity rejection, impact guard, multi-hypothesis) — validated.
- The **167-test suite** and pipeline/validation framework — must keep green.
- **History DB** schema + idempotent save.
- **Config** thresholds/semantics (ICC 15°, injury thresholds) — documented conventions.

---

## 38. RECOMMENDED PRIORITY ORDER (TOP 10)

1. **Make real-video ball tracking fast/reliable** (YOLO on full res is the blocker). Impact: end-to-end demo works on real clips. Effort: med. Risk: med.
2. **Ship ONE Analysis Replay as the solo video; move ball/pose/reels to Technical** — cuts clutter, matches product vision. Effort: low–med. Risk: low.
3. **Add upload preview + explicit Analyze button + input transcode to H.264.** Effort: low. Risk: low.
4. **Fix pose-overlay 1-frame edge** (require N≥some frames or fall back gracefully). Effort: low. Risk: low.
5. **Fix left-arm mirror to use actual frame width** (app.py:1470). Effort: trivial. Risk: low.
6. **Clean generated temp MP4s** after the session/page. Effort: low. Risk: low.
7. **Populate an evaluation run** on several real clips (H.264 outputs + metrics) to back claims/hackathon. Effort: med. Risk: med.
8. **Add a minimal ML-quality regression test** (synthetic CV + metrics bounds). Effort: low. Risk: low.
9. **Refresh README** to match current layout + a short demo script. Effort: low. Risk: none.
10. **Optional: train a cricket-specific ball detector** with `cricket_ball_dataset`. Effort: high. Risk: high (needs labels/GPU) — defer.

---

## 39. THREE DEVELOPMENT PLANS

**A. HACKATHON PLAN (before demo):** pick 1–2 reliable real clips; verify H.264 hero replay; add upload preview + explicit Analyze; de-clutter to ONE hero video + 4-level result hierarchy; pre-warm models so no retrain wait; 60–90 s script (upload→preview→analyze→stages→replay→score→risk→why→coaching). Fix hardcoded mirror width, temp cleanup.

**B. RESEARCH PLAN:** curate/label a real biomechanics dataset (use `corrected_all_data`); establish ground truth (ball track, release, pose, outcomes); retrain performance/injury on real data with holdout + calibration; add baselines (dumb features, no-pose), ablations (world vs 2D, front-foot rule), and report mAP/MAE/ROC with error bars; document methods/limitations; reproduce with pinned deps + seeds.

**C. LONG-TERM PRODUCT PLAN:** cricket-specific ball detector; GPU/ONNX acceleration + caching frames; clinician review workflow; real deployment auth + secure temp handling; multi-camera 3D reconstruction; longitudinal per-bowler trends with the history DB; calibration to target populations; compliance with medical-software expectations (clearly experimental).

---

## 40. FINAL EXECUTIVE SUMMARY

- **PROJECT:** PACEAI.
- **CURRENT STATE:** A mature, well-abstained, hackathon-grade prototype: a complete CV→biomechanics→ML→SHAP→coaching pipeline behind a rich Streamlit dashboard, with 167 passing tests and a broad model zoo — but the primary performance/injury pair is trained on **synthetic** data, real-video ball tracking is slow, and the default UI is information-dense with multiple overlapping videos.
- **BIGGEST STRENGTH:** End-to-end depth + engineering discipline — real video runs through detection, multi-hypothesis Kalman tracking, MediaPipe pose, view-independent world-landmark biomechanics, ML, SHAP, and coaching, all modular and tested (167 pass), with honest synthetic-data disclaimers.
- **BIGGEST WEAKNESS:** Credibility of the headline predictions rests on synthetic-data models, and the real-video path is the least reliable (slow ball tracking, pose/ball flakiness) — the parts that matter most for a live demo are the least proven.
- **MOST CRITICAL BUG:** The output video path silently produced browser-unplayable MPEG-4 (`mp4v`) files (triggered by a ffmpeg-pipe deadlock) — the serializer now guarantees H.264 (verified), but the pose-overlay output can still be a near-empty 1-frame clip, and downloaded real clips can time out in ball tracking.
- **MOST IMPORTANT RESEARCH GAP:** No real, labeled cricket-biomechanics dataset or external/holdout validation for the core performance/injury models — everything heads toward a paper but the quantitative spine (real ground truth + baselines + results) is missing.
- **MOST IMPORTANT HACKATHON FIX:** Make ONE hero Analysis Replay the sole video and de-clutter the result to a 4-level (Overview → Replay → Biomechanics → AI Insights) hierarchy, so a judge sees a coherent story in ~90 s; on a pre-verified real clip with H.264 output.
- **OVERALL SCORE:** 68/100.
- **FINAL VERDICT:** **NEEDS POLISH** — the concept, code architecture, and test rigor are strong (hackathon-viable), but real-video reliability and synthetic-model credibility, plus UI clutter, must be tightened before it feels like a finished, trustworthy product rather than an impressive prototype.

---

## CONCISE SUMMARY

```
CORE PIPELINE:  Video->preprocess->cricket-gate->ball track(Kalman)->bowler track(ByteTrack)->
                >MediaPipe pose->feature engineering(10, world-landmark)->ML(perf+injury)->SHAP->coaching
VIDEO STATUS:   Output now H.264/MP4 (avc1, verified); pose-overlay can be 1-frame; upload preview absent;
                real-clip ball tracking slow (timeouts on some AVI)
ML STATUS:      RF/XGB/CatBoost/CNN-LSTM/Transformer zoo; performance+injury on SYNTHETIC data;
                sports/cricket injury models on real data (secondary)
CV STATUS:      YOLOv11 person + generic sports-ball (not cricket-tuned); mediapipe heavy; multi-hypothesis Kalman
UI STATUS:      Rich + dense; multiple overlapping videos (ball/pose/reels); no hero replay by default;
                hardcoded 640 left-arm mirror; auto-run on upload
TEST STATUS:    167 tests, 10 files, ALL PASSING (verified)
RESEARCH STATUS: Prototype-level; synthetic core data; no baselines/ablations/quant results; evaluation dirs empty
BIGGEST BUG:    Real-video ball tracking can time out; pose-overlay can emit a 1-frame/broken clip
BIGGEST GAP:    No real labeled dataset + external validation for the core performance/injury models
TOP 3 NEXT:     1) Make real-video tracking fast/reliable  2) Ship ONE Hero Analysis Replay + 4-level UI
                3) Add upload preview + explicit Analyze + input H.264 transcode
OVERALL SCORE:  68/100
```
