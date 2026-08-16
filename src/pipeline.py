"""
End-to-end orchestration: video file -> coaching recommendations.

    Video --> preprocessing --> detection --> tracking --> pose_estimation
      --> feature_engineering --> ml_models (performance + injury)
      --> explainability (SHAP) --> coaching

This ties every module together exactly as in the architecture diagram.
Heavy stages (YOLOv11 detection/tracking, MediaPipe pose) require model
weights fetched over the network on first run -- see each module's docstring.
"""
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np

from . import config, preprocessing, tracking, pose_estimation
from . import ball_tracking_v2 as ball_tracking
from . import feature_engineering as feateng
from . import ml_models, explainability, coaching


@dataclass
class AnalysisResult:
    feature_vector: dict
    performance_score: Optional[float]
    injury_risk: Optional[dict]
    shap_contributions_performance: Optional[dict]
    shap_contributions_injury: Optional[dict]
    coaching_notes: list
    stage_times: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    camera_view: Optional[str] = None
    bowling_arm: str = "right"
    video_path: Optional[str] = None       # annotated ball-tracking MP4 (video mode)
    ball_stats: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _crop_to_bbox(frame, bbox, pad_frac: float = 0.3) -> Optional[np.ndarray]:
    """Crop a frame to an expanded bounding box (best-effort; None if too small)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x, pad_y = pad_frac * bw, pad_frac * bh
    nx1 = max(0, int(x1 - pad_x))
    ny1 = max(0, int(y1 - pad_y))
    nx2 = min(w, int(x2 + pad_x))
    ny2 = min(h, int(y2 + pad_y))
    if nx2 - nx1 < 16 or ny2 - ny1 < 16:
        return None
    return frame[ny1:ny2, nx1:nx2]


def _crop_frames_to_bowler(frames, tracks, bowler) -> list:
    """Return a new frame list cropped to the bowler's track bbox (carrying the
    previous bbox forward over brief gaps). Bboxes must be in frame coordinates."""
    bbox_by_frame = dict(zip(bowler.frames, bowler.bboxes))
    last_bbox = None
    cropped = []
    for idx, ts, frame in frames:
        bbox = bbox_by_frame.get(idx, last_bbox)
        if bbox is not None:
            last_bbox = bbox
            cut = _crop_to_bbox(frame, bbox)
            if cut is not None:
                cropped.append((idx, ts, cut))
                continue
        cropped.append((idx, ts, frame))
    return cropped


def analyze_video(video_path: str, bowling_arm: str = "right",
                   performance_bundle: ml_models.TrainedBundle = None,
                   injury_bundle: ml_models.TrainedBundle = None,
                   target_fps: int = config.TARGET_FPS,
                   resize_dim=config.RESIZE_DIM,
                   denoise: bool = config.DENOISE,
                   camera_view: str = "behind",
                   run_ml: bool = True) -> AnalysisResult:
    """
    Full pipeline on a single delivery video clip. Requires:
      - models/pose_landmarker_heavy.task (MediaPipe pose model, download separately)
      - trained performance_bundle / injury_bundle (see train_demo_model.py)
    Detection+tracking (YOLOv11/ByteTrack) crops to the bowler before pose
    estimation when `ultralytics` is installed; otherwise pose estimation runs
    on the full frame (fine for single-bowler, tightly-framed clips).
    `target_fps` / `resize_dim` / `denoise` override the preprocessing defaults
    (speed vs. accuracy trade-off). `camera_view` is recorded and passed to the
    feature engineering (2D fallbacks assume a rear/behind view).
    `run_ml=False` skips the prediction/SHAP/coaching stages (when the caller
    will re-run them on the same feature vector) -- avoids a wasted ML pass.
    """
    timings = {}
    warnings = []
    t_start = time.perf_counter()

    # 1: preprocess
    t0 = time.perf_counter()
    frames = list(preprocessing.preprocess_video(video_path, target_fps=target_fps,
                                                  resize_dim=resize_dim, denoise=denoise))
    timings["preprocess"] = time.perf_counter() - t0

    # 1b: ball detection + tracking -> annotated output video (run on the full
    # frames BEFORE the bowler crop, so the ball is never cut out of frame).
    video_path = None
    ball_stats = {}
    t0 = time.perf_counter()
    try:
        track, track_stats = ball_tracking.track_ball(frames)
        if track:
            impact_idx = track_stats.get("impact_idx")
            # The ball is only relevant until it hits the bat/pad/ground: clip
            # the track at impact so the red box (and the stats below) stop at
            # contact instead of chasing the deflected/bounced ball.
            display_track = ([p for p in track if p.frame_idx <= impact_idx]
                             if impact_idx is not None else track)
            annotated = ball_tracking.annotate_frames(frames, display_track)
            video_path = ball_tracking.write_mp4(
                annotated, ball_tracking.make_output_path(), fps=target_fps)
            ball_stats = ball_tracking.summarize(display_track, fps=target_fps)
            ball_stats["release_idx"] = track_stats.get("release_idx")
            ball_stats["impact_idx"] = impact_idx
            ball_stats["coverage_pct"] = (ball_stats.get("n_frames", 0) / max(1, len(frames))) * 100
            warnings.append(
                f"Ball tracking: {ball_stats['n_detected']} detected + "
                f"{ball_stats['n_interpolated']} predicted frames "
                f"({ball_stats['coverage_pct']:.0f}% of clip) -- annotated video below."
            )
            if impact_idx is not None:
                warnings.append(
                    f"Ball tracking: box stops at frame {impact_idx} (detected bat/pad/ground "
                    f"contact) so it doesn't chase the ball after it hits the bat.")
            else:
                warnings.append("Ball tracking: no clear bat/pad contact detected in this clip, "
                                "so the box follows the ball for the whole tracked segment.")
        else:
            warnings.append("Ball tracking: ball not detected reliably in this clip "
                            "(no annotated video produced).")
    except Exception as exc:
        warnings.append(f"Ball tracking skipped ({exc}).")
    timings["ball_tracking"] = time.perf_counter() - t0

    # 2-3: detection + tracking + bowler crop (best effort)
    t0 = time.perf_counter()
    crop_stats = None
    try:
        tracker = tracking.BowlerTracker()
        tracks = tracker.track_frames([(idx, fr) for idx, ts, fr in frames])
        bowler = tracking.select_bowler_track(tracks)
        if bowler is not None and len(bowler) >= 3:
            frames = _crop_frames_to_bowler(frames, tracks, bowler)
            crop_stats = {"track_id": bowler.track_id, "frames_tracked": len(bowler)}
            warnings.append(
                f"Detection/tracking: cropped to bowler track #{bowler.track_id} "
                f"({len(bowler)} frames) before pose estimation."
            )
        else:
            warnings.append("Detection/tracking found no stable bowler track; "
                            "pose estimation ran on full frames.")
    except Exception as exc:
        warnings.append(f"Detection/tracking skipped ({exc}); pose ran on full frames.")
    timings["detection_tracking"] = time.perf_counter() - t0

    # 4-5: pose estimation -> 33 landmarks/frame
    t0 = time.perf_counter()
    with pose_estimation.PoseEstimator() as estimator:
        pose_sequence = estimator.process_video_frames(iter(frames))
    timings["pose_estimation"] = time.perf_counter() - t0

    if len(pose_sequence) < 3:
        raise RuntimeError("Not enough frames with a detected pose -- check video quality/framing.")

    # 6: biomechanical feature engineering (+ delivery-quality diagnostics)
    t0 = time.perf_counter()
    feature_vector, diagnostics = feateng.analyze_delivery(
        pose_sequence, bowling_arm=bowling_arm, camera_view=camera_view)
    timings["feature_engineering"] = time.perf_counter() - t0

    if diagnostics.get("reliable") is not True:
        warnings.append(f"Delivery quality: {diagnostics.get('reliability_reason')}")
    if crop_stats is not None and diagnostics.get("n_frames", 0) < 10:
        warnings.append("Few pose frames after cropping -- consider tighter framing.")

    # 7: ML predictions
    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None

    if run_ml and performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if run_ml and injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)

    # 9: coaching recommendations
    t0 = time.perf_counter()
    notes = coaching.generate_recommendations(
        feature_vector, performance_score, injury_risk,
        shap_contributions=shap_injury or shap_perf,
    )
    if run_ml:
        timings["coaching"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start

    return AnalysisResult(
        feature_vector=feature_vector,
        performance_score=performance_score,
        injury_risk=injury_risk,
        shap_contributions_performance=shap_perf,
        shap_contributions_injury=shap_injury,
        coaching_notes=notes,
        stage_times=timings,
        warnings=warnings,
        camera_view=camera_view,
        bowling_arm=bowling_arm,
        video_path=video_path,
        ball_stats=ball_stats,
    )


def analyze_feature_vector(feature_vector: dict,
                            performance_bundle: ml_models.TrainedBundle = None,
                            injury_bundle: ml_models.TrainedBundle = None,
                            camera_view: str = "behind") -> AnalysisResult:
    """    Same as analyze_video but skips CV/pose stages -- useful for the Streamlit
    manual-entry mode and for testing without a video file."""
    timings = {}
    t_start = time.perf_counter()

    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None

    if performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)

    notes = coaching.generate_recommendations(
        feature_vector, performance_score, injury_risk,
        shap_contributions=shap_injury or shap_perf,
    )

    timings["total"] = time.perf_counter() - t_start

    return AnalysisResult(
        feature_vector=feature_vector,
        performance_score=performance_score,
        injury_risk=injury_risk,
        shap_contributions_performance=shap_perf,
        shap_contributions_injury=shap_injury,
        coaching_notes=notes,
        stage_times=timings,
        camera_view=camera_view,
    )
