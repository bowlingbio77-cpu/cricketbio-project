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
    reels_video_path: Optional[str] = None  # slow-mo + zoom MP4 (video mode)
    ball_stats: dict = field(default_factory=dict)
    bowler_bboxes: Optional[dict] = None   # frame_idx -> (x1,y1,x2,y2) padded crop bbox
    original_frame_dims: Optional[tuple] = None  # (height, width) of pre-crop frames

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


# --------------------------------------------------------------------------- #
# Wrist-proxy pre-release ball tracking
# --------------------------------------------------------------------------- #

def _extract_wrist_pixel_positions(pose_sequence, bowling_arm, bowler_bboxes,
                                    frame_dims):
    """Convert per-frame wrist landmarks from cropped-normalised (0-1) to
    full-frame pixel coordinates.

    Parameters
    ----------
    pose_sequence : list[PoseFrame]
        Pose landmarks (cropped-frame normalised coords).
    bowling_arm : str
        "right" or "left".
    bowler_bboxes : dict
        frame_idx -> (x1, y1, x2, y2) padded crop bbox in full-frame coords.
    frame_dims : tuple
        (height, width) of the original full-frame.

    Returns
    -------
    dict  frame_idx -> (px_x, px_y)  pixel coords in the full frame.
    """
    from . import config as _cfg
    wrist_name = "right_wrist" if bowling_arm == "right" else "left_wrist"
    wrist_idx = _cfg.POSE_LANDMARK_NAMES.index(wrist_name)   # 16 or 15
    full_h, full_w = frame_dims
    positions = {}
    for pf in pose_sequence:
        bbox = bowler_bboxes.get(pf.frame_idx)
        if bbox is None:
            continue
        bx1, by1, bx2, by2 = bbox
        crop_w = max(1.0, bx2 - bx1)
        crop_h = max(1.0, by2 - by1)
        nx = float(pf.landmarks[wrist_idx, 0])   # 0-1 normalised in crop
        ny = float(pf.landmarks[wrist_idx, 1])
        px = bx1 + nx * crop_w
        py = by1 + ny * crop_h
        # Clamp to frame bounds
        px = max(0.0, min(float(full_w - 1), px))
        py = max(0.0, min(float(full_h - 1), py))
        positions[pf.frame_idx] = (px, py)
    return positions


def _augment_trajectory_with_wrist_proxy(trajectory, wrist_positions,
                                          release_frame, handoff_frames=3):
    """Merge wrist-proxy points into the ball trajectory for pre-release frames.

    For frames *before* the release where no YOLO ball detection exists,
    the bowling wrist is a physically accurate stand-in (the ball is in hand
    and travels with the wrist).  At the release point a short linear blend
    avoids a visible jump between the proxy and the real tracker.

    Parameters
    ----------
    trajectory : list[BallPoint]
        Existing ball-tracking trajectory (may start at or after release).
    wrist_positions : dict
        frame_idx -> (x, y) pixel coords from _extract_wrist_pixel_positions.
    release_frame : int or None
        Index of the release frame (from feature engineering).
    handoff_frames : int
        Number of frames over which to blend wrist -> ball at release.

    Returns
    -------
    list[BallPoint]  Merged trajectory (sorted by frame_idx).
    """
    if not wrist_positions or release_frame is None:
        return trajectory

    bt_mod = __import__("src.ball_tracking_v2", fromlist=["BallPoint"])
    BallPoint = bt_mod.BallPoint

    existing = {p.frame_idx: p for p in trajectory}
    merged = {}

    # --- wrist-proxy for pre-release frames --------------------------------
    for fidx, (wx, wy) in wrist_positions.items():
        if fidx > release_frame:
            continue
        # EMA smoothing of wrist position (alpha=0.5 across consecutive frames)
        prev = merged.get(fidx - 1)
        if prev and prev.source == "wrist_proxy":
            sx = 0.5 * prev.x + 0.5 * wx
            sy = 0.5 * prev.y + 0.5 * wy
        else:
            sx, sy = wx, wy
        merged[fidx] = BallPoint(
            frame_idx=fidx, timestamp_sec=0.0,
            x=sx, y=sy, confidence=0.0, detected=False,
            w=12.0, h=12.0, source="wrist_proxy",
        )

    # --- handoff blending at release ± handoff_frames ----------------------
    blend_start = max(0, release_frame - handoff_frames)
    blend_end = release_frame + handoff_frames
    for fidx in range(blend_start, blend_end + 1):
        wp = merged.get(fidx)
        bp = existing.get(fidx)
        if wp is None and bp is None:
            continue
        if bp is not None and wp is None:
            # Ball tracker already has this frame; keep it
            merged[fidx] = bp
            continue
        if bp is None:
            # Only wrist proxy available (pre-release or gap)
            continue
        # Both exist: linear blend based on proximity to release_frame
        alpha = max(0.0, min(1.0, (fidx - blend_start) / max(1, blend_end - blend_start)))
        # alpha=0 at blend_start -> mostly wrist; alpha=1 at blend_end -> mostly ball
        bx = (1.0 - alpha) * wp.x + alpha * bp.x
        by = (1.0 - alpha) * wp.y + alpha * bp.y
        merged[fidx] = BallPoint(
            frame_idx=fidx, timestamp_sec=bp.timestamp_sec,
            x=bx, y=by,
            confidence=max(bp.confidence, 0.1),
            detected=bp.detected,
            w=max(bp.w, wp.w), h=max(bp.h, wp.h),
            source="blended" if 0.1 < alpha < 0.9 else bp.source,
        )

    # --- fill in remaining ball-tracking points (post-release) -------------
    for fidx, bp in existing.items():
        if fidx not in merged:
            merged[fidx] = bp

    return sorted(merged.values(), key=lambda p: p.frame_idx)


def analyze_video(video_path: str, bowling_arm: str = "right",
                   performance_bundle: ml_models.TrainedBundle = None,
                   injury_bundle: ml_models.TrainedBundle = None,
                   target_fps: int = config.TARGET_FPS,
                   resize_dim=config.RESIZE_DIM,
                   denoise: bool = config.DENOISE,
                   camera_view: str = "behind",
                   slow_factor: float = 2.5,
                   zoom_end: float = 1.8,
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
    track = None
    display_track = None
    track_stats = {}
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
            ball_stats["outcome"] = track_stats.get("outcome")
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

    # Fallback: display_track = track (will be overridden by wrist-proxy if applied)
    if display_track is None:
        display_track = track

    # 2-3: detection + tracking + bowler crop (best effort)
    t0 = time.perf_counter()
    crop_stats = None
    bowler_bboxes = None
    original_frame_dims = None
    try:
        # Snapshot full-frame dimensions before any cropping
        if frames:
            original_frame_dims = (frames[0][2].shape[0], frames[0][2].shape[1])
        tracker = tracking.BowlerTracker()
        tracks = tracker.track_frames([(idx, fr) for idx, ts, fr in frames])
        bowler = tracking.select_bowler_track(tracks)
        if bowler is not None and len(bowler) >= 3:
            # Reconstruct padded bboxes (same logic as _crop_to_bbox)
            bbox_by_frame = dict(zip(bowler.frames, bowler.bboxes))
            padded = {}
            h, w = frames[0][2].shape[:2]
            last_bb = None
            for idx, _ts, _fr in frames:
                bb = bbox_by_frame.get(idx, last_bb)
                if bb is not None:
                    last_bb = bb
                    bx1, by1, bx2, by2 = bb
                    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
                    pad_x, pad_y = 0.3 * bw, 0.3 * bh
                    padded[idx] = (
                        max(0, int(bx1 - pad_x)), max(0, int(by1 - pad_y)),
                        min(w, int(bx2 + pad_x)), min(h, int(by2 + pad_y)),
                    )
            bowler_bboxes = padded
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

    # 6b: wrist-proxy pre-release ball trajectory augmentation
    if (track is not None and bowler_bboxes is not None
            and original_frame_dims is not None and pose_sequence):
        release_frame = diagnostics.get("release_frame_idx")
        wrist_pos = _extract_wrist_pixel_positions(
            pose_sequence, bowling_arm, bowler_bboxes, original_frame_dims)
        if wrist_pos and release_frame is not None:
            track = _augment_trajectory_with_wrist_proxy(
                track, wrist_pos, release_frame, handoff_frames=3)
            # Re-clip at impact and re-annotate the video
            impact_idx = track_stats.get("impact_idx")
            display_track = ([p for p in track if p.frame_idx <= impact_idx]
                             if impact_idx is not None else track)
            annotated = ball_tracking.annotate_frames(frames, display_track)
            video_path = ball_tracking.write_mp4(
                annotated, ball_tracking.make_output_path(), fps=target_fps)
            ball_stats = ball_tracking.summarize(display_track, fps=target_fps)
            ball_stats["release_idx"] = track_stats.get("release_idx")
            ball_stats["impact_idx"] = impact_idx
            ball_stats["outcome"] = track_stats.get("outcome")
            ball_stats["coverage_pct"] = (
                ball_stats.get("n_frames", 0) / max(1, len(frames))) * 100
            wrist_count = sum(1 for p in display_track if p.source == "wrist_proxy")
            if wrist_count:
                warnings.append(
                    f"Wrist proxy: {wrist_count} pre-release frames augmented "
                    f"from bowling-arm wrist landmark.")
                ball_stats["wrist_proxy_frames"] = wrist_count
        else:
            warnings.append("Wrist-proxy skipped: no wrist landmarks or release frame detected.")
    elif track is not None:
        missing = []
        if bowler_bboxes is None:
            missing.append("no bowler crop (detection needed)")
        if original_frame_dims is None:
            missing.append("no frame dimensions")
        if not pose_sequence:
            missing.append("no pose data")
        warnings.append(f"Wrist-proxy skipped: {', '.join(missing)}.")

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

    # 10: slow-mo + zoom post-processing (reels effect)
    reels_video_path = None
    if track and video_path is not None and display_track:
        try:
            reels_video_path = ball_tracking.make_output_path(prefix="reels_")
            render_stats = ball_tracking.render_slowmo_zoom(
                video_path, reels_video_path, display_track,
                slow_factor=slow_factor, zoom_end=zoom_end)
            ball_stats["has_reels"] = True
            ball_stats["reels_stats"] = render_stats
            warnings.append("Slow-motion + zoom replay generated (see below).")
        except Exception as exc:
            warnings.append(f"Reels effect skipped ({exc}).")
            reels_video_path = None

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
        reels_video_path=reels_video_path,
        ball_stats=ball_stats,
        bowler_bboxes=bowler_bboxes,
        original_frame_dims=original_frame_dims,
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
