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
from . import video_validity as validity
from . import analysis_replay


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
    pose_video_path: Optional[str] = None  # pose-skeleton overlay MP4 (ball-tracking fallback)
    reels_video_path: Optional[str] = None  # slow-mo + zoom MP4 (video mode)
    analysis_replay_path: Optional[str] = None  # unified Analysis Replay MP4 (hero video)
    ball_stats: dict = field(default_factory=dict)
    bowler_bboxes: Optional[dict] = None   # frame_idx -> (x1,y1,x2,y2) padded crop bbox
    original_frame_dims: Optional[tuple] = None  # (height, width) of pre-crop frames
    bowler_track_id: Optional[int] = None  # locked bowler track id (identity lock)
    bowler_confidence: Optional[float] = None  # cricket-evidence confidence 0..1
    feature_provenance: Optional[dict] = None  # per-feature source + visibility proxy
    landmark_source_summary: Optional[dict] = None  # world3d/2d/missing frame counts
    subject_verified: Optional[bool] = None  # False -> ML/coaching refused (wrong-subject risk)
    scoring_blocked_reason: Optional[str] = None  # human-readable why scoring was withheld
    stage_backends: dict = field(default_factory=dict)  # detection/tracking backend actually used
    delivery_reliable: Optional[bool] = None  # from pose-quality diagnostics (G6)
    player_roles: Optional[dict] = None  # track_id -> {role, confidence, scores} for non-bowler players
    # Bowler-identification confirmation state (Phase 12/19): when the top
    # cricket-evidence candidate is weak or tied with the runner-up, the system
    # reports BOWLER NOT CONFIRMED instead of silently analysing the wrong person.
    bowler_confirmed: Optional[bool] = None  # False -> auto-lock rejected, UI confirmation needed
    bowler_confirm_reason: Optional[str] = None  # "low_evidence" | "ambiguous_margin" | None
    bowler_candidates: list = field(default_factory=list)  # ranked candidate evidence for the UI picker
    identity_switch_count: int = 0  # half-clip windows where the top bowler candidate churned
    # Clean main-pipeline picture: BOWLER / STRIKER / NON-STRIKER (everything else ignored).
    batting_stances: Optional[dict] = None  # track_id -> {role: "striker"|"non_striker", confidence}
    striker_track_id: Optional[int] = None
    non_striker_track_id: Optional[int] = None

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


BOWLER_CARRY_GAP_FRAMES = 5  # frames a bowler bbox may be carried over a brief
                             # gap; beyond that the bowler is "gone" -- no box,
                             # no crop, no substitute (identity is not swapped).


def _crop_frames_to_bowler(frames, tracks, bowler, gap_frames: int = BOWLER_CARRY_GAP_FRAMES) -> list:
    """Return a new frame list cropped to the bowler's track bbox, carrying the
    previous bbox forward only over brief (<= gap_frames) gaps. Bboxes must be in
    frame coordinates. Frames after the bowler is lost are kept uncropped -- and
    pose estimation later runs only on the bowler-bound frames, so a missing
    bowler never silently becomes a batsman."""
    bbox_by_frame = dict(zip(bowler.frames, bowler.bboxes))
    last_bbox, last_seen = None, -10 ** 9
    cropped = []
    for idx, ts, frame in frames:
        bbox = bbox_by_frame.get(idx)
        if bbox is not None:
            last_bbox, last_seen = bbox, idx
            cut = _crop_to_bbox(frame, bbox)
            if cut is not None:
                cropped.append((idx, ts, cut))
                continue
        elif last_bbox is not None and idx - last_seen <= gap_frames:
            cut = _crop_to_bbox(frame, last_bbox)
            if cut is not None:
                cropped.append((idx, ts, cut))
                continue
        cropped.append((idx, ts, frame))
    return cropped


def _padded_bowler_bboxes(bowler, n_frames, frame_h, frame_w,
                          pad_frac: float = 0.3,
                          gap_frames: int = BOWLER_CARRY_GAP_FRAMES) -> dict:
    """Full-frame padded crop bboxes for the locked bowler track, carried over
    brief (<= gap_frames) gaps only. Frames outside the bowler's presence get
    NO bbox, so the Analysis Replay stops drawing the bowler box the moment the
    bowler leaves the frame -- it never swaps in a batsman/keeper/fielder."""
    bbox_by_frame = dict(zip(bowler.frames, bowler.bboxes))
    out = {}
    last_bb, last_seen = None, -10 ** 9
    for idx in range(n_frames):
        bb = bbox_by_frame.get(idx)
        if bb is None and (last_bb is None or idx - last_seen > gap_frames):
            continue
        if bb is None:
            bb = last_bb
        else:
            last_bb, last_seen = bb, idx
        bx1, by1, bx2, by2 = bb
        bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
        pad_x, pad_y = pad_frac * bw, pad_frac * bh
        out[idx] = (
            max(0, int(bx1 - pad_x)), max(0, int(by1 - pad_y)),
            min(frame_w, int(bx2 + pad_x)), min(frame_h, int(by2 + pad_y)),
        )
    return out


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


class CricketPrecheckError(RuntimeError):
    """Raised when a video fails the lightweight cricket-validity pre-check.

    Used to stop analysis early (before the expensive ball tracking, bowler
    cropping and full pose stages) so a non-cricket clip never wastes compute
    and never yields meaningless biomechanics numbers."""


def _precheck_cricket(frames, bowling_arm: str = "right",
                      max_frames: int = 60,
                      step: int = 1) -> validity.CheckResult:
    """Cheap, sub-sampled human-pose gate run before the heavy CV stages.

    Runs MediaPipe pose estimation on (at most) `max_frames` sampled frames
    and checks for a human torso + arms via `video_validity.check_human_pose`.
    This is deliberately subsampled and limited so the gate is much cheaper
    than the full-clip analysis it protects.
    """
    sampled = []
    n = len(frames)
    if n == 0:
        return validity.check_human_pose([])
    # Evenly sample up to `max_frames` frames across the clip.
    idxs = sorted({int(round(i)) for i in
                   np.linspace(0, n - 1, min(n, max_frames))})
    step = max(1, step)
    grabbed = []
    total = len(frames)
    for k in range(0, len(idxs), step):
        grabbed.append(frames[idxs[k]])
    if len(grabbed) < 3:
        grabbed = frames[: min(3, total)]
    with pose_estimation.PoseEstimator() as estimator:
        pose_seq = estimator.process_video_frames(iter(grabbed))
    return validity.check_human_pose(pose_seq)


def analyze_video(video_path: str, bowling_arm: str = "right",
                   performance_bundle: ml_models.TrainedBundle = None,
                   injury_bundle: ml_models.TrainedBundle = None,
                   target_fps: int = config.TARGET_FPS,
                   resize_dim=config.RESIZE_DIM,
                   denoise: bool = config.DENOISE,
                   camera_view: str = "behind",
                   slow_factor: float = 2.5,
                   zoom_end: float = 1.8,
                   run_ml: bool = True,
                   precheck: bool = True,
                   progress_cb=None,
                   debug_overlay: bool = False,
                   bowler_track_override: Optional[int] = None) -> AnalysisResult:
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
    `progress_cb(done: int, total: int, label: str)` is invoked as each major
    stage completes, letting a UI render a live staged checklist.
    `debug_overlay=True` adds a diagnostic text overlay (locked bowler track id +
    cricket-evidence confidence, ball state) to the Analysis Replay.
    """
    timings = {}
    warnings = []
    t_start = time.perf_counter()
    _stages = ["Video loaded", "Cricket pre-check", "Ball detection", "Bowler detection",
               "Pose extraction", "Ball tracking", "Biomechanics", "ML analysis", "Complete"]
    _done = 0

    def _progress(label):
        nonlocal _done
        _done += 1
        if progress_cb is not None:
            progress_cb(_done, len(_stages), label)

    # 1: preprocess
    t0 = time.perf_counter()
    frames = list(preprocessing.preprocess_video(video_path, target_fps=target_fps,
                                                  resize_dim=resize_dim, denoise=denoise))
    timings["preprocess"] = time.perf_counter() - t0
    _progress("Video loaded")

    # 1c: lightweight cricket-validity pre-check (hard gate). Runs a cheap,
    # sub-sampled human-pose pass BEFORE the expensive ball tracking, bowler
    # detection/cropping and full pose stages, so a non-cricket clip fails fast
    # instead of wasting compute or producing meaningless numbers.
    if not frames:
        raise CricketPrecheckError("Video produced no frames to analyse.")
    if precheck:
        t0 = time.perf_counter()
        pre_ok = _precheck_cricket(frames, bowling_arm=bowling_arm)
        timings["cricket_precheck"] = time.perf_counter() - t0
        if not pre_ok.ok:
            raise CricketPrecheckError(
                "This clip does not appear to be a cricket bowling video: "
                f"{pre_ok.reason} Analysis stopped before the CV stages.")
    _progress("Cricket pre-check")

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
            ball_stats["total_frames"] = int(track_stats.get("total_frames") or len(frames))
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
    _progress("Ball detection")

    # Fallback: display_track = track (will be overridden by wrist-proxy if applied)
    if display_track is None:
        display_track = track

    # 2-3: detection + tracking + bowler crop (best effort)
    t0 = time.perf_counter()
    crop_stats = None
    bowler_bboxes = None
    original_frame_dims = None
    frames_full = None
    bowler_track_id = None
    bowler_confidence = None
    stage_backends = {}
    player_roles = None
    bowler_confirmed = None
    bowler_confirm_reason = None
    bowler_candidates = []
    identity_switch_count = 0
    batting_stances = None
    striker_track_id = None
    non_striker_track_id = None
    try:
        # Snapshot full-frame dimensions before any cropping
        if frames:
            original_frame_dims = (frames[0][2].shape[0], frames[0][2].shape[1])
            frames_full = list(frames)
        tracker = tracking.BowlerTracker()
        stage_backends = {
            "tracking": tracker.backend,
            "detection": getattr(getattr(tracker, "detector", None), "backend",
                                 "bytetrack"),
        }
        if getattr(tracker, "fallback_reason", None):
            warnings.append(f"Detection/tracking degraded: {tracker.fallback_reason}")
        tracks = tracker.track_frames([(idx, fr) for idx, ts, fr in frames])
        h, w = frames[0][2].shape[:2]
        # Appearance-grade the identity stitch (G3): the tracker hands the
        # selector a way to read real pre-crop pixels so a continuation that
        # merely happens to be spatially continuous is rejected if its colour
        # profile does not match the locked bowler.
        frames_by_idx = {idx: fr for idx, ts, fr in (frames_full or [])}

        def _frame_provider(frame_idx, bbox):
            fr = frames_by_idx.get(frame_idx)
            if fr is None:
                return None
            return fr

        if bowler_track_override is not None and bowler_track_override in tracks:
            # User-confirmed lock (Phase 19): analyse the chosen track directly.
            bowler = tracks[bowler_track_override]
            bowler_meta = {
                "track_id": bowler.track_id,
                "score": None,
                "confidence": None,
                "confirmed": True,
                "confirm_reason": "user_override",
                "candidates": [],
                "identity_switch_count": 0,
            }
            warnings.append(
                f"Bowler track #{bowler_track_override} locked by user confirmation."
            )
        else:
            bowler, bowler_meta = tracking.select_bowler_track_with_meta(
                tracks, frame_dims=(h, w), total_frames=len(frames),
                frame_provider=_frame_provider)
        if bowler is not None and len(bowler) >= 3:
            bowler_bboxes = _padded_bowler_bboxes(
                bowler, len(frames), h, w,
                gap_frames=BOWLER_CARRY_GAP_FRAMES)
            frames = _crop_frames_to_bowler(frames, tracks, bowler)
            crop_stats = {"track_id": bowler.track_id, "frames_tracked": len(bowler)}
            bowler_track_id = bowler_meta["track_id"] if bowler_meta else bowler.track_id
            bowler_confidence = bowler_meta.get("confidence") if bowler_meta else None
            bowler_confirmed = bool(bowler_meta.get("confirmed", True))
            bowler_confirm_reason = bowler_meta.get("confirm_reason")
            bowler_candidates = list(
                bowler_meta.get("candidates") or []) if bowler_meta else []
            identity_switch_count = int(
                bowler_meta.get("identity_switch_count", 0)) if bowler_meta else 0
            conf_txt = f"{bowler_confidence:.2f}" if bowler_confidence is not None else "n/a"
            warnings.append(
                f"Detection/tracking: locked to bowler track #{bowler_track_id} "
                f"({len(bowler)} frames, confidence {conf_txt}) before pose estimation."
            )
            if not bowler_confirmed:
                warnings.append(
                    "BOWLER NOT CONFIRMED: the cricket-evidence score is "
                    f"{bowler_confirm_reason or 'weak'}. ML/coaching are withheld "
                    "until the bowler is confirmed in the UI."
                )
            if identity_switch_count > 0:
                warnings.append(
                    f"Identity check: the top bowler candidate switched "
                    f"{identity_switch_count}x across the clip (bowler <-> batsman "
                    "churn observed)."
                )
            # Classify remaining tracks into cricket roles (batsman, keeper, umpire)
            player_roles = tracking.classify_player_roles(
                tracks, bowler.track_id, frame_dims=(h, w), total_frames=len(frames))
            if player_roles:
                role_summary = ", ".join(
                    f"#{tid}: {r['role']}({r['confidence']:.2f})"
                    for tid, r in player_roles.items())
                warnings.append(f"Player roles: {role_summary}")
            # Clean main-pipeline picture: bowler + striker + non-striker
            batting_stances = tracking.classify_batting_stances(
                tracks, bowler, frame_dims=(h, w), total_frames=len(frames))
            for tid, st in (batting_stances or {}).items():
                if st["role"] == tracking.ROLE_STRIKER:
                    striker_track_id = tid
                elif st["role"] == tracking.ROLE_NON_STRIKER:
                    non_striker_track_id = tid
            if batting_stances and bowler_bboxes:
                stance_summary = ", ".join(
                    f"#{tid}: {st['role']}({st['confidence']:.2f})"
                    for tid, st in batting_stances.items())
                warnings.append(f"Batting stances: {stance_summary}")
        else:
            bowler_track_id = None
            bowler_confidence = None
            warnings.append(
                "Detection/tracking found no bowler: nobody in the clip moves like a "
                "bowler (run-up + delivery). Identity is NOT inferred -- pose ran on "
                "full frames without a bowler box."
            )
    except Exception as exc:
        warnings.append(f"Detection/tracking skipped ({exc}); pose ran on full frames.")
        bowler_track_id = None
        bowler_confidence = None
        stage_backends = {"error": f"{type(exc).__name__}: {exc}"}
    timings["detection_tracking"] = time.perf_counter() - t0

    # 4-5: pose estimation -> 33 landmarks/frame.
    #
    # Preferred (identity-lock) path: when a bowler crop exists, ONLY the
    # bowler's own frames are fed to the pose model (a missing bowler never
    # becomes a batsman). If that snapshot is too weak (tiny/fast crop,
    # occlusion) the pipeline DEGRADES instead of failing:
    #   level 1 -> all cropped frames (matches the pre-crop-clip behaviour)
    #   level 2 -> original pre-crop FULL frames (never needed when the cricket
    #              pre-check already proved a visible human pose, but kept so a
    #              genuinely hard clip still analyses instead of erroring).
    # `pose_source` records which coordinate space the landmarks live in so the
    # wrist-proxy / overlay / replay consumers map pixels correctly.
    attempts = []
    if bowler_bboxes:
        attempts.append(("bowler_crops", [
            (idx, ts, fr) for idx, ts, fr in frames if idx in bowler_bboxes]))
    attempts.append(("all_frames", list(frames)))
    if frames_full is not None:
        attempts.append(("full_frames", list(frames_full)))
    pose_source = None
    pose_sequence = []
    for name, src in attempts:
        t0 = time.perf_counter()
        with pose_estimation.PoseEstimator() as estimator:
            pose_sequence = estimator.process_video_frames(iter(src))
        if name == "bowler_crops":
            timings["pose_estimation"] = time.perf_counter() - t0
        else:
            timings[f"pose_fallback_{name}"] = time.perf_counter() - t0
        if len(pose_sequence) >= 3:
            pose_source = name
            break
        pose_sequence = []

    if pose_source is None:
        # No usable person pose in ANY coordinate space: genuinely not a usable
        # bowling video (blurry, cut mid-run-up, framed too wide, no person).
        counts = ", ".join(f"{n}: {len(s)}" for n, s in attempts)
        raise RuntimeError(
            "No usable pose found in this clip "
            f"(pose frames per fallback level -> {counts}). The video may be "
            "blurry, cut mid-run-up, or framed too wide. Please try a clip "
            "where the bowler is clearly visible for about a second."
        )
    if pose_source != "bowler_crops":
        warnings.append(
            "Pose estimation: bowler-bound pose was too weak (<3 frames); "
            f"fell back to {pose_source.replace('_', ' ')}. If the skeleton "
            "looks loose, enable the debug overlay to inspect the bowler lock."
        )
    # Subject-verification gate (G1): only the identity-locked bowler-crop path
    # guarantees the skeleton belongs to the bowler. The FULL-FRAME fallback has
    # no bowler box at all, so we check whether the pose model saw more than one
    # person in any frame -- if it did, the measured features may be from a
    # batsman/keeper/fielder and ML/coaching must NOT run (the "all_frames" path
    # is still bowler-bound because those frames are already bowler crops).
    subject_verified = True
    scoring_blocked_reason = None
    if bowler_track_id is not None and bowler_confirmed is False:
        # Fail-safe (Phase 19): a selected-but-unconfirmed bowler (low evidence
        # or a tight bowler<->batsman margin) is never silently scored. The UI
        # asks the user to lock a candidate; ML/coaching stay refused until then.
        subject_verified = False
        scoring_blocked_reason = (
            "bowler not confirmed (cricket-evidence "
            f"{bowler_confirm_reason or 'weak'}); ML/coaching withheld until the "
            "bowler is confirmed in the UI."
        )
    elif pose_source == "full_frames":
        pose_frames = sum(1 for pf in pose_sequence if pf.n_people > 1)
        if pose_frames > 0:
            subject_verified = False
            scoring_blocked_reason = (
                "pose detected multiple people in the frame (no bowler lock); the "
                "measured skeleton may not be the bowler."
            )
        else:
            warnings.append(
                "Subject check: full-frame fallback saw a single person throughout, "
                "so the skeleton is treated as the bowler."
            )
    elif pose_source != "bowler_crops":
        warnings.append(
            "Subject check: pose ran on bowler-cropped frames (media extracted "
            "before the missing identity-check step)."
        )
    if scoring_blocked_reason is not None:
        warnings.append(
            "Scoring withheld — subject not verified: "
            f"{scoring_blocked_reason}"
        )
    _progress("Bowler detection")
    _progress("Pose extraction")

    # 6: biomechanical feature engineering (+ delivery-quality diagnostics)
    t0 = time.perf_counter()
    feature_vector, diagnostics = feateng.analyze_delivery(
        pose_sequence, bowling_arm=bowling_arm, camera_view=camera_view)
    timings["feature_engineering"] = time.perf_counter() - t0
    _progress("Biomechanics")

    if diagnostics.get("reliable") is not True:
        warnings.append(f"Delivery quality: {diagnostics.get('reliability_reason')}")
    delivery_reliable = diagnostics.get("reliable") is True
    if crop_stats is not None and diagnostics.get("n_frames", 0) < 10:
        warnings.append("Few pose frames after cropping -- consider tighter framing.")

    # 6b: wrist-proxy pre-release ball trajectory augmentation
    if (track is not None and bowler_bboxes is not None
            and pose_source != "full_frames"
            and original_frame_dims is not None and pose_sequence):
        release_frame = diagnostics.get("release_frame_idx")
        wrist_pos = _extract_wrist_pixel_positions(
            pose_sequence, bowling_arm, bowler_bboxes, original_frame_dims)
        if wrist_pos and release_frame is not None:
            track = _augment_trajectory_with_wrist_proxy(
                track, wrist_pos, release_frame, handoff_frames=3)
            # Re-clip at impact and update stats (single video output)
            impact_idx = track_stats.get("impact_idx")
            display_track = ([p for p in track if p.frame_idx <= impact_idx]
                             if impact_idx is not None else track)
            ball_stats = ball_tracking.summarize(display_track, fps=target_fps)
            ball_stats["release_idx"] = track_stats.get("release_idx")
            ball_stats["impact_idx"] = impact_idx
            ball_stats["outcome"] = track_stats.get("outcome")
            ball_stats["total_frames"] = int(track_stats.get("total_frames") or len(frames))
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
        if pose_source == "full_frames":
            missing.append("full-frame pose (weak bowler lock)")
        if original_frame_dims is None:
            missing.append("no frame dimensions")
        if not pose_sequence:
            missing.append("no pose data")
        warnings.append(f"Wrist-proxy skipped: {', '.join(missing)}.")

    # 6c: pose-skeleton overlay video (used as the ball-tracking fallback, and
    # as an always-available "show the AI working" clip). Drawn on the frames
    # the pose was actually estimated on: bowler crops for crop-space landmarks,
    # the original full frames when pose fell back to full-frame space.
    pose_video_path = None
    try:
        pose_by_idx = {pf.frame_idx: pf for pf in pose_sequence}
        overlay_frames = frames if pose_source != "full_frames" else (frames_full or frames)
        overlays = []
        for idx, ts, fr in overlay_frames:
            pf = pose_by_idx.get(idx)
            img = fr if pf is None else pose_estimation.draw_skeleton(fr, pf)
            overlays.append((idx, ts, img))
        if overlays:
            pose_video_path = ball_tracking.write_mp4(
                overlays, ball_tracking.make_output_path("pose"), fps=target_fps)
    except Exception as exc:
        warnings.append(f"Pose-overlay video skipped ({exc}).")
    if video_path is None and pose_video_path is not None:
        warnings.append("Ball not detected reliably; showing pose-skeleton overlay video.")
    _progress("Ball tracking")

    # 7: ML predictions. Refused when the pose subject could not be verified as
    # the bowler (wrong-subject risk), the delivery window was unreliable
    # (G6), or the caller disabled the ML pass.
    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None
    runs_ml = (
        run_ml
        and (config.SUBJECT_VERIFICATION_REQUIRED is False or subject_verified is True)
        and (config.REFUSE_ML_ON_UNRELIABLE_DELIVERY is False or delivery_reliable is True)
    )
    if not runs_ml and run_ml and subject_verified is False:
        warnings.append(
            "ML analysis skipped: scoring a skeleton that may not be the bowler "
            "would produce misleading numbers. Features are still shown, clearly "
            "marked as from an unverified subject."
        )
    if not runs_ml and run_ml and delivery_reliable is False:
        warnings.append(
            "ML analysis skipped: the delivery window was unreliable (no clear "
            "release detected) so the interpolated features would produce "
            "misleading predictions."
        )

    if runs_ml and performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if runs_ml and injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)
    _progress("ML analysis")

    # 9: coaching recommendations
    t0 = time.perf_counter()
    if subject_verified is True and (delivery_reliable is not False or config.REFUSE_ML_ON_UNRELIABLE_DELIVERY is False):
        notes = coaching.generate_recommendations(
            feature_vector, performance_score, injury_risk,
            shap_contributions=shap_injury or shap_perf,
        )
    else:
        reason = (
            "pose subject could not be verified as the bowler"
            if subject_verified is False else
            "the delivery window was unreliable (no clear release detected)"
        )
        notes = [f"Coaching withheld — {reason}."]
    if run_ml:
        timings["coaching"] = time.perf_counter() - t0

    # 10: slow-mo + progressive zoom-toward-the-ball (reels effect).
    # Pure post-processing rendering on the already-tracked annotated video --
    # it does NOT touch detection, tracking, pose, or any ML stage.
    reels_video_path = None
    try:
        traj = (ball_stats or {}).get("trajectory") or []
        if video_path and traj and os.path.exists(video_path):
            reels_video_path = ball_tracking.make_output_path("reels")
            ball_tracking.render_slowmo_zoom(
                video_path,
                reels_video_path,
                traj,
                slow_factor=slow_factor,
                zoom_end=zoom_end,
                output_fps=target_fps,
            )
    except Exception as exc:
        reels_video_path = None
        warnings.append(f"Reels (slow-mo/zoom) rendering skipped ({exc}).")

    # 11: unified Analysis Replay -- ONE hero video combining original footage
    # with real, frame-synchronized overlays (ball box, trajectory, pose
    # skeleton, release marker). Pure post-processing on real pipeline outputs;
    # it does NOT change any detection/tracking/pose/ML result.
    analysis_replay_path = None
    try:
        if frames_full:
            release_frame = (ball_stats or {}).get("release_idx")
            if release_frame is None:
                release_frame = diagnostics.get("release_frame_idx")
            analysis_replay_path = analysis_replay.render_analysis_replay(
                frames_full,
                display_track or [],
                pose_sequence,
                bowler_bboxes or {},
                release_frame,
                ball_tracking.make_output_path("analysis_replay"),
                fps=float(target_fps) / slow_factor,
                frame_dims=original_frame_dims,
                debug=debug_overlay,
                bowler_track_id=bowler_track_id,
                bowler_confidence=bowler_confidence,
                pose_in_full_frame=pose_source == "full_frames",
                player_roles=player_roles,
                all_tracks=tracks,
            )
            warnings.append(f"Analysis Replay generated: {analysis_replay_path}")
        else:
            warnings.append(
                "Analysis Replay skipped: no full frames available "
                "(detection/tracking may have failed before frames were captured)."
            )
    except Exception as exc:
        warnings.append(f"Analysis Replay rendering failed ({type(exc).__name__}: {exc}).")
        # Fallback: try a minimal replay without overlays so the user still
        # gets a playable video of the original footage.
        try:
            if frames_full:
                analysis_replay_path = analysis_replay.render_analysis_replay(
                    frames_full, [], [], {}, None,
                    ball_tracking.make_output_path("analysis_replay_fallback"),
                    fps=float(target_fps) / slow_factor,
                    frame_dims=original_frame_dims,
                )
                warnings.append(
                    f"Analysis Replay fallback (no overlays) generated: "
                    f"{analysis_replay_path}"
                )
        except Exception:
            warnings.append("Analysis Replay fallback also failed; no video will be shown.")

    timings["total"] = time.perf_counter() - t_start
    _progress("Complete")

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
        analysis_replay_path=analysis_replay_path,
        ball_stats=ball_stats,
        bowler_bboxes=bowler_bboxes,
        original_frame_dims=original_frame_dims,
        bowler_track_id=bowler_track_id,
        bowler_confidence=bowler_confidence,
        pose_video_path=pose_video_path,
        feature_provenance=diagnostics.get("feature_provenance"),
        landmark_source_summary=diagnostics.get("landmark_source_summary"),
        subject_verified=subject_verified,
        scoring_blocked_reason=scoring_blocked_reason,
        stage_backends=stage_backends,
        delivery_reliable=delivery_reliable,
        player_roles=player_roles,
        bowler_confirmed=bowler_confirmed,
        bowler_confirm_reason=bowler_confirm_reason,
        bowler_candidates=bowler_candidates,
        identity_switch_count=identity_switch_count,
        batting_stances=batting_stances,
        striker_track_id=striker_track_id,
        non_striker_track_id=non_striker_track_id,
    )


def analyze_feature_vector(feature_vector: dict,
                            performance_bundle: ml_models.TrainedBundle = None,
                            injury_bundle: ml_models.TrainedBundle = None,
                            camera_view: str = "behind",
                            subject_verified: Optional[bool] = None,
                            reliable: Optional[bool] = None) -> AnalysisResult:
    """    Same as analyze_video but skips CV/pose stages -- useful for the Streamlit
    manual-entry mode and for testing without a video file.

    Parameters
    ----------
    subject_verified : bool or None
        Passed through from the video pipeline. When ``False``, ML predictions,
        SHAP explanations and coaching notes are **refused** to prevent scoring
        a skeleton that may not be the bowler (the subject-verification gate).
        ``None`` or ``True`` means scoring proceeds (manual-entry mode has no
        video to verify, so it always scores).
    reliable : bool or None
        Passed through from the video pipeline. When ``False`` AND
        ``config.REFUSE_ML_ON_UNRELIABLE_DELIVERY`` is on, ML/SHAP/coaching are
        refused because the delivery window was noisy/short (G6).
    """
    timings = {}
    t_start = time.perf_counter()

    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None
    scoring_blocked_reason = None
    runs_ml = config.SUBJECT_VERIFICATION_REQUIRED is False or subject_verified is not False
    if config.REFUSE_ML_ON_UNRELIABLE_DELIVERY and reliable is False:
        runs_ml = False
    if not runs_ml and subject_verified is False:
        scoring_blocked_reason = (
            "pose detected multiple people in the frame (no bowler lock); the "
            "measured skeleton may not be the bowler."
        )
    elif not runs_ml and reliable is False:
        scoring_blocked_reason = (
            "the delivery window was unreliable (no clear release detected); "
            "the interpolated features would produce misleading predictions."
        )

    if runs_ml and performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if runs_ml and injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)

    if not runs_ml and scoring_blocked_reason is not None:
        notes = [f"Coaching withheld — {scoring_blocked_reason}"]
    else:
        notes = coaching.generate_recommendations(
            feature_vector, performance_score, injury_risk,
            shap_contributions=shap_injury or shap_perf,
        )

    timings["total"] = time.perf_counter() - t_start

    # Manual slider mode: every feature is user-entered, NOT measured from video.
    manual_provenance = {
        name: {
            "source": "manual_entry",
            "confidence": "unknown",
            "mean_landmark_visibility": None,
            "confidence_note": "user-entered slider value; not measured from video",
        }
        for name in feature_vector
    }

    return AnalysisResult(
        feature_vector=feature_vector,
        performance_score=performance_score,
        injury_risk=injury_risk,
        shap_contributions_performance=shap_perf,
        shap_contributions_injury=shap_injury,
        coaching_notes=notes,
        stage_times=timings,
        camera_view=camera_view,
        feature_provenance=manual_provenance,
        landmark_source_summary={"total_frames": 0, "world_3d_frames": 0,
                                 "normalized_2d_frames": 0, "raw_array_frames": 0,
                                 "missing_frames": 0},
        subject_verified=subject_verified,
        scoring_blocked_reason=scoring_blocked_reason,
        delivery_reliable=reliable,
    )
