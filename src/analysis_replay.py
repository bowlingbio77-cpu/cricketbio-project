"""
PaceAI ‑ Unified "Analysis Replay" renderer.

Turns the EXTRA pipeline outputs (full-frame videos) into ONE synchronized,
premium analysis video. This module only *consumes* the real outputs of
pipeline.analyze_video():

    * full-frame frames           -> original footage
    * ball tracking trajectory    -> ball box + path            (full-frame px)
    * pose landmarks              -> skeleton                   (crop-normalized
                                                                  -> full-frame px)
    * bowler crop bboxes          -> crop->full landmark mapping
    * release frame               -> release-point marker
    * ball stats                  -> real frame / speed info

It does NOT run, duplicate, or fabricate any detection, tracking, pose, or ML
stage. Every overlay comes from a real pipeline value for the exact frame being
drawn. If a piece of data is missing/weak for a frame, that overlay is simply
omitted for that frame.

All coordinates must be full-frame pixel coordinates; pose landmarks are mapped
from the bowler-crop normalized space into the full frame using the real crop
bbox for the same frame (same mapping as pipeline._extract_wrist_pixel_positions).
"""
import os
import cv2
import numpy as np

from . import config
from .pose_estimation import _SKELETON_CONNECTIONS, _SKELETON_POINTS
from . import ball_tracking_v2 as bt

# Accent palette (BGR), matching the app's restrained dark theme.
_BALL_DETECTED = (0, 0, 255)          # red ball box -- used for ALL ball sources in the
                                      # replay (YOLO/wrist-proxy/blended). The replay
                                      # deliberately uses ONE ball color so the ball is
                                      # unambiguous; the cyan/yellow source variants are
                                      # internal to ball_tracking_v2 only.
_BALL_PRED = (0, 165, 255)            # amber for interpolated/predicted (dashed)
_BOWLER_BOX = (80, 210, 255)          # orange bowler crop/identity box
_BONE = (0, 204, 255)                 # cyan
_JOINT = (255, 80, 80)                # red
_RELEASE = (0, 230, 118)              # green release marker
_TRAJ = (0, 230, 118)                 # trajectory path
_HEADER_BG = (14, 18, 26)
_HEADER_TEXT = (240, 246, 252)
_DEBUG_TEXT = (255, 220, 120)         # warm amber for debug diagnostics


def _clamp_pt(x, y, w, h):
    x = max(0, min(w - 1, int(round(x))))
    y = max(0, min(h - 1, int(round(y))))
    return x, y


def _landmarks_to_full_pixels(pose_frame, bowler_bboxes, frame_dims,
                              min_visibility=0.4):
    """Map crop-normalized MediaPipe landmarks to full-frame pixel coords.

    Returns dict landmark_idx -> (px, py). Weak/out-of-bbox landmarks omitted.
    """
    bbox = bowler_bboxes.get(pose_frame.frame_idx)
    if bbox is None:
        return {}
    bx1, by1, bx2, by2 = bbox
    crop_w = max(1.0, bx2 - bx1)
    crop_h = max(1.0, by2 - by1)
    fh, fw = frame_dims
    out = {}
    lm = pose_frame.landmarks
    for i in range(min(lm.shape[0], len(config.POSE_LANDMARK_NAMES))):
        x, y, z, vis = lm[i]
        if vis < min_visibility:
            continue
        px = bx1 + float(x) * crop_w
        py = by1 + float(y) * crop_h
        px, py = _clamp_pt(px, py, fw, fh)
        out[i] = (px, py)
    return out


def _draw_pose(img, pts):
    """Draw skeleton bones + joints from a landmark_idx -> (px,py) dict."""
    for a, b in _SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], _BONE, 2, cv2.LINE_AA)
    for i in _SKELETON_POINTS:
        if i in pts:
            cv2.circle(img, pts[i], 4, _JOINT, -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], 4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_ball_box(img, pt, frame_h, frame_w, debug=False):
    """Draw the real ball box. ONE color for every *real* ball source -- red for
    detected AND for wrist-proxy/blended pre-release points -- and an amber
    dashed box for interpolated/predicted points. Labels stay minimal: "ball"
    (or "ball (est)"); the confidence is only shown in debug mode."""
    w_box = pt.w if pt.w > 0 else bt.DEFAULT_BALL_DIAMETER_PX
    h_box = pt.h if pt.h > 0 else bt.DEFAULT_BALL_DIAMETER_PX
    x1 = int(round(pt.x - w_box / 2))
    y1 = int(round(pt.y - h_box / 2))
    x2 = int(round(pt.x + w_box / 2))
    y2 = int(round(pt.y + h_box / 2))
    x1, y1 = _clamp_pt(x1, y1, frame_w, frame_h)
    x2, y2 = _clamp_pt(x2, y2, frame_w, frame_h)
    src = getattr(pt, "source", "motion")
    real = pt.detected or src in ("wrist_proxy", "blended")
    if real:
        cv2.rectangle(img, (x1, y1), (x2, y2), _BALL_DETECTED, 2, cv2.LINE_AA)
        if debug:
            conf = getattr(pt, "confidence", 0.0)
            label = f"ball {conf:.2f}" if conf > 0 else "ball"
        else:
            label = "ball"
    else:
        bt._draw_dashed_rect(img, (x1, y1), (x2, y2), _BALL_PRED, 1)
        label = "ball (est)"
    bt._draw_label_tag(img, x1, y1, label,
                       _BALL_DETECTED if real else _BALL_PRED)


def _draw_bowler_box(img, bbox, frame_h, frame_w):
    """Identity-locked bowler box: drawn only in frames where the bowler crop
    bbox genuinely exists (carried over at most a few tracker-gap frames).
    A frame with no bowler bbox draws NO bowler box -- there is no substitute
    person to highlight."""
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = _clamp_pt(x1, y1, frame_w, frame_h)
    x2, y2 = _clamp_pt(x2, y2, frame_w, frame_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), _BOWLER_BOX, 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "BOWLER"
    ty = max(18, y1 - 8)
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 2)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), _BOWLER_BOX, -1)
    cv2.putText(img, label, (x1 + 4, ty - 2), font, 0.5, (14, 18, 26), 2, cv2.LINE_AA)


def _draw_debug_panel(img, frame_w, frame_h, lines):
    """Bottom-left diagnostic overlay (debug mode only)."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 20
    pad, x = 10, 12
    y0 = frame_h - pad
    panel_h = line_h * len(lines) + 2 * pad
    y_top = max(0, y0 - panel_h)
    overlay = img[y_top:y0, 0:frame_w]
    if overlay.size:
        cv2.rectangle(overlay, (0, 0), (frame_w, overlay.shape[0]), (14, 18, 26), -1)
        img[y_top:y0, 0:frame_w] = cv2.addWeighted(overlay, 0.72, img[y_top:y0, 0:frame_w], 0.28, 0)
    y = y0 - pad - line_h
    for ln in lines:
        cv2.putText(img, ln, (x, y), font, 0.5, _DEBUG_TEXT, 1, cv2.LINE_AA)
        y -= line_h


def _draw_header(img, frame_idx, release_frame, frame_w):
    """Subtle top bar: product name + real current-frame info."""
    hdr_h = 34
    overlay = img[0:hdr_h, 0:frame_w]
    cv2.rectangle(overlay, (0, 0), (frame_w, hdr_h), _HEADER_BG, -1)
    img[0:hdr_h, 0:frame_w] = cv2.addWeighted(overlay, 0.72, img[0:hdr_h, 0:frame_w], 0.28, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PACEAI  ·  ANALYSIS REPLAY", (12, 22), font, 0.55,
                (_BONE[0], _BONE[1], _BONE[2]), 1, cv2.LINE_AA)

    if release_frame is not None and frame_idx == release_frame:
        text = f"frame {frame_idx}  ·  RELEASE"
    else:
        text = f"frame {frame_idx}"
    tw, th = cv2.getTextSize(text, font, 0.5, 1)[0]
    x = frame_w - tw - 14
    cv2.putText(img, text, (x, 22), font, 0.5, _HEADER_TEXT, 1, cv2.LINE_AA)


def render_analysis_replay(
    frames,                # list[(idx, ts, BGR_full)]
    trajectory,            # list[BallPoint] full-frame coords (merged display track)
    pose_sequence,         # list[PoseFrame] (crop-normalized landmarks)
    bowler_bboxes,         # dict frame_idx -> (x1,y1,x2,y2) full-frame crop bbox
    release_frame,         # int | None
    output_path,           # str
    fps: float = 20.0,
    frame_dims=None,       # (h, w) of the full frame (default from first frame)
    min_visibility: float = 0.4,
    debug: bool = False,
    bowler_track_id=None,       # int | None  (locked bowler identity)
    bowler_confidence=None,     # float | None
) -> str:
    """Render ONE synchronized analysis video. Returns the output path.

    Every overlay is drawn only where real data exists for that exact frame:
      * bowler box -- the identity-locked bowler crop bbox for this frame
                      (no box when the bowler is not in frame -- never a
                      substitute person)
      * ball box   -- red for detected/wrist-proxy/blended, amber dashed for
                      predicted (from trajectory[frame_idx]) -- nearest by index
      * trajectory -- running polyline through REAL track points up to now
      * skeleton   -- pose landmarks mapped crop->full for this frame
      * release    -- a marker at the release frame (if provided)

    `debug=True` adds a bottom-left diagnostic panel (bowler track id +
    cricket-evidence confidence + ball state) and shows ball confidence.
    """
    if not frames:
        raise ValueError("No frames to render.")

    if frame_dims is None:
        frame_dims = frames[0][2].shape[:2]
    frame_h, frame_w = frame_dims

    # Index trajectory + pose by frame_idx for O(1) lookup.
    ball_by_idx = {p.frame_idx: p for p in trajectory}
    ordered_idx = [p.frame_idx for p in sorted(trajectory, key=lambda p: p.frame_idx)]
    pose_px = {}
    for pf in pose_sequence:
        pose_px[pf.frame_idx] = _landmarks_to_full_pixels(
            pf, bowler_bboxes, frame_dims, min_visibility=min_visibility)

    ordered_idx_set = set(ordered_idx)

    # Running path points (real, in order) so we render O(n), not O(n^2).
    path_pts = []
    started = False

    # Ball-state counters for the debug panel.
    n_det, n_pred = 0, 0
    for p in trajectory:
        if p.detected or getattr(p, "source", "") in ("wrist_proxy", "blended"):
            n_det += 1
        else:
            n_pred += 1

    out_frames = []
    for idx, ts, fr in frames:
        img = fr.copy()
        h, w = img.shape[:2]

        pt = ball_by_idx.get(idx)

        # --- Identity-locked bowler box (real bowler crop for this frame) ---
        if bowler_bboxes:
            _draw_bowler_box(img, bowler_bboxes.get(idx), h, w)

        # --- Ball box (real detection for this frame) ---
        if pt is not None:
            _draw_ball_box(img, pt, h, w, debug=debug)

        # --- Trajectory path: real track points up to (and incl.) this frame ---
        if idx in ordered_idx_set:
            started = True
            p = ball_by_idx[idx]
            px, py = _clamp_pt(p.x, p.y, w, h)
            path_pts.append((px, py))
        if started and len(path_pts) >= 2:
            color_seg = _TRAJ if getattr(pt, "detected", True) else _BALL_PRED
            for a, b in zip(path_pts, path_pts[1:]):
                cv2.line(img, a, b, color_seg, 1, cv2.LINE_AA)

        # --- Release-point marker (only if a real release frame is provided) ---
        if release_frame is not None and idx == release_frame:
            rp = ball_by_idx.get(release_frame)
            if rp is not None:
                rx, ry = _clamp_pt(rp.x, rp.y, w, h)
            else:
                rx, ry = w // 2, h // 2
            cv2.circle(img, (rx, ry), 8, _RELEASE, 2, cv2.LINE_AA)
            cv2.circle(img, (rx, ry), 3, _RELEASE, -1, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "RELEASE", (rx + 12, ry - 8), font, 0.5,
                        _RELEASE, 2, cv2.LINE_AA)

        # --- Pose skeleton (mapped to full frame for THIS frame) ---
        if idx in pose_px and pose_px[idx]:
            _draw_pose(img, pose_px[idx])

        # --- Info header ---
        _draw_header(img, idx, release_frame, w)

        # --- Debug diagnostics (off by default in the app) ---
        if debug:
            bowler_line = "BOWLER: none  (no bowler-like motion in clip)"
            if bowler_track_id is not None:
                conf_s = f" (conf {bowler_confidence:.2f})" \
                    if bowler_confidence is not None else ""
                bowler_line = f"BOWLER TRACK #{bowler_track_id}{conf_s}"
            ball_state = f"ball: {n_det} real + {n_pred} pred frames"
            if pt is None:
                ball_state += "  |  no box this frame"
            elif pt.detected or getattr(pt, "source", "") in ("wrist_proxy", "blended"):
                ball_state += "  |  box: REAL"
            else:
                ball_state += "  |  box: PREDICTED"
            _draw_debug_panel(img, w, h,
                              [bowler_line, ball_state])

        out_frames.append((idx, ts, img))

    return bt.write_mp4(out_frames, output_path, fps=fps)
