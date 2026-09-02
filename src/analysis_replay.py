"""
PaceAI ‑ Unified "Analysis Replay" renderer.

Turns the EXTRA pipeline outputs (full-frame videos) into ONE synchronized,
premium analysis video. This module only *consumes* the real outputs of
pipeline.analyze_video():

    * full-frame frames           -> original footage
    * ball tracking trajectory    -> ball box + path            (full-frame px)
    * pose landmarks              -> skeleton                   (crop-normalized
                                                                  -> full-frame px;
                                                                  full-frame-normalized
                                                                  when the pipeline's
                                                                  bowler-pose lock failed
                                                                  and it fell back)
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

# ---- Styling constants (lifted out of the draw functions) ----
_BOWLER_THICKNESS = 3                  # bowler box outline weight
_BOWLER_BAND_H = 20                    # label band height under the bowler
_BOWLER_LABEL_SCALE = 0.6              # font scale for the "BOWLER #N" tag
_BOWLER_FONT_BASELINE = 6              # baseline offset inside the label band
_LABEL_PAD = 7                         # horizontal padding in label bands
_BALL_BLEND_ALPHA = 0.45               # opacity of the real ball-box overlay
_BALL_ROI_PAD = 4                      # padded region around the ball box for the blend

# ---- Spatial de-focus (bowler focus treatment) constants ----
# Presentation-only: applied to the raw frame BEFORE any analysis overlays.
# Controls a shallow-depth-of-field effect that keeps the bowler sharp while
# blurring and dimming the background (batter, fielders, crowd).
#
# The treatment is fully deterministic, does not affect any CV/ML calculation,
# and is not exposed to the end user.  All parameters can be tuned here.
#
# _FOCUS_FEATHER_RADIUS : int
#     Half-width (in pixels) of the soft transition zone around the bowler
#     bounding box.  Larger values produce a wider, more gradual falloff.
#     Recommended range: 15-60.
_FOCUS_FEATHER_RADIUS = 30

# _FOCUS_BLUR_KSIZE : int (must be odd, >= 3)
#     Kernel size for the Gaussian blur applied to the background.
#     Larger values produce a stronger defocus.  The kernel is clamped to
#     be odd and >= 3 automatically.
_FOCUS_BLUR_KSIZE = 51

# _FOCUS_DIM_FACTOR : float  (0.0 = black, 1.0 = no change)
#     Brightness multiplier applied to the blurred background region.
#     Values < 1.0 darken the background, making the bowler pop.
#     Recommended range: 0.30-0.60.
_FOCUS_DIM_FACTOR = 0.38

# _FOCUS_SAT_FACTOR : float  (0.0 = grayscale, 1.0 = no change)
#     Saturation multiplier applied to the blurred background region.
#     Values < 1.0 partially desaturate the background, further reducing
#     visual competition with the bowler.
#     Recommended range: 0.35-0.75.
_FOCUS_SAT_FACTOR = 0.50

# _FOCUS_MIN_BBOX_PX : int
#     Minimum bowler bounding-box width or height (in pixels) below which
#     the focus treatment is skipped entirely.  Prevents the effect from
#     activating on tiny, unreliable detections.
_FOCUS_MIN_BBOX_PX = 30


def _clamp_pt(x, y, w, h):
    x = max(0, min(w - 1, int(round(x))))
    y = max(0, min(h - 1, int(round(y))))
    return x, y


def _landmarks_to_full_pixels(pose_frame, bowler_bboxes, frame_dims,
                              min_visibility=0.4, in_full_frame=False):
    """Map pose landmarks to full-frame pixel coords.

    `in_full_frame=False` (default): landmarks are normalized to the bowler crop
    bbox -- map them back through the real crop bbox of this frame.
    `in_full_frame=True`: landmarks are normalized to the whole frame straight
    from the pose model -- full-frame pixel coords are direct.

    Returns dict landmark_idx -> (px, py). Weak/out-of-bbox landmarks omitted.
    """
    fh, fw = frame_dims
    out = {}
    lm = pose_frame.landmarks
    if in_full_frame:
        for i in range(min(lm.shape[0], len(config.POSE_LANDMARK_NAMES))):
            x, y, z, vis = lm[i]
            if vis < min_visibility:
                continue
            px, py = _clamp_pt(float(x) * fw, float(y) * fh, fw, fh)
            out[i] = (px, py)
        return out
    bbox = bowler_bboxes.get(pose_frame.frame_idx)
    if bbox is None:
        return {}
    bx1, by1, bx2, by2 = bbox
    crop_w = max(1.0, bx2 - bx1)
    crop_h = max(1.0, by2 - by1)
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
        # De-emphasized, semi-transparent ball box: the ball is a *tracked
        # object*, never the subject. The sharp/opaque bowler overlay is drawn
        # AFTER this so the bowler always reads as the visual focus.
        # Blend is scoped to a small padded ROI around the box (not the whole
        # frame) -- same visual result, without a full-frame copy+blend on
        # every frame that has a ball point.
        pad = _BALL_ROI_PAD
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(frame_w, x2 + pad), min(frame_h, y2 + pad)
        if rx2 > rx1 and ry2 > ry1:
            roi = img[ry1:ry2, rx1:rx2]
            roi_overlay = roi.copy()
            cv2.rectangle(roi_overlay, (x1 - rx1, y1 - ry1), (x2 - rx1, y2 - ry1),
                          _BALL_DETECTED, 1, cv2.LINE_AA)
            roi[:] = cv2.addWeighted(roi_overlay, _BALL_BLEND_ALPHA, roi,
                                     1.0 - _BALL_BLEND_ALPHA, 0)
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


def _draw_bowler_box(img, bbox, frame_h, frame_w, bowler_track_id=None):
    """Identity-locked bowler box: drawn only in frames where the bowler crop
    bbox genuinely exists (carried over at most a few tracker-gap frames).
    A frame with no bowler bbox draws NO bowler box -- there is no substitute
    person to highlight."""
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = _clamp_pt(x1, y1, frame_w, frame_h)
    x2, y2 = _clamp_pt(x2, y2, frame_w, frame_h)
    # Thicker, brighter box so the bowler is unmistakable against any
    # background activity (batter/runner/fielders in the full-frame footage).
    cv2.rectangle(img, (x1, y1), (x2, y2), _BOWLER_BOX, _BOWLER_THICKNESS, cv2.LINE_AA)
    bw = max(1, x2 - x1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tag = f"BOWLER #{bowler_track_id}" if bowler_track_id is not None else "BOWLER"
    (tw, th), _ = cv2.getTextSize(tag, font, _BOWLER_LABEL_SCALE, 2)
    label_w = max(tw + 2 * _LABEL_PAD, bw + 2 * _LABEL_PAD)
    y_bottom = min(frame_h - 1, y2)
    # Full-width band under the bowler so the label is huge and legible even
    # when the bowler is small in frame.
    cv2.rectangle(img, (x1, y_bottom - _BOWLER_BAND_H), (x1 + label_w, y_bottom),
                  _BOWLER_BOX, -1)
    cv2.putText(img, tag, (x1 + _LABEL_PAD, y_bottom - _BOWLER_FONT_BASELINE), font,
                _BOWLER_LABEL_SCALE, (14, 18, 26), 2, cv2.LINE_AA)


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


def _draw_header(img, frame_idx, release_frame, frame_w, bowler_track_id=None):
    """Subtle top bar: product name + real current-frame info.

    When the bowler identity is locked it is called out on the right so it is
    always obvious the analysis is on the bowler (not the batter/background)."""
    hdr_h = 34
    overlay = img[0:hdr_h, 0:frame_w]
    cv2.rectangle(overlay, (0, 0), (frame_w, hdr_h), _HEADER_BG, -1)
    img[0:hdr_h, 0:frame_w] = cv2.addWeighted(overlay, 0.72, img[0:hdr_h, 0:frame_w], 0.28, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PACEAI  ·  ANALYSIS REPLAY", (12, 22), font, 0.55,
                (_BONE[0], _BONE[1], _BONE[2]), 1, cv2.LINE_AA)

    # Left-of-right text: bowler-lock callout, then frame/release info.
    right_texts = []
    if bowler_track_id is not None:
        right_texts.append(f"BOWLER LOCKED #{bowler_track_id}")
    if release_frame is not None and frame_idx == release_frame:
        right_texts.append("RELEASE")
    right_texts.append(f"frame {frame_idx}")

    x = frame_w - 14
    for txt in reversed(right_texts):
        tw, th = cv2.getTextSize(txt, font, 0.5, 1)[0]
        x -= tw
        color = (_RELEASE[0], _RELEASE[1], _RELEASE[2]) if txt == "RELEASE" \
            else (_BOWLER_BOX[0], _BOWLER_BOX[1], _BOWLER_BOX[2]) if txt.startswith("BOWLER") \
            else _HEADER_TEXT
        cv2.putText(img, txt, (x, 22), font, 0.5, color, 1, cv2.LINE_AA)
        x -= 14


def _apply_focus_treatment(img, bowler_bbox, frame_h, frame_w):
    """Apply a spatial de-focus (shallow depth-of-field) to a single frame.

    The bowler region stays sharp and bright; the background is blurred,
    darkened, and partially desaturated.  This is a PRESENTATION-ONLY effect
    applied to the raw frame BEFORE any analysis overlays are drawn.

    Parameters are read from the module-level _FOCUS_* constants.

    If bowler_bbox is ``None`` or too small, the frame is returned unchanged.

    Returns the modified frame (in-place mutation of *img*).
    """
    if bowler_bbox is None:
        return img

    bx1, by1, bx2, by2 = [int(round(v)) for v in bowler_bbox]
    bw, bh = bx2 - bx1, by2 - by1
    if bw < _FOCUS_MIN_BBOX_PX or bh < _FOCUS_MIN_BBOX_PX:
        return img

    # --- Step 1: build feathered binary mask from the bowler bbox ---
    mask = np.zeros((frame_h, frame_w), dtype=np.float32)
    _cx1 = max(0, bx1)
    _cy1 = max(0, by1)
    _cx2 = min(frame_w, bx2)
    _cy2 = min(frame_h, by2)
    if _cx2 <= _cx1 or _cy2 <= _cy1:
        return img
    mask[_cy1:_cy2, _cx1:_cx2] = 1.0
    feather = max(3, _FOCUS_FEATHER_RADIUS)
    ksize = _FOCUS_BLUR_KSIZE if _FOCUS_BLUR_KSIZE % 2 == 1 else _FOCUS_BLUR_KSIZE + 1
    ksize = max(3, ksize)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # --- Step 2: blurred + darkened + desaturated background ---
    blur_ksize = max(3, ksize)
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Dim the blurred background.
    if _FOCUS_DIM_FACTOR < 1.0:
        blurred = (blurred.astype(np.float32) * _FOCUS_DIM_FACTOR).astype(np.uint8)

    # Partially desaturate the blurred background.
    if _FOCUS_SAT_FACTOR < 1.0:
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= _FOCUS_SAT_FACTOR
        blurred = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- Step 3: composite — sharp bowler (mask=1) + blurred background (mask=0) ---
    mask3 = mask[:, :, np.newaxis]
    composite = (img.astype(np.float32) * mask3
                 + blurred.astype(np.float32) * (1.0 - mask3))
    np.clip(composite, 0, 255, out=composite)
    img[:] = composite.astype(np.uint8)
    return img


def render_analysis_replay(
    frames,                # list[(idx, ts, BGR_full)]
    trajectory,            # list[BallPoint] full-frame coords (merged display track)
    pose_sequence,         # list[PoseFrame] landmarks
    bowler_bboxes,         # dict frame_idx -> (x1,y1,x2,y2) full-frame crop bbox
    release_frame,         # int | None
    output_path,           # str
    fps: float = 20.0,
    frame_dims=None,       # (h, w) of the full frame (default from first frame)
    min_visibility: float = 0.4,
    debug: bool = False,
    bowler_track_id=None,       # int | None  (locked bowler identity)
    bowler_confidence=None,     # float | None
    pose_in_full_frame: bool = False,  # landmarks normalized to the full frame
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
            pf, bowler_bboxes, frame_dims, min_visibility=min_visibility,
            in_full_frame=pose_in_full_frame)

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

        # --- 1. Spatial de-focus: blur + dim background, keep bowler sharp ---
        # Applied to the raw frame BEFORE any overlays so all analysis
        # annotations (trajectory, ball box, skeleton, bowler box) are drawn
        # on top of the treated frame and remain crisp.  Overlay coordinates
        # are unchanged — the treatment is purely visual.
        bowler_bbox = bowler_bboxes.get(idx) if bowler_bboxes else None
        _apply_focus_treatment(img, bowler_bbox, h, w)

        # --- 2. Trajectory path: real track points up to (and incl.) this frame ---
        if idx in ordered_idx_set:
            started = True
            p = ball_by_idx[idx]
            px, py = _clamp_pt(p.x, p.y, w, h)
            path_pts.append((px, py))
        if started and len(path_pts) >= 2:
            color_seg = _TRAJ if getattr(pt, "detected", True) else _BALL_PRED
            for a, b in zip(path_pts, path_pts[1:]):
                cv2.line(img, a, b, color_seg, 1, cv2.LINE_AA)

        # --- 3. Ball box (real detection for this frame) ---
        # Semi-transparent; drawn before the bowler so the bowler always
        # reads as the topmost visual subject.
        if pt is not None:
            _draw_ball_box(img, pt, h, w, debug=debug)

        # --- 4. Pose skeleton (mapped to full frame for THIS frame) ---
        if idx in pose_px and pose_px[idx]:
            _draw_pose(img, pose_px[idx])

        # --- 5. Release-point marker (only if a real release frame is provided) ---
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

        # --- 6. Identity-locked bowler box (real bowler crop for this frame) ---
        # Drawn LAST among the scene overlays so the sharp, bright bowler
        # box always wins z-order against the blurred background.
        if bowler_bboxes:
            _draw_bowler_box(img, bowler_bboxes.get(idx), h, w,
                             bowler_track_id=bowler_track_id)

        # --- 7. Info header ---
        _draw_header(img, idx, release_frame, w,
                     bowler_track_id=bowler_track_id)

        # --- 8. Debug diagnostics (off by default in the app) ---
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
