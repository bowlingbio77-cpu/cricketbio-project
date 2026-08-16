"""
Stage 2b: Ball Detection & Tracking + Annotated Output Video

Detects the cricket ball across frames using YOLO's COCO "sports ball"
class (id 32), links detections into a smooth trajectory with a small
constant-velocity motion model (predict -> nearest-match -> update), fills
short gaps by linear extrapolation, and renders an H.264 MP4 with a red
bounding box drawn around the ball for display in the Streamlit app.

When ultralytics is unavailable (or the ball is missed by YOLO), a lightweight
frame-difference motion detector contributes candidate blobs. Candidates are
source-tagged ("yolo" / "motion") and the matcher is source-gated so the box
cannot drift onto the bowler's moving limbs:
  - YOLO detections at/above `seed_min_conf` *seed* tracks (a running up /
    delivery arm is never the ball, so motion blobs are not allowed to start
    a track when a detector is loaded);
  - when extending a track, a YOLO candidate inside the search radius always
    wins; motion blobs only bridge short gaps, and only while the track has
    seen a YOLO detection within the last `YOLO_LIVENESS_FRAMES` frames --
    otherwise the track just extrapolates and dies instead of following limbs;
  - motion blobs are also size-filtered to ball-like blobs (compact and small),
    which drops the large/elongated blobs a moving arm or leg produces.
Without a YOLO model, motion-only tracking still works: a small blob that
persists two frames running is allowed to start a track.

The raw trajectory is then trimmed down to the release -> impact window (ball
leaving the bowler's hand through contact with the bat/pad/ground). There is
no bowler/bat pose model in this module, so the window is located heuristically
from the ball's own motion signature -- see `_trim_to_release_impact`.

The output video is written with the ffmpeg binary bundled by imageio-ffmpeg
(libx264), which is the only reliable way to get a browser-playable H.264 MP4
-- OpenCV's wheels have no working H.264 encoder and MPEG-4 Part 2 (mp4v)
does not play in HTML5 <video>.
"""
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from . import config

try:
    from ultralytics import YOLO
    from .detection import resolve_weights
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

BALL_CLASS_ID = 32          # COCO "sports ball"
SEED_MIN_CONF = 0.3         # YOLO detections this confident can start a track
MOTION_CONF = 0.15          # motion-blob votes (below SEED_MIN_CONF: extend/confirm only)
MAX_GAP_FRAMES = 8          # longest gap (frames) the tracker bridges by extrapolation
MIN_TRACK_FRAMES = 3        # below this many frames the track is discarded
MATCH_BASE_RADIUS = 30.0    # px -- minimum matching radius
MATCH_VELOCITY_FACTOR = 1.8  # radius grows with the ball's speed
YOLO_LIVENESS_FRAMES = 3    # frames of YOLO support needed for motion to extend a track

# Motion-blob size limits: the cricket ball is a small, roughly circular blob
# even at release, whereas a moving limb produces a large or elongated blob.
MOTION_MIN_AREA = 8.0       # px^2 -- below this it's frame noise
MOTION_MAX_AREA = 1200.0    # px^2 -- above this it's a limb/body part
MOTION_MAX_ASPECT = 2.5     # width/height ratio -- elongated blobs are limbs

# Track-validity limits: a ball in flight always travels a meaningful distance
# across the frame, so a "track" that stays in one spot is a false positive --
# YOLO firing repeatedly on a fixed round object (a stump, helmet, or lens
# flare). MIN_TRACK_SPREAD = the flight must span this much of the frame;
# MIN_TRACK_PATH = and travel at least this far in total. A slow object whose
# box keeps growing (a person walking toward the camera) is rejected by the
# combined box-growth + speed rule.
MIN_TRACK_SPREAD_PX = 60.0
MIN_TRACK_PATH_PX = 100.0
MAX_SEGMENT_JUMP_PX = 80.0   # a same-track point can't teleport this far in 1 frame
BOX_GROWTH_LIMIT = 2.8       # box grew more than this AND moved slowly => a person
MIN_AVG_SPEED_PX_S = 400.0

# -- release / impact windowing ------------------------------------------- #
# Heuristic constants: release is where the ball's frame-to-frame speed
# jumps to a sustained high value (leaving the hand); impact is the first
# sharp direction change / speed drop after release (bat, pad, or ground
# contact). These are motion-signature heuristics, not a learned model --
# tune them if your source footage's frame rate / resolution differs a lot
# from what they were picked for.
RELEASE_MIN_SPEED_FACTOR = 1.6   # release speed must exceed this x the pre-release baseline
RELEASE_SUSTAIN_FRAMES = 3       # consecutive fast frames needed to confirm release
IMPACT_ANGLE_THRESHOLD_DEG = 35.0  # velocity-direction change (deg) that signals impact
IMPACT_SPEED_DROP_FACTOR = 0.55    # or: speed suddenly drops to this fraction of prior speed
IMPACT_MIN_SPEED_PX_FRAME = 6.0    # ignore impact candidates with tiny prior speed (jitter)

# -- rendering -------------------------------------------------------------- #
BOX_COLOR = (0, 0, 255)          # BGR red
DEFAULT_BALL_DIAMETER_PX = 16.0  # fallback box size when no detector gave us real dimensions


@dataclass
class BallPoint:
    frame_idx: int
    timestamp_sec: float
    x: float
    y: float
    confidence: float        # 0.0 => extrapolated/predicted, >0 => detected
    detected: bool = True
    w: float = 0.0            # bounding-box width in px (0 => unknown, use fallback)
    h: float = 0.0            # bounding-box height in px (0 => unknown, use fallback)


class BallTracker:
    """Detect + track the ball through a sequence of frames."""

    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf_threshold: float = 0.1,
                 seed_min_conf: float = SEED_MIN_CONF,
                 model=None):
        # conf_threshold is deliberately low (0.1): at delivery-cam resolution
        # the ball is a small fast object that COCO's generic "sports ball"
        # class only scores at 0.05-0.35. The 0.3 seed gate keeps low-conf
        # false positives from starting a track.
        self.conf_threshold = conf_threshold
        self.seed_min_conf = seed_min_conf
        self._model = model
        self.model_error = None
        if self._model is None and _HAS_ULTRALYTICS:
            try:
                self._model = YOLO(resolve_weights(weights))
            except Exception as exc:
                self.model_error = str(exc)
                self._model = None

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def _candidates(self, frame: np.ndarray, prev_gray: np.ndarray) -> List[Tuple[float, float, float, float, float, str]]:
        """(x, y, confidence, w, h, source) candidates: YOLO ball detections + motion blobs.

        `source` is "yolo" or "motion" -- the matching logic in `_step` uses it
        so motion/color blobs can never outrank a YOLO detection just for being
        closer to the predicted position (this is what used to let the bowler's
        glove/pad drift the box onto a limb mid-delivery).
        """
        dets = []
        if self._model is not None:
            try:
                results = self._model.predict(
                    frame, classes=[BALL_CLASS_ID],
                    conf=self.conf_threshold, verbose=False,
                )
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        # Keep the real confidence -- do NOT clamp it up to
                        # seed_min_conf here, or every detection above
                        # conf_threshold would silently qualify as a "seed".
                        dets.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf, x2 - x1, y2 - y1, "yolo"))
            except Exception:
                pass
        dets.extend(self._motion_candidates(frame, prev_gray))
        return dets

    @staticmethod
    def _motion_candidates(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> List[Tuple[float, float, float, float, float, str]]:
        """Small fast-moving ball-sized blobs between consecutive frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return []
        diff = cv2.absdiff(gray, prev_gray)
        _, th = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MOTION_MIN_AREA or area > MOTION_MAX_AREA:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            if max(bw, bh) / min(bw, bh) > MOTION_MAX_ASPECT:
                continue  # elongated blob: a limb, bat, or seam, not the ball
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], MOTION_CONF,
                        float(bw), float(bh), "motion"))
        return out

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def track(self, frames, fps: Optional[float] = None) -> Tuple[List[BallPoint], dict]:
        """
        `frames`: iterable of (frame_idx, timestamp_sec, frame_bgr).

        Returns (trajectory, stats). The returned trajectory is the longest
        contiguous tracking segment (NOT trimmed) so the annotated video shows
        the box for every frame the ball was followed; the release -> impact
        window (release_idx / impact_idx / trimmed) is reported in `stats`
        (callers that want the box to stop at bat/pad/ground contact clip the
        track at `impact_idx` themselves). The trajectory is empty when no
        track survived the minimum-length check (MIN_TRACK_FRAMES).
        `fps`, when given, makes `summarize` report speeds in px/sec rather
        than px/frame.
        """
        track: List[BallPoint] = []
        state = {"last": None, "velocity": np.zeros(2), "gaps": 0,
                 "box_size": None, "pending": None, "since_yolo": 0}
        prev_gray = None
        total_frames = 0

        for idx, ts, frame in frames:
            total_frames += 1
            cands = self._candidates(frame, prev_gray)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = gray

            pt = self._step(idx, ts, cands, state)
            if pt is not None:
                track.append(pt)

        # The longest raw segment is the natural candidate, but a longer
        # segment can be a fake track (a person walking toward the camera,
        # a stationary object YOLO keeps re-firing on) while a shorter one is
        # the real ball. So validate every segment longest-first and return
        # the first one that looks like a genuine ball flight.
        h, w = frames[0][2].shape[:2]
        margin = 8
        longest = []
        for cand in sorted(self._split_segments(track), key=len, reverse=True):
            if len(cand) < MIN_TRACK_FRAMES:
                continue
            # The ball can't be followed after it leaves the frame: drop
            # trailing extrapolated points that run off the image edge
            # (otherwise the box keeps drifting out of view after the ball is
            # hit past the batsman).
            while cand and (cand[-1].x < -margin or cand[-1].x > w + margin
                            or cand[-1].y < -margin or cand[-1].y > h + margin):
                cand.pop()
            if len(cand) < MIN_TRACK_FRAMES:
                continue
            # A delivery ball always moves: reject segments pinned to one spot
            # (YOLO confidently re-firing on a fixed round object every frame),
            # and reject slow segments whose box balloons out -- that's a
            # person walking toward the camera (or a motion-blob takeover),
            # not a ball in flight. Box growth is max/min so a segment that
            # *starts* on an already-big object is still caught; the speed
            # gate is NET speed (end-to-end), which is immune to zig-zag
            # Kalman/noise paths inflating a segment's path length.
            pts = np.array([[p.x, p.y] for p in cand], dtype=float)
            span = np.max(pts, axis=0) - np.min(pts, axis=0)
            spread = float(np.hypot(span[0], span[1]))
            seg = np.diff(pts, axis=0)
            path = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
            dt = cand[-1].timestamp_sec - cand[0].timestamp_sec
            net_speed = (float(np.hypot(cand[-1].x - cand[0].x, cand[-1].y - cand[0].y)) / dt
                         if dt > 0 else 0.0)
            ws = [p.w for p in cand]
            growth = max(ws) / max(min(ws), 1.0)
            if (spread >= MIN_TRACK_SPREAD_PX and path >= MIN_TRACK_PATH_PX
                    and not (growth > BOX_GROWTH_LIMIT and net_speed < MIN_AVG_SPEED_PX_S)):
                longest = cand
                break

        trimmed, window_info = self._trim_to_release_impact(longest) if longest else ([], {})

        stats = {"total_frames": total_frames}
        stats.update(window_info)
        stats.update(summarize(longest, fps=fps))
        return longest, stats

    def _step(self, idx: int, ts: float, cands, state: dict) -> Optional[BallPoint]:
        """Advance the tracker one frame; returns a BallPoint or None."""
        # Extend an existing track. YOLO candidates within radius always win
        # over motion/color candidates, regardless of distance -- otherwise a
        # closer glove/pad/sock blob can hijack an already-good YOLO-seeded
        # track (this is what caused the frame-21 drift onto the bowler's
        # sock). Motion candidates are only allowed to bridge short gaps while
        # the track has been YOLO-supported recently (`since_yolo`); after that
        # the track just extrapolates until it dies rather than latching onto
        # a moving limb.
        if state["last"] is not None and state["gaps"] <= MAX_GAP_FRAMES:
            predicted = state["last"] + state["velocity"]
            radius = max(MATCH_BASE_RADIUS,
                         float(np.linalg.norm(state["velocity"])) * MATCH_VELOCITY_FACTOR + MATCH_BASE_RADIUS)

            def _nearest_in_radius(source: Optional[str]):
                best, best_d = None, None
                for cx, cy, conf, cw, ch, src in cands:
                    if source is not None and src != source:
                        continue
                    d = float(np.hypot(cx - predicted[0], cy - predicted[1]))
                    if d <= radius and (best_d is None or d < best_d):
                        best, best_d = (cx, cy, conf, cw, ch), d
                return best

            best = _nearest_in_radius("yolo")
            yolo_match = best is not None
            if best is None and state.get("since_yolo", 0) <= YOLO_LIVENESS_FRAMES:
                best = _nearest_in_radius("motion")
            if best is not None:
                cx, cy, conf, cw, ch = best
                delta = np.array([cx, cy]) - state["last"]
                if state["gaps"] == 0:
                    state["velocity"] = 0.6 * state["velocity"] + 0.4 * delta
                else:
                    state["velocity"] = delta
                state["gaps"] = 0
                state["last"] = np.array([cx, cy])
                state["since_yolo"] = 0 if yolo_match else state.get("since_yolo", 0) + 1
                w, h = self._update_box_size(state, cw, ch)
                return BallPoint(idx, ts, float(cx), float(cy), conf, detected=True, w=w, h=h)

            # Bridge a short gap by extrapolation.
            w, h = state["box_size"] or (DEFAULT_BALL_DIAMETER_PX, DEFAULT_BALL_DIAMETER_PX)
            pt = BallPoint(idx, ts, float(predicted[0]), float(predicted[1]), 0.0, detected=False, w=w, h=h)
            state["gaps"] += 1
            state["last"] = predicted
            state["since_yolo"] = state.get("since_yolo", 0) + 1
            state["velocity"] = 0.9 * state["velocity"]
            return pt

        # (Re)start a new track. When a real model is loaded, only YOLO
        # detections at/above seed_min_conf may originate a track -- a moving
        # limb during the run-up or delivery must never re-seed us. Without a
        # model, a small fast blob persisting two frames running is the best
        # evidence we have, so motion-only seeding is allowed there.
        if self._model is not None:
            seedable = [c for c in cands if c[5] == "yolo" and c[2] >= self.seed_min_conf]
        else:
            seedable = self._confirmed_motion_seed(cands, state, idx)
        if seedable:
            cx, cy, conf, cw, ch = max(seedable, key=lambda c: c[2])[0:5]
            state["last"] = np.array([cx, cy])
            state["velocity"] = np.zeros(2)
            state["gaps"] = 0
            state["box_size"] = None
            state["pending"] = None
            state["since_yolo"] = 0 if self._model is not None else 1
            w, h = self._update_box_size(state, cw, ch)
            return BallPoint(idx, ts, float(cx), float(cy), conf, detected=True, w=w, h=h)

        state["last"] = None
        state["gaps"] = 0
        return None

    @staticmethod
    def _confirmed_motion_seed(cands, state: dict, idx: int) -> list:
        """
        Motion-only fallback seed (used only when no detector is loaded): a
        small fast-moving blob that appears at roughly the same place two
        frames running is likely the ball (a limb moves erratically and won't
        hold position across frames). Returns a one-element candidate list
        when confirmed, otherwise [] -- and stashes the strongest motion blob
        in `state["pending"]` for the next frame.
        """
        motion = [c for c in cands if abs(c[2] - MOTION_CONF) < 1e-6]
        if not motion:
            state["pending"] = None
            return []
        strongest = max(motion, key=lambda c: c[3] * c[4])
        pending = state.get("pending")
        if pending is not None:
            px, py, pframe = pending
            if idx - pframe <= 2:
                best = min(motion, key=lambda c: np.hypot(c[0] - px, c[1] - py))
                if np.hypot(best[0] - px, best[1] - py) <= MATCH_BASE_RADIUS * 2:
                    state["pending"] = None
                    return [best]
        state["pending"] = (strongest[0], strongest[1], idx)
        return []

    @staticmethod
    def _update_box_size(state: dict, cw: float, ch: float) -> Tuple[float, float]:
        """Maintain an EMA of the ball's on-screen box size for extrapolated frames."""
        if cw <= 0 or ch <= 0:
            w, h = state["box_size"] or (DEFAULT_BALL_DIAMETER_PX, DEFAULT_BALL_DIAMETER_PX)
            return w, h
        if state["box_size"] is None:
            state["box_size"] = (cw, ch)
        else:
            pw, ph = state["box_size"]
            state["box_size"] = (0.7 * pw + 0.3 * cw, 0.7 * ph + 0.3 * ch)
        return state["box_size"]

    @staticmethod
    def _split_segments(track: List[BallPoint]) -> List[List[BallPoint]]:
        # A gap in frame_idx means `_step` returned None for those frames
        # (a full reset), so the points on either side belong to different
        # tracking episodes and must never be merged. The jump guard catches
        # the same-tick case where a reset and a re-seed land next to each
        # other in index space (f41 -> f42) but are two different objects.
        if not track:
            return []
        segments: List[List[BallPoint]] = []
        cur = [track[0]]
        for a, b in zip(track, track[1:]):
            jump = float(np.hypot(b.x - a.x, b.y - a.y))
            if b.frame_idx - a.frame_idx <= 1 and jump <= MAX_SEGMENT_JUMP_PX:
                cur.append(b)
            else:
                segments.append(cur)
                cur = [b]
        segments.append(cur)
        return segments

    @staticmethod
    def _trim_to_release_impact(track: List[BallPoint]) -> Tuple[List[BallPoint], dict]:
        """
        Heuristically trim a trajectory down to the release -> impact window.

        There's no bowler/bat detector here, so this works purely off the
        ball's own motion signature:
          - "release": the point where frame-to-frame speed first jumps to a
            sustained high value. Before release, any candidates being
            picked up are the bowler's hand/arm, which moves more slowly and
            erratically; the instant the ball leaves the hand it accelerates
            into an essentially straight, fast, ballistic path.
          - "impact": the first sharp change in direction (or a sudden speed
            drop) after release, corresponding to the ball hitting the bat,
            pad, or ground.

        Falls back to the release-onward (or the full) track if a boundary
        can't be confidently located -- e.g. the clip doesn't contain a
        clean release, or the ball is never redirected within the frames
        given. `info["trimmed"]` tells the caller whether trimming actually
        did anything.
        """
        info = {"release_idx": None, "impact_idx": None, "trimmed": False}
        if len(track) < RELEASE_SUSTAIN_FRAMES + 2:
            return track, info

        pts = np.array([[p.x, p.y] for p in track])
        vel = np.diff(pts, axis=0)
        speed = np.hypot(vel[:, 0], vel[:, 1])  # speed[i] is between track[i] and track[i+1]

        # --- locate release -------------------------------------------- #
        baseline = float(np.median(speed[: min(5, len(speed))])) if len(speed) else 0.0
        threshold = max(baseline * RELEASE_MIN_SPEED_FACTOR, MATCH_BASE_RADIUS * 0.5)
        release_i = None
        for i in range(len(speed) - RELEASE_SUSTAIN_FRAMES + 1):
            if np.all(speed[i:i + RELEASE_SUSTAIN_FRAMES] >= threshold):
                release_i = i
                break
        if release_i is None:
            release_i = 0  # no clear release signature; keep from the start

        # --- locate impact (search after release) ----------------------- #
        # Use windowed (median) speeds so a single jittery motion-blob point
        # can't masquerade as a bat/pad contact, and require both velocities
        # to be non-trivial before trusting an angle (a nearly-stationary
        # point makes the angle ill-conditioned). A real ball stopping at the
        # bat is still caught by the windowed speed-drop test.
        speeds = np.asarray(speed, dtype=float)
        impact_i = None
        for i in range(release_i + 1, len(speed) - 1):
            v1, v2 = vel[i - 1], vel[i]
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if n1 < IMPACT_MIN_SPEED_PX_FRAME:
                continue
            angle_deg = 0.0
            if n2 >= IMPACT_MIN_SPEED_PX_FRAME:
                cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_ang)))
            s_prev = float(np.median(speeds[max(0, i - 2):i]))
            s_next = float(np.median(speeds[i:i + 2]))
            speed_drop = s_next < s_prev * IMPACT_SPEED_DROP_FACTOR
            if angle_deg >= IMPACT_ANGLE_THRESHOLD_DEG or speed_drop:
                impact_i = i + 1  # the point *after* the deflection is the impact point
                break

        release_idx = track[release_i].frame_idx
        if impact_i is not None:
            impact_idx = track[impact_i].frame_idx
            trimmed = [p for p in track if release_idx <= p.frame_idx <= impact_idx]
            info.update(release_idx=release_idx, impact_idx=impact_idx, trimmed=True)
        else:
            trimmed = [p for p in track if p.frame_idx >= release_idx]
            info.update(release_idx=release_idx, impact_idx=None, trimmed=release_i > 0)

        if len(trimmed) < MIN_TRACK_FRAMES:
            return track, info
        return trimmed, info


def track_ball(frames, fps: Optional[float] = None) -> Tuple[List[BallPoint], dict]:
    """Convenience wrapper: build a tracker, track, return (trajectory, stats)."""
    tracker = BallTracker()
    if tracker.model_error:
        raise RuntimeError(f"Ball tracker: YOLO failed to load ({tracker.model_error}) -- "
                           "motion-only tracking disabled.")
    return tracker.track(frames, fps=fps)


def summarize(track: List[BallPoint], fps: Optional[float] = None) -> dict:
    """Derive display stats from a trajectory."""
    if not track:
        return {"n_frames": 0, "n_detected": 0, "n_interpolated": 0,
                "coverage_pct": 0.0, "avg_speed_px_s": 0.0, "max_speed_px_s": 0.0,
                "trajectory": []}
    pts = np.array([[p.x, p.y] for p in track], dtype=float)
    n_det = sum(1 for p in track if p.detected)
    span = track[-1].timestamp_sec - track[0].timestamp_sec
    seg = np.diff(pts, axis=0)
    step = np.hypot(seg[:, 0], seg[:, 1])
    avg_speed = float(np.sum(step) / span) if span > 0 else 0.0
    if fps:
        max_step = float(np.max(step)) if len(step) else 0.0
        max_speed = max_step * fps
    else:
        max_speed = float(np.max(step)) if len(step) else 0.0
    return {
        "n_frames": len(track),
        "n_detected": n_det,
        "n_interpolated": len(track) - n_det,
        "coverage_pct": round(100.0 * n_det / len(track), 1),
        "avg_speed_px_s": round(avg_speed, 1),
        "max_speed_px_s": round(max_speed, 1),
        "trajectory": [[float(p.x), float(p.y), int(p.detected)] for p in track],
    }


def annotate_frames(frames, track: List[BallPoint], box_color=BOX_COLOR) -> List[tuple]:
    """Return frames (idx, ts, img) with a red box drawn around the ball.

    A solid box marks an actual detection; a dashed box marks a frame
    bridged by extrapolation during a short tracking gap. Only frames within
    `track` get a box -- frames where the ball wasn't tracked are passed
    through untouched. Callers that want the box to stop at bat/pad/ground
    contact should pass a track clipped at the impact frame (see
    `_trim_to_release_impact`).
    """
    by_idx = {p.frame_idx: p for p in track}
    out = []
    for idx, ts, frame in frames:
        img = frame.copy()
        pt = by_idx.get(idx)
        if pt is not None:
            w = pt.w if pt.w > 0 else DEFAULT_BALL_DIAMETER_PX
            h = pt.h if pt.h > 0 else DEFAULT_BALL_DIAMETER_PX
            x1, y1 = int(round(pt.x - w / 2)), int(round(pt.y - h / 2))
            x2, y2 = int(round(pt.x + w / 2)), int(round(pt.y + h / 2))
            if pt.detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            else:
                _draw_dashed_rect(img, (x1, y1), (x2, y2), box_color, 1)
        out.append((idx, ts, img))
    return out


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash):
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for i in range(0, length, dash * 2):
        s = i / length
        e = min(i + dash, length) / length
        sx, sy = int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s)
        ex, ey = int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash=6):
    """cv2 has no built-in dashed rectangle -- draw one from four dashed lines."""
    x1, y1 = pt1
    x2, y2 = pt2
    for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                 ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        _draw_dashed_line(img, a, b, color, thickness, dash)


def write_mp4(frames, path: str, fps: float) -> str:
    """
    Encode frames (idx, ts, img) as an H.264 MP4 via imageio-ffmpeg's bundled
    ffmpeg (libx264). Returns the output path.
    """
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0][2].shape[:2]

    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                exe, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(float(fps)),
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast",
                path,
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            for _idx, _ts, frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode == 0 and os.path.getsize(path) > 0:
                return path
        except Exception:
            pass

    # Fallback: OpenCV MPEG-4 Part 2 (plays locally but not in browsers).
    writer = None
    for codec in ("mp4v", "avc1"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
        if vw.isOpened():
            writer = vw
            break
        vw.release()
    if writer is None:
        raise RuntimeError("Could not open any video writer for ball-tracking output")
    try:
        for _idx, _ts, frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_output_path(prefix: str = "ball_track_") -> str:
    """Create a fresh temp path for the annotated video."""
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix=prefix)
    os.close(fd)
    return path
