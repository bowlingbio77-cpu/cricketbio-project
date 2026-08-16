"""
Stage 2b: Ball Detection & Tracking + Annotated Output Video (v2)

Multi-hypothesis, Kalman-filtered ball tracking. This is a redesign of the
v1 heuristic-stacking approach (`ball_tracking.py`) around three ideas:

1. MULTI-HYPOTHESIS TRACKING. v1 ran a single hypothesis: one bad early seed
   (e.g. the umpire's glove) could hijack the whole clip. Here several
   candidate tracks run in parallel (bounded by MAX_ACTIVE_TRACKS) and the
   winner is chosen by evidence -- not by luck of being first -- once the
   clip ends.

2. A REAL MOTION MODEL. Each track carries a constant-velocity Kalman
   filter; association is a principled Mahalanobis gate against the filter's
   own predicted uncertainty, not a hand-tuned pixel radius. YOLO boxes get
   a tight measurement covariance (trusted); motion blobs get a loose one.

3. BALLISTIC CONSISTENCY AS EVIDENCE. A local quadratic (constant-
   acceleration) fit is maintained per track and used to (a) softly penalize
   physically-inconsistent candidate matches and (b) rank the surviving
   tracks at the end: a smooth parabolic flight beats a longer jittery one.

Detector priority is unchanged from the v1 fix: with a YOLO model loaded,
only YOLO detections at/above SEED_MIN_CONF may *originate* a track, and
YOLO candidates are assigned before motion candidates (expressed here as an
assignment-order rule across all tracks simultaneously, which is what let a
closer motion blob steal an already-good track in v1). Motion candidates may
only extend a track that still has recent YOLO support (YOLO_LIVENESS_FRAMES),
and are size-filtered to ball-like blobs -- a moving limb must never latch
onto a track.

The v1 safeguards that were validated on real delivery clips are all kept:
track-validity rejection (a pinned-to-one-spot "track" is YOLO re-firing on a
fixed object; a slow growing box is a person walking at the camera), segment
splitting on implausible teleports, trailing off-frame trimming, and the
windowed median-speed impact guard (a single jittery motion point can't fake
a bat/pad contact).

Everything below the `BallTracker` class (rendering, MP4 writing, summarize)
shares its contract with v1 and is reused as-is.
"""
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
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
SEED_MIN_CONF = 0.3         # only YOLO detections this confident can start a track
MOTION_CONF = 0.15          # motion-blob votes (extend only, never seed with a model)
MAX_GAP_FRAMES = 8          # longest gap (frames) a track survives with no match
MIN_TRACK_FRAMES = 3        # below this many frames a track/segment is discarded
YOLO_LIVENESS_FRAMES = 3    # motion may only extend a track with this many frames of YOLO support

# -- Kalman filter / association -------------------------------------------- #
# Measurement noise (px std-dev, applied as R = diag(sigma^2, sigma^2)).
# YOLO boxes are tight and reliable; motion blobs are noisy (pitch texture,
# compression artifacts), so they get a much looser R -- the filter trusts
# them less and they pull the estimate around less.
YOLO_MEAS_STD = 4.0
MOTION_MEAS_STD = 14.0
# Process noise: how much we let the ball's velocity change frame-to-frame
# (accommodates gravity/drag curving the path, and bounce/impact direction
# changes). Position process noise is kept small -- the filter should trust
# its own dynamics between measurements, not drift on its own.
PROCESS_POS_STD = 1.5
PROCESS_VEL_STD = 8.0
GATE_CHI2_2DOF_99 = 9.21   # Mahalanobis^2 gate threshold (99% conf, 2 dof)
YOLO_GATE_RADIUS_PX = 48.0  # absolute-radius fallback so YOLO dets are never
                            # lost to a constant-velocity model running ahead
MAX_ACTIVE_TRACKS = 6       # cap on concurrently-tracked hypotheses
DUPLICATE_SUPPRESS_RADIUS_PX = 12.0  # overlapping YOLO boxes on one object:
                            # never spawn a duplicate track this close to one
                            # that is already being tracked

# -- ballistic consistency -------------------------------------------------- #
BALLISTIC_FIT_WINDOW = 8    # points used for the local quadratic fit
BALLISTIC_MIN_POINTS = 5    # minimum points before a fit is trusted
BALLISTIC_PENALTY_WEIGHT = 0.5  # how strongly fit residual inflates match cost

# -- re-seed (gap recovery) ------------------------------------------------- #
# A track that gaps out (misses > MAX_GAP_FRAMES) before the ball is hit can
# still be the real delivery. Once it dies we keep its ballistic extrapolation
# alive briefly and allow ONE re-seed: the first confident YOLO detection near
# the predicted position starts a successor track that is stitched to the dead
# one, so occluded-but-still-ballistic flights are recovered end-to-end.
RESEED_LOOKAHEAD_FRAMES = 8   # frames after death during which a re-seed may fire
RESEED_RADIUS_PX = 48.0       # max distance from the ballistic prediction for a re-seed

# -- motion-blob size limits (validated in v1) ------------------------------ #
MOTION_MIN_AREA = 8.0       # px^2 -- below this it's frame noise
MOTION_MAX_AREA = 1200.0    # px^2 -- above this it's a limb/body part
MOTION_MAX_ASPECT = 2.5     # width/height ratio -- elongated blobs are limbs

# -- track-validity limits (validated in v1) -------------------------------- #
# A delivery ball always travels a meaningful distance across the frame; a
# "track" pinned to one spot is YOLO confidently re-firing on a fixed round
# object (a stump, helmet, or lens flare). MIN_TRACK_SPREAD = the flight must
# span this much of the frame; MIN_TRACK_PATH = travel at least this far in
# total. A slow object whose box keeps growing (a person walking toward the
# camera) is rejected by the combined box-growth + speed rule.
MIN_TRACK_SPREAD_PX = 60.0
MIN_TRACK_PATH_PX = 100.0
MAX_SEGMENT_JUMP_PX = 80.0   # a same-track point can't teleport this far in 1 frame
BOX_GROWTH_LIMIT = 2.8       # box grew more than this AND moved slowly => a person
MIN_AVG_SPEED_PX_S = 400.0

# -- release / impact windowing (validated in v1) --------------------------- #
RELEASE_MIN_SPEED_FACTOR = 1.6
RELEASE_SUSTAIN_FRAMES = 3
IMPACT_ANGLE_THRESHOLD_DEG = 35.0
IMPACT_SPEED_DROP_FACTOR = 0.55
IMPACT_MIN_SPEED_PX_FRAME = 6.0  # ignore impact candidates with tiny prior speed (jitter)
MATCH_BASE_RADIUS = 30.0    # kept only as the release-heuristic's speed floor

# -- rendering ---------------------------------------------------------------- #
BOX_COLOR = (0, 0, 255)          # BGR red
DEFAULT_BALL_DIAMETER_PX = 16.0


@dataclass
class BallPoint:
    frame_idx: int
    timestamp_sec: float
    x: float
    y: float
    confidence: float        # 0.0 => extrapolated/predicted, >0 => detected
    detected: bool = True
    w: float = 0.0
    h: float = 0.0
    source: str = "motion"   # "yolo" | "motion" | "predicted"


# --------------------------------------------------------------------------- #
# Kalman filter: constant-velocity model, state = [x, y, vx, vy]
# --------------------------------------------------------------------------- #
class _CVKalman:
    __slots__ = ("x", "P")

    _F = np.array([[1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=float)
    _H = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]], dtype=float)
    _Q = np.diag([PROCESS_POS_STD**2, PROCESS_POS_STD**2,
                  PROCESS_VEL_STD**2, PROCESS_VEL_STD**2])

    def __init__(self, x0: float, y0: float, vx0: float = 0.0, vy0: float = 0.0,
                 init_std: float = 20.0):
        self.x = np.array([x0, y0, vx0, vy0], dtype=float)
        self.P = np.diag([init_std**2, init_std**2, (init_std * 2)**2, (init_std * 2)**2])

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self.x[:2].copy(), self.P.copy()

    def innovation(self, z: np.ndarray, meas_std: float) -> Tuple[float, np.ndarray]:
        """Mahalanobis^2 distance of measurement z under the current predicted state."""
        R = np.eye(2) * (meas_std ** 2)
        S = self._H @ self.P @ self._H.T + R
        y = z - self._H @ self.x
        maha2 = float(y.T @ np.linalg.solve(S, y))
        return maha2, S

    def update(self, z: np.ndarray, meas_std: float) -> None:
        R = np.eye(2) * (meas_std ** 2)
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        y = z - self._H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self._H) @ self.P


def _meas_std(source: str) -> float:
    return YOLO_MEAS_STD if source == "yolo" else MOTION_MEAS_STD


def _spread_path(pts: List[BallPoint]) -> Tuple[float, float]:
    """Bounding-box diagonal (spread, px) and path length (px) of a segment."""
    arr = np.array([[p.x, p.y] for p in pts], dtype=float)
    span = np.max(arr, axis=0) - np.min(arr, axis=0)
    spread = float(np.hypot(span[0], span[1]))
    seg = np.diff(arr, axis=0)
    path = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    return spread, path


def _quadratic_fit_residual(points: List[BallPoint]) -> Optional[float]:
    """RMS residual (px) of the last BALLISTIC_FIT_WINDOW points against a
    local quadratic (constant-acceleration) fit in x(t) and y(t). Lower is
    smoother/more ballistic. None if there isn't enough history yet."""
    pts = points[-BALLISTIC_FIT_WINDOW:]
    if len(pts) < BALLISTIC_MIN_POINTS:
        return None
    t = np.array([p.frame_idx for p in pts], dtype=float)
    t = t - t[0]
    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    try:
        cx = np.polyfit(t, xs, 2)
        cy = np.polyfit(t, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    rx = xs - np.polyval(cx, t)
    ry = ys - np.polyval(cy, t)
    return float(np.sqrt(np.mean(rx**2 + ry**2)))


def _predict_quadratic(points: List[BallPoint], frame_idx: int) -> Optional[np.ndarray]:
    """Extrapolate the local quadratic fit to `frame_idx`. None if not enough history."""
    pts = points[-BALLISTIC_FIT_WINDOW:]
    if len(pts) < BALLISTIC_MIN_POINTS:
        return None
    t = np.array([p.frame_idx for p in pts], dtype=float)
    t0 = t[0]
    t = t - t0
    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    try:
        cx = np.polyfit(t, xs, 2)
        cy = np.polyfit(t, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    tq = frame_idx - t0
    return np.array([np.polyval(cx, tq), np.polyval(cy, tq)])


def _owner_of(segment: List[BallPoint], finished: List[_Track]) -> Optional[_Track]:
    """Find which finished track a chosen segment belongs to. Split segments
    reuse the track's BallPoint objects, so the first point's identity locates
    the owning track even after trailing points were popped."""
    if not segment:
        return None
    first = segment[0]
    for t in finished:
        if any(p is first for p in t.points):
            return t
    return None


def _resolve_chain(t: _Track, finished: List[_Track]) -> List[_Track]:
    """Return the full re-seed chain [oldest_ancestor, ..., newest_successor]
    that `t` belongs to, walking the `stitched_from` links both ways."""
    by_id = {ft.track_id: ft for ft in finished}
    first = t
    while first.stitched_from is not None:
        parent = by_id.get(first.stitched_from)
        if parent is None:
            break
        first = parent
    chain: List[_Track] = [first]
    cur = first
    while True:
        nxt = next((ft for ft in finished if ft.stitched_from == cur.track_id), None)
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt
    return chain


def _full_chain_points(chain: List[_Track]) -> List[BallPoint]:
    """Concatenate a re-seeded chain (oldest ancestor first) into one trajectory.
    Each track's Kalman-predicted tail is dropped (it can carry a corrupted
    velocity from noise latches during the gap) and the holes are filled with
    ballistic bridge points extrapolated from the oldest ancestor's real
    detections -- a ball that re-appears near the ballistic prediction was in
    flight, so the gap is a straight stretch of the same parabola. Only the
    oldest ancestor's detections feed the fit: it is the sole pre-impact part
    of the chain, and a successor's post-deflection points would bend the
    parabola the bridge must follow."""
    yolo_pts: List[BallPoint] = [p for p in chain[0].points if p.source == "yolo"] if chain else []
    pts: List[BallPoint] = []
    for i, c in enumerate(chain):
        tail = _real_tail(c)
        pts.extend(tail)
        if i < len(chain) - 1 and tail:
            nxt = chain[i + 1]
            if nxt.points:
                bridge = _ballistic_bridge(yolo_pts, tail[-1], nxt.points[0])
                if bridge:
                    pts.extend(bridge)
    return pts


def _real_tail(track: _Track) -> List[BallPoint]:
    """A track's points up to (not including) its Kalman-predicted tail."""
    for i, p in enumerate(track.points):
        if p.source == "predicted":
            return track.points[:i]
    return track.points


def _ballistic_bridge(yolo_pts: List[BallPoint], tail_end: BallPoint,
                      succ_start: BallPoint) -> List[BallPoint]:
    """Synthesize the points between `tail_end` and `succ_start` by
    extrapolating the yolo-only quadratic fit one frame at a time."""
    if len(yolo_pts) < BALLISTIC_MIN_POINTS:
        return []
    span = succ_start.frame_idx - tail_end.frame_idx
    if span <= 1:
        return []
    bridge = []
    for f in range(tail_end.frame_idx + 1, succ_start.frame_idx):
        pos = _predict_quadratic(yolo_pts, f)
        if pos is None:
            return []
        frac = (f - tail_end.frame_idx) / float(span)
        ts = tail_end.timestamp_sec + (succ_start.timestamp_sec - tail_end.timestamp_sec) * frac
        w = max(1.0, np.median([p.w for p in yolo_pts]))
        h = max(1.0, np.median([p.h for p in yolo_pts]))
        bridge.append(BallPoint(f, ts, float(pos[0]), float(pos[1]), 0.0,
                                detected=False, w=float(w), h=float(h), source="predicted"))
    return bridge


@dataclass
class _Track:
    track_id: int
    kf: _CVKalman
    points: List[BallPoint] = field(default_factory=list)
    misses: int = 0
    yolo_hits: int = 0
    alive: bool = True
    last_yolo: int = -10**9   # frame_idx of the last YOLO match (liveness gate)
    death_frame: Optional[int] = None   # frame_idx the track gapped out on
    stitched_from: Optional[int] = None  # track_id of the dead track this one continues
    reseeded: bool = False    # this dead track already spawned a successor

    @property
    def hits(self) -> int:
        return len(self.points)

    def score(self) -> Tuple[float, float, int]:
        """Lexicographic ranking key for picking the winning track at the
        end: (1) trust real-detector evidence first, (2) reward smoother/
        more physically-plausible motion, (3) longer tracks as a tiebreak."""
        residual = _quadratic_fit_residual(self.points)
        smoothness = -residual if residual is not None else -1e9
        return (float(self.yolo_hits), smoothness, self.hits)


class BallTracker:
    """Detect + track the ball through a sequence of frames using
    multi-hypothesis, Kalman-filtered association."""

    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf_threshold: float = 0.1,
                 seed_min_conf: float = SEED_MIN_CONF,
                 model=None):
        # conf_threshold is deliberately low (0.1): at delivery-cam resolution
        # the ball is a small fast object that COCO's generic "sports ball"
        # class only scores at 0.05-0.35. The seed gate keeps low-conf false
        # positives from starting a track.
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
    def _candidates(self, frame: np.ndarray, prev_gray: np.ndarray
                     ) -> List[Tuple[float, float, float, float, float, str]]:
        """(x, y, confidence, w, h, source) candidates: YOLO ball detections + motion blobs."""
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
                        dets.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf, x2 - x1, y2 - y1, "yolo"))
            except Exception:
                pass
        dets.extend(self._motion_candidates(frame, prev_gray))
        return dets

    @staticmethod
    def _motion_candidates(frame: np.ndarray, prev_gray: Optional[np.ndarray]
                            ) -> List[Tuple[float, float, float, float, float, str]]:
        """Small fast-moving ball-sized blobs between consecutive frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return []
        diff = cv2.absdiff(gray, prev_gray)
        _, th = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        Runs multiple candidate tracks (_Track) in parallel across the whole
        clip, then picks the single best one -- ranked by score() -- that also
        passes the track-validity checks (fixed-object and person tracks are
        rejected; see module constants). The release/impact window is reported
        in `stats`; the returned trajectory is the validated segment (NOT
        pre-trimmed) so callers can clip at `impact_idx` themselves.
        """
        prev_gray = None
        total_frames = 0
        active: List[_Track] = []
        finished: List[_Track] = []
        zombies: List[_Track] = []
        next_id = 0

        for idx, ts, frame in frames:
            total_frames += 1
            cands = self._candidates(frame, prev_gray)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = gray

            active, done, next_id = self._step_all(idx, ts, cands, active, next_id, zombies)
            finished.extend(done)

        finished.extend(t for t in active if t.hits >= MIN_TRACK_FRAMES)

        survivors = [t for t in finished if t.hits >= MIN_TRACK_FRAMES]
        if not survivors:
            model_error = getattr(self, "model_error", None)
            stats = {"total_frames": total_frames, "n_tracks_considered": len(finished),
                     "outcome": "no_yolo_model" if model_error else "track_too_short"}
            stats.update(summarize([], fps=fps))
            return [], stats

        h, w = frames[0][2].shape[:2]
        chosen: Optional[List[BallPoint]] = None
        for t in sorted(survivors, key=lambda t: t.score(), reverse=True):
            for cand in sorted(self._split_segments(t.points), key=len, reverse=True):
                if len(cand) < MIN_TRACK_FRAMES:
                    continue
                # Drop trailing extrapolated points that run off the image
                # edge (the box must not drift out of view after impact).
                margin = 8
                while cand and (cand[-1].x < -margin or cand[-1].x > w + margin
                                or cand[-1].y < -margin or cand[-1].y > h + margin):
                    cand.pop()
                if len(cand) < MIN_TRACK_FRAMES:
                    continue
                if self._valid_track(cand):
                    chosen = cand
                    break
            if chosen is not None:
                break

        if chosen is None:
            model_error = getattr(self, "model_error", None)
            stats = {"total_frames": total_frames, "n_tracks_considered": len(finished),
                     "outcome": "no_yolo_model" if model_error else "track_too_short"}
            stats.update(summarize([], fps=fps))
            return [], stats

        # Stitch a re-seeded chain back together (whichever endpoint won), so
        # a delivery that gapped out mid-flight (occlusion, missed detections)
        # still spans release -> impact as one trajectory.
        owner = _owner_of(chosen, finished)
        if owner is not None:
            chain = _resolve_chain(owner, finished)
            if len(chain) > 1:
                merged = _full_chain_points(chain)
                if len(merged) >= len(chosen) and self._valid_track(merged):
                    chosen = merged

        _trimmed, window_info = self._trim_to_release_impact(chosen)
        impact_found = window_info.get("impact_idx") is not None
        if impact_found:
            outcome = "ok"
        elif window_info.get("release_idx") is None:
            outcome = "no_release_found"
        else:
            outcome = "release_no_impact"
        stats = {
            "total_frames": total_frames,
            "n_tracks_considered": len(finished),
            "winning_track_yolo_hits": best_yolo_hits_of(chosen),
            "winning_track_hits": len(chosen),
            "outcome": outcome,
        }
        stats.update(window_info)
        stats.update(summarize(chosen, fps=fps))
        return chosen, stats

    @staticmethod
    def _valid_track(cand: List[BallPoint]) -> bool:
        """A delivery ball always moves: reject segments pinned to one spot
        (YOLO confidently re-firing on a fixed round object every frame), and
        reject slow segments whose box balloons out -- that's a person walking
        toward the camera (or a motion-blob takeover), not a ball in flight.
        Box growth is max/min so a segment that *starts* on an already-big
        object is still caught; the speed gate is NET speed (end-to-end),
        which is immune to zig-zag Kalman/noise paths inflating path length.
        """
        spread, path = _spread_path(cand)
        dt = cand[-1].timestamp_sec - cand[0].timestamp_sec
        net_speed = (float(np.hypot(cand[-1].x - cand[0].x, cand[-1].y - cand[0].y)) / dt
                     if dt > 0 else 0.0)
        ws = [p.w for p in cand]
        growth = max(ws) / max(min(ws), 1.0)
        return (spread >= MIN_TRACK_SPREAD_PX and path >= MIN_TRACK_PATH_PX
                and not (growth > BOX_GROWTH_LIMIT and net_speed < MIN_AVG_SPEED_PX_S))

    def _step_all(self, idx: int, ts: float, cands, active: List[_Track], next_id: int,
                  zombies: Optional[List[_Track]] = None
                  ) -> Tuple[List[_Track], List[_Track], int]:
        """Advance every active track one frame, spawn new hypotheses from
        unclaimed candidates, and retire tracks that have gapped out.
        `zombies` (when given) receives gapped-out tracks so the re-seed pass
        below can re-acquire the ball a few frames later."""
        # 1. Predict every active track's next position.
        for t in active:
            t.kf.predict()

        # 2. Build every (track, candidate, cost) pair that passes the
        #    Mahalanobis gate, soft-penalized by ballistic-fit inconsistency.
        #    Motion candidates are only eligible for tracks with recent YOLO
        #    support (a limb must never latch onto a dead track).
        pairs = []
        for ti, t in enumerate(active):
            motion_ok = t.last_yolo >= idx - YOLO_LIVENESS_FRAMES
            for ci, (cx, cy, conf, cw, ch, src) in enumerate(cands):
                if src == "motion" and not motion_ok:
                    continue
                z = np.array([cx, cy])
                maha2, _ = t.kf.innovation(z, _meas_std(src))
                if src == "motion":
                    gate_ok = maha2 <= GATE_CHI2_2DOF_99
                else:
                    # YOLO detections must never be thrown away just because
                    # the constant-velocity model ran ahead of a decelerating
                    # ball (that was killing the impact signature). The tight
                    # R stays for the Kalman UPDATE; the gate additionally
                    # accepts YOLO dets within a generous absolute radius.
                    dist = float(np.hypot(z[0] - t.kf.x[0], z[1] - t.kf.x[1]))
                    gate_ok = maha2 <= GATE_CHI2_2DOF_99 or dist <= YOLO_GATE_RADIUS_PX
                if not gate_ok:
                    continue
                pred_quad = _predict_quadratic(t.points, idx)
                ballistic_penalty = 0.0
                if pred_quad is not None:
                    ballistic_penalty = BALLISTIC_PENALTY_WEIGHT * float(np.hypot(*(z - pred_quad)))
                cost = maha2 + ballistic_penalty
                pairs.append((src == "motion", cost, ti, ci))  # yolo pairs (False) sort first

        # 3. Greedy assignment: YOLO-source pairs considered before any
        #    motion-source pair, ties broken by lowest cost.
        pairs.sort(key=lambda p: (p[0], p[1]))
        claimed_tracks, claimed_cands = set(), set()
        assignment = {}  # ti -> ci
        for is_motion, cost, ti, ci in pairs:
            if ti in claimed_tracks or ci in claimed_cands:
                continue
            assignment[ti] = ci
            claimed_tracks.add(ti)
            claimed_cands.add(ci)

        # 4. Apply matches / misses.
        still_active, done = [], []
        for ti, t in enumerate(active):
            if ti in assignment:
                cx, cy, conf, cw, ch, src = cands[assignment[ti]]
                z = np.array([cx, cy])
                t.kf.update(z, _meas_std(src))
                t.points.append(BallPoint(idx, ts, float(cx), float(cy), conf,
                                           detected=True, w=cw, h=ch, source=src))
                t.misses = 0
                if src == "yolo":
                    t.yolo_hits += 1
                    t.last_yolo = idx
            else:
                t.misses += 1
                px, py = t.kf.x[0], t.kf.x[1]
                w, h = (t.points[-1].w, t.points[-1].h) if t.points else (DEFAULT_BALL_DIAMETER_PX, DEFAULT_BALL_DIAMETER_PX)
                t.points.append(BallPoint(idx, ts, float(px), float(py), 0.0,
                                           detected=False, w=w, h=h, source="predicted"))
            if t.misses > MAX_GAP_FRAMES:
                t.alive = False
                t.death_frame = idx
                done.append(t)
                if zombies is not None:
                    zombies.append(t)
            else:
                still_active.append(t)

        # 5. One-time re-seed (runs BEFORE spawning so a gapped-out ball gets
        #    first claim on a fresh detection): a track that gapped out
        #    mid-flight may be the real ball that simply wasn't detected for a
        #    stretch (occlusion, fast ball, YOLO dropout). While its ballistic
        #    prediction is still fresh, spawn a successor at the first confident
        #    YOLO detection near that prediction and remember the stitch.
        spawned_cands: set = set()
        if zombies is not None and len(still_active) < MAX_ACTIVE_TRACKS:
            still_zombies = []
            for z in zombies:
                if z.reseeded:
                    continue
                if z.death_frame is not None and idx - z.death_frame > RESEED_LOOKAHEAD_FRAMES:
                    continue
                still_zombies.append(z)
                # Fit the extrapolation on the REAL detections only: noise
                # motion blobs that latched on during the gap (and the
                # Kalman-predicted tail) can carry a corrupted velocity that
                # points the prediction at the wrong corner of the frame.
                yolo_pts = [p for p in z.points if p.source == "yolo"]
                pred = _predict_quadratic(yolo_pts, idx)
                if pred is None:
                    continue
                best, best_ci, best_d = None, None, None
                for ci, (cx, cy, conf, cw, ch, src) in enumerate(cands):
                    if ci in claimed_cands or src != "yolo" or conf < self.seed_min_conf:
                        continue
                    d = float(np.hypot(cx - pred[0], cy - pred[1]))
                    if d <= RESEED_RADIUS_PX and (best_d is None or d < best_d):
                        best, best_ci, best_d = (cx, cy, conf, cw, ch), ci, d
                if best is None:
                    continue
                cx, cy, conf, cw, ch = best
                if any(np.hypot(cx - o.points[-1].x, cy - o.points[-1].y)
                       <= DUPLICATE_SUPPRESS_RADIUS_PX for o in still_active if o.points):
                    continue
                kf = _CVKalman(cx, cy)
                new_t = _Track(track_id=next_id, kf=kf, stitched_from=z.track_id)
                next_id += 1
                new_t.points.append(BallPoint(idx, ts, float(cx), float(cy), conf,
                                               detected=True, w=cw, h=ch, source="yolo"))
                new_t.yolo_hits += 1
                new_t.last_yolo = idx
                still_active.append(new_t)
                spawned_cands.add(best_ci)
                z.reseeded = True
            zombies[:] = still_zombies

        # 6. Spawn new hypotheses from unclaimed candidates, bounded by
        #    MAX_ACTIVE_TRACKS. Only YOLO candidates may originate a track
        #    when a real model is loaded (validated v1 rule).
        if len(still_active) < MAX_ACTIVE_TRACKS:
            unclaimed = [(ci, c) for ci, c in enumerate(cands)
                         if ci not in claimed_cands and ci not in spawned_cands]
            if self._model is None:
                seedable = unclaimed
            else:
                seedable = [(ci, c) for ci, c in unclaimed if c[5] == "yolo" and c[2] >= self.seed_min_conf]
            seedable.sort(key=lambda ic: ic[1][2], reverse=True)  # highest confidence first
            room = MAX_ACTIVE_TRACKS - len(still_active)
            for ci, (cx, cy, conf, cw, ch, src) in seedable[:room]:
                # YOLO fires overlapping boxes on the same object; don't spawn
                # a duplicate track that will fight the real one for later
                # detections (a duplicate stole the f51 detection on
                # fast_left_1 and flattened the impact signature).
                if any(np.hypot(cx - o.points[-1].x, cy - o.points[-1].y)
                       <= DUPLICATE_SUPPRESS_RADIUS_PX
                       for o in still_active if o.points):
                    continue
                kf = _CVKalman(cx, cy)
                new_t = _Track(track_id=next_id, kf=kf)
                next_id += 1
                new_t.points.append(BallPoint(idx, ts, float(cx), float(cy), conf,
                                               detected=True, w=cw, h=ch, source=src))
                if src == "yolo":
                    new_t.yolo_hits += 1
                    new_t.last_yolo = idx
                still_active.append(new_t)

        return still_active, done, next_id

    @staticmethod
    def _split_segments(track: List[BallPoint]) -> List[List[BallPoint]]:
        # A gap in frame_idx means the track reset (a candidate went unclaimed
        # long enough that tracking restarted) -- the points on either side
        # belong to different episodes and must never be merged. The jump
        # guard catches the same-tick case where a reset and a re-seed land
        # next to each other in index space (f41 -> f42) but are two
        # different objects.
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
        """Heuristically trim a trajectory down to the release -> impact window.

        There's no bowler/bat detector here, so this works purely off the
        ball's own motion signature:
          - "release": the point where frame-to-frame speed first jumps to a
            sustained high value (the ball leaving the hand).
          - "impact": the first sharp change in direction (or a sudden speed
            drop) after release, corresponding to the ball hitting the bat,
            pad, or ground.

        Uses windowed (median) speeds so a single jittery motion-blob point
        can't masquerade as a bat/pad contact: `s_next` is a 4-frame median so
        a one-frame transient stall (a point where the ball appears to pause
        but resumes next frame) doesn't fire the speed-drop test, while a
        genuine bat/pad stop drops speed for several consecutive frames and
        still registers. Requires the velocity before a candidate impact to be
        non-trivial (a nearly-stationary point makes the angle ill-conditioned).
        """
        info = {"release_idx": None, "impact_idx": None, "release_found": False,
                "trimmed": False}
        if len(track) < RELEASE_SUSTAIN_FRAMES + 2:
            return track, info

        pts = np.array([[p.x, p.y] for p in track])
        vel = np.diff(pts, axis=0)
        speed = np.hypot(vel[:, 0], vel[:, 1])

        baseline = float(np.median(speed[: min(5, len(speed))])) if len(speed) else 0.0
        threshold = max(baseline * RELEASE_MIN_SPEED_FACTOR, MATCH_BASE_RADIUS * 0.5)
        release_i = None
        for i in range(len(speed) - RELEASE_SUSTAIN_FRAMES + 1):
            if np.all(speed[i:i + RELEASE_SUSTAIN_FRAMES] >= threshold):
                release_i = i
                break
        if release_i is None:
            release_i = 0

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
            s_next = float(np.median(speeds[i:i + 4]))
            speed_drop = s_next < s_prev * IMPACT_SPEED_DROP_FACTOR
            if angle_deg >= IMPACT_ANGLE_THRESHOLD_DEG or speed_drop:
                impact_i = i + 1
                break

        release_idx = track[release_i].frame_idx
        info["release_found"] = release_i > 0
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


def best_yolo_hits_of(segment: List[BallPoint]) -> int:
    """Count YOLO-sourced points in a segment (used for stats)."""
    return sum(1 for p in segment if p.source == "yolo")


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
    by_idx = {p.frame_idx: p for p in track}
    out = []
    for idx, ts, frame in frames:
        img = frame.copy()
        pt = by_idx.get(idx)
        if pt is not None:
            w = pt.w if pt.w > 0 else DEFAULT_BALL_DIAMETER_PX
            h = pt.h if pt.h > 0 else DEFAULT_BALL_DIAMETER_PX
            hh, ww = img.shape[:2]
            x1 = int(round(pt.x - w / 2))
            y1 = int(round(pt.y - h / 2))
            x2 = int(round(pt.x + w / 2))
            y2 = int(round(pt.y + h / 2))
            x1 = max(0, min(ww - 1, x1))
            y1 = max(0, min(hh - 1, y1))
            x2 = max(0, min(ww - 1, x2))
            y2 = max(0, min(hh - 1, y2))
            if pt.detected:
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                label = "ball" + (f" {pt.confidence:.2f}" if pt.confidence > 0 else "")
                _draw_label_tag(img, x1, y1, label, box_color)
            else:
                _draw_dashed_rect(img, (x1, y1), (x2, y2), box_color, 1)
                _draw_label_tag(img, x1, y1, "ball (est)", box_color)
        out.append((idx, ts, img))
    return out


def _draw_label_tag(img, x1, y1, text, color):
    """Filled label strip just above the box, sized with cv2.getTextSize.

    The canonical detection-render pattern (OpenCV's mask_rcnn sample /
    Roboflow's bbox-label tutorial): measure the text, draw a filled
    background strip above the box, then the text in it. Clamped to the top
    edge so balls near the top of the frame don't push the tag off-screen.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    tag_h = th + base + 6
    top = max(0, y1 - tag_h)
    cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + tag_h), color, -1)
    cv2.putText(img, text, (x1 + 3, top + th + 3), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


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
    x1, y1 = pt1
    x2, y2 = pt2
    for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                 ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        _draw_dashed_line(img, a, b, color, thickness, dash)


def write_mp4(frames, path: str, fps: float) -> str:
    """Encode frames (idx, ts, img) as an H.264 MP4 via imageio-ffmpeg's
    bundled ffmpeg (libx264). Returns the output path."""
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
