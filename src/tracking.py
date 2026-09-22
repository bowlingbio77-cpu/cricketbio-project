"""
Stage 3: Bowler Tracking (ByteTrack)

Ultralytics ships ByteTrack as a built-in tracker, so tracking is exposed via
`model.track(...)` rather than a separate library. This module wraps that,
returning every tracked person.

The bowler is then selected with a *cricket-aware evidence score* at the end
of the clip (see `select_bowler_track_with_meta`): the bowler is the person
who runs a straight line at speed over a good fraction of the clip
(high cumulative motion, lots of clearly-moving frames), covers real ground
in the frame (the run-up + delivery spans the frame), and grows in size as
they close on the camera (a *temporal* size-change signal, never absolute
bbox area). Track identity is LOCKED once selected: only that track's
detections are ever boxed, a brief occlusion is bridged by re-stitching the
same person, and when the bowler genuinely leaves the frame nobody else is
substituted.

Falls back to a lightweight IoU-based tracker (greedy nearest-bbox matching)
when ultralytics isn't installed, so the pipeline still runs end-to-end offline.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
from . import config
from .detection import Detection, _bbox_area

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except (ImportError, OSError):
    _HAS_ULTRALYTICS = False


@dataclass
class Track:
    track_id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[tuple] = field(default_factory=list)

    def __len__(self):
        return len(self.frames)


class BowlerTracker:
    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf_threshold: float = config.DETECTION_CONF_THRESHOLD,
                 detector: "BowlerDetector" = None):
        self.conf_threshold = conf_threshold
        self.fallback_reason: Optional[str] = None
        self.backend = "bytetrack" if _HAS_ULTRALYTICS else "iou_fallback"
        if self.backend == "bytetrack":
            # G2: a weights file that can't load (missing + no network, corrupt,
            # version mismatch) must NOT throw out of the constructor -- that
            # would silently disable ALL bowler detection. Degrade to the IoU
            # fallback tracker (with its own HOG detector if YOLO also fails)
            # and record why so the pipeline flags the reduced confidence.
            try:
                from .detection import resolve_weights
                self.model = YOLO(resolve_weights(weights))
            except Exception as exc:
                self.fallback_reason = (
                    f"ByteTrack could not be initialised ({type(exc).__name__}: {exc}); "
                    "degraded to the IoU fallback tracker (lower confidence)."
                )
                self.backend = "iou_fallback"
        if self.backend == "iou_fallback":
            from .detection import BowlerDetector
            detector = detector or BowlerDetector(weights=weights,
                                                  conf_threshold=conf_threshold)
            self.detector = detector
            if getattr(detector, "fallback_reason", None):
                self.fallback_reason = (
                    f"{self.fallback_reason or 'IoU fallback tracker'}; "
                    f"{detector.fallback_reason}"
                )
        self._iou_tracks: Dict[int, Track] = {}
        self._next_id = 0

    def track_video(self, video_path: str) -> Dict[int, Track]:
        """Run tracking over an entire video file, return {track_id: Track}."""
        if self.backend == "bytetrack":
            return self._track_bytetrack(video_path)
        raise RuntimeError(
            "IoU fallback tracker works frame-by-frame; call track_frames() in a loop instead."
        )

    def track_frames(self, frames) -> Dict[int, Track]:
        """
        Run tracking over an in-memory sequence of frames (already decoded,
        e.g. after preprocessing) so the video file is only read once.

        `frames`: iterable of (frame_idx, frame_bgr) tuples.
        Returns {track_id: Track} with bboxes in the same frame coordinates.
        """
        if self.backend == "bytetrack":
            return self._track_bytetrack_frames(frames)
        tracks: Dict[int, Track] = {}
        for idx, frame in frames:
            dets = self.detector.detect(frame, idx)
            assigned = self.track_frame(idx, dets)
            for tid, bbox in assigned.items():
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].frames.append(idx)
                tracks[tid].bboxes.append(bbox)
        return tracks

    def _track_bytetrack_frames(self, frames) -> Dict[int, Track]:
        """ByteTrack over a list of (frame_idx, frame_bgr) tuples."""
        source = [frame for _, frame in frames]
        tracks: Dict[int, Track] = {}
        if not source:
            return tracks
        results = self.model.track(
            source=source, classes=[config.BOWLER_CLASS_ID],
            conf=self.conf_threshold, tracker=config.BYTETRACK_CONFIG,
            persist=True, stream=True, verbose=False,
        )
        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue
            ids = r.boxes.id.int().tolist()
            xyxys = r.boxes.xyxy.tolist()
            for tid, box in zip(ids, xyxys):
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].frames.append(frame_idx)
                tracks[tid].bboxes.append(tuple(box))
        return tracks

    def _track_bytetrack(self, video_path: str) -> Dict[int, Track]:
        tracks: Dict[int, Track] = {}
        results = self.model.track(
            source=video_path, classes=[config.BOWLER_CLASS_ID],
            conf=self.conf_threshold, tracker=config.BYTETRACK_CONFIG,
            persist=True, stream=True, verbose=False,
        )
        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue
            ids = r.boxes.id.int().tolist()
            xyxys = r.boxes.xyxy.tolist()
            for tid, box in zip(ids, xyxys):
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].frames.append(frame_idx)
                tracks[tid].bboxes.append(tuple(box))
        return tracks

    # --- Fallback: simple greedy IoU tracker, one frame at a time ---
    def track_frame(self, frame_idx: int, detections: List[Detection]) -> Dict[int, tuple]:
        assigned = {}
        used_tracks = set()
        for det in detections:
            best_id, best_iou = None, 0.3  # IoU threshold to continue a track
            for tid, tr in self._iou_tracks.items():
                if tid in used_tracks or not tr.bboxes:
                    continue
                iou = _iou(det.bbox, tr.bboxes[-1])
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                best_id = self._next_id
                self._iou_tracks[best_id] = Track(track_id=best_id)
                self._next_id += 1
            self._iou_tracks[best_id].frames.append(frame_idx)
            self._iou_tracks[best_id].bboxes.append(det.bbox)
            used_tracks.add(best_id)
            assigned[best_id] = det.bbox
        return assigned

    def get_iou_tracks(self) -> Dict[int, Track]:
        return self._iou_tracks


# --------------------------------------------------------------------------- #
# Cricket-aware bowler selection (replaces "longest track x largest bbox").
#
# Why the old heuristic was wrong: a batsman stands in the frame for the whole
# clip with a big, steady bbox, so `len * avg_area` always crowned the batsman.
# A bowler distinguishes himself by HOW HE MOVES, not by how big he is:
#   * runs a straight, consistent line at real speed (run-up),
#   * covers real ground in the frame (run-up + delivery spans the image),
#   * changes size over time (closing the ground as the run-up progresses),
#   * then decelerates to plant and deliver (delivery-phase movement).
# Absolute bbox size is deliberately NOT a ranking feature -- it only enters as
# a *growth ratio over time*, a run-up pattern, never as a "biggest person" test.
# --------------------------------------------------------------------------- #
BOWLER_W_MOTION = 2.0    # body-normalized locomotion: mean step / OWN bbox diag
BOWLER_W_RELATIVE = 2.0  # locomotion above the clip's shared camera drift (body units)
BOWLER_W_ACTIVE = 1.5    # fraction of steps that are real locomotion (>= ACTIVE_BODY_MIN)
BOWLER_W_GROWTH = 1.0    # monotonic apparent-size trend (closing range on the runway)
BOWLER_W_SPAN = 0.5      # temporal continuity: how much of the clip is covered
BOWLER_W_STRAIGHT = 0.8  # straightness of the run-up path (not random wander)
BOWLER_W_RANGE = 0.3     # spatial coverage of the center path in the frame
BOWLER_W_BAND = 0.3      # mostly inside the frame's central band (the runway)
BOWLER_W_DECEL = 0.4     # bowler slows late in the clip (plant + delivery)
BOWLER_W_DIRECTED = 0.6  # direction persistence (a run-up is one-directional)
BOWLER_W_DELIVERY = 0.4  # late stride "plant" signature (view-agnostic peak step)
BOWLER_W_VERT = 0.15     # legacy vertical-motion term, kept minimal: it encodes a
                          # behind-bowler camera view; every other broadcast/side view
                          # shows the bowler moving horizontally, so it must not dominate
                          # (see scripts/debug_bowler_selection.py ADVER-1 vs CONTROL).

# Body-scaled locomotion threshold: a genuine human step moves the center >= 5%
# of the person's OWN bbox diagonal per frame. Raw pixels reward whoever is
# nearest the camera (a fidgeting batsman) over the real (far) bowler, so all
# motion evidence is divided by the actor's own size -- view/zoom-invariant.
BOWLER_ACTIVE_BODY_MIN = 0.05

# No-bowler gate: if the best track shows almost no real locomotion (in body
# units), the clip has no confidently-detected bowler -- return None (honest)
# instead of boxing a static batsman/keeper.
BOWLER_MIN_MOTION_FRACTION = 0.08
BOWLER_MIN_ACTIVE_FRACTION = 0.05

# Selection confirmation gate: auto-lock only when the winner's evidence is
# strong enough AND clearly ahead of the runner-up. When the margin collapses
# (the reported bowler <-> batsman tie), the clip reports "BOWLER NOT CONFIRMED"
# so the UI asks the user to lock a candidate instead of silently analysing the
# wrong person.
BOWLER_CONFIRM_MIN_SCORE = 0.35
BOWLER_CONFIRM_MIN_MARGIN = 0.06

# Batting-stance classifier: a non-bowler person is "static" (striker /
# non-striker) only when its body-normalized speed stays below the same
# locomotion bar; the two static batting-end players are picked by projecting
# their mean centers onto the bowler's own run axis.
BOWLER_STANCE_MIN_SEP_FRAC = 0.12  # minimum axis separation (x frame width) to label both ends

# Identity-stitch limits: merge a continuation track back into the "locked"
# bowler only when it starts soon after the bowler's last frame (brief
# occlusion / miss / camera cut) AND really is the same physical person. For a
# MOVING bowler the acceptance window is projected with the bowler's own
# velocity (speed px/frame x gap frames), because a run-up bowler can be tens
# of pixels away after an occlusion; a static person is held to its own box
# diagonal instead. Anything else means the bowler was temporarily lost --
# prefer no box over a wrong identity.
BOWLER_STITCH_MAX_GAP = 8
BOWLER_STITCH_MIN_IOU = 0.12
BOWLER_STITCH_CENTER_MULT = 0.5      # x own-last-box max-diagonal fraction
BOWLER_STITCH_VEL_MULT = 1.5         # x bowler speed x gap (projected window)
BOWLER_STITCH_APPEARANCE_MIN = 0.30  # minimum HSV-histogram correlation for a re-stitch

_FRAME_DIMS_DEFAULT = (360, 640)      # used only when caller omits frame_dims


def _bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _consecutive_steps(tr: Track) -> np.ndarray:
    """Per-step bbox-center displacements (px) between successive frames of a
    track. Steps spanning a big internal frame gap (>2) are skipped so a
    ByteTrack re-acquisition jump never registers as motion."""
    steps = []
    for k in range(1, len(tr.frames)):
        prev, cur = tr.frames[k - 1], tr.frames[k]
        if cur - prev > 2:
            continue
        c0 = _bbox_center(tr.bboxes[k - 1])
        c1 = _bbox_center(tr.bboxes[k])
        steps.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])))
    if steps:
        return np.asarray(steps, dtype=float)
    return np.zeros(1, dtype=float)


def _track_centers(tr: Track) -> np.ndarray:
    return np.asarray([_bbox_center(b) for b in tr.bboxes], dtype=float)


def _bbox_diag(bbox) -> float:
    return float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def _step_sequences(tr: Track):
    """Per-consecutive-step evidence arrays used by the scorer:
    (end_frames, dx, dy, mags, scales) where ``scales`` is the actor's OWN
    bbox diagonal at that step (max of the two boxes). Steps over a big
    internal frame gap (>2) are dropped so a ByteTrack re-acquisition jump
    never registers as motion."""
    end_frames, dxs, dys, mags, scales = [], [], [], [], []
    for k in range(1, len(tr.frames)):
        if tr.frames[k] - tr.frames[k - 1] > 2:
            continue
        c0 = _bbox_center(tr.bboxes[k - 1])
        c1 = _bbox_center(tr.bboxes[k])
        _dx = float(c1[0] - c0[0])
        _dy = float(c1[1] - c0[1])
        end_frames.append(tr.frames[k])
        dxs.append(_dx)
        dys.append(_dy)
        mags.append(float(np.hypot(_dx, _dy)))
        scales.append(max(_bbox_diag(tr.bboxes[k - 1]), _bbox_diag(tr.bboxes[k]), 1e-6))
    return (np.asarray(end_frames, dtype=int), np.asarray(dxs, dtype=float),
            np.asarray(dys, dtype=float), np.asarray(mags, dtype=float),
            np.asarray(scales, dtype=float))


def _mean_body_speed(tr: Track) -> float:
    """Mean center speed in units of the actor's OWN bbox diagonal per frame --
    scale/zoom-invariant, so a far-away small bowler walking is not swamped by a
    near-camera fidgeting batsman."""
    _ef, _dx, _dy, mags, scales = _step_sequences(tr)
    if not len(mags):
        return 0.0
    return float(np.mean(mags / scales))


def _score_components(tr: Track, diag: float, n_total: int,
                      frame_w: float, frame_h: float,
                      rel_mags: Optional[np.ndarray] = None) -> dict:
    """Per-track normalized (0..1) evidence components for bowler selection.

    Motion evidence is expressed in BODY units (actor's own bbox diagonal):
      - a raw-pixel path rewards whoever is nearest the camera, which is how a
        footworking/advancing batsman used to out-move the real bowler;
      - ``rel_mags`` optionally supplies the same steps AFTER subtracting the
        clip's shared camera drift, so a sustained pan inflating every static
        person is cancelled (the bowler keeps moving against/along the pan).
    """
    _ef, dxs, dys, steps, scales = _step_sequences(tr)
    body = steps / scales
    n = len(body)

    motion = float(np.tanh(4.0 * float(np.mean(body)))) if n else 0.0
    active = (float(np.mean(body >= BOWLER_ACTIVE_BODY_MIN)) if n else 0.0)
    relative = 0.0
    if rel_mags is not None and len(rel_mags) == n and n:
        relative = float(np.tanh(4.0 * float(np.mean(rel_mags))))
    span = len(tr) / max(1, n_total)
    path = float(np.sum(steps))

    areas = np.asarray([_bbox_area(b) for b in tr.bboxes], dtype=float)
    growth = 0.0
    if len(areas) >= 3:
        med_area = max(float(np.median(areas)), 1e-6)
        diffs = np.diff(areas)
        nonzero = diffs[diffs != 0]
        if len(nonzero) and abs(areas[-1] - areas[0]) > 1e-6:
            direction = np.sign(areas[-1] - areas[0])
            consistent = float(np.mean(np.sign(nonzero) == direction))
            mag = float(np.clip(abs(areas[-1] - areas[0]) / med_area, 0.0, 2.0))
            growth = float(np.clip(consistent * min(1.0, 0.5 * mag), 0.0, 1.0))

    centers = _track_centers(tr)
    straight = 0.0
    if len(centers) >= 2 and path > 1e-6:
        net = float(np.hypot(centers[-1][0] - centers[0][0],
                             centers[-1][1] - centers[0][1]))
        straight = float(np.clip(net / path, 0.0, 1.0))

    range_cov = 0.0
    if len(centers) >= 2:
        span_x = float(np.ptp(centers[:, 0]))
        span_y = float(np.ptp(centers[:, 1]))
        range_cov = float(np.hypot(span_x, span_y)) / max(diag, 1e-6)

    band = 1.0
    if len(centers) and frame_w > 0 and frame_h > 0:
        cx, cy = centers[:, 0], centers[:, 1]
        lo, hi = 0.15 * frame_w, 0.85 * frame_w
        inner = float(np.mean((cx >= lo) & (cx <= hi)))
        v_span = float(np.ptp(cy))
        v_cov = min(1.0, v_span / max(0.6 * frame_h, 1e-6))
        band = float(inner * (0.4 + 0.6 * v_cov))

    decel = 0.0
    if n >= 8:
        k = max(2, int(0.2 * n))
        late = float(np.median(steps[-k:]))
        overall = float(np.median(steps))
        if overall > 1e-6 and late < overall * 0.85:
            decel = float(np.clip(1.0 - late / (overall * 0.85), 0.0, 1.0))

    # Direction persistence: a run-up keeps one heading; jitter/batting
    # footwork toggles. View-agnostic (horizontal OR vertical run -> ~1).
    directed = 0.0
    if n >= 2 and np.any(steps > 1e-6):
        hx = float(np.sum(dxs))
        hy = float(np.sum(dys))
        hn = np.hypot(hx, hy)
        if hn > 1e-6:
            hx, hy = hx / hn, hy / hn
            align = (dxs * hx + dys * hy) / np.maximum(steps, 1e-6)
            directed = float(np.mean(align > 0.5))

    # Legacy vertical-motion term (kept for the metric shape and old tests but
    # weighted near-zero): only meaningful for a behind-bowler camera.
    vert = 0.0
    if len(centers) >= 2:
        dy = np.abs(np.diff(centers[:, 1]))
        dx = np.abs(np.diff(centers[:, 0]))
        total_dist = dy + dx + 1e-6
        vert = float(np.mean(dy / total_dist))

    # Delivery-phase signature, view-agnostic: the LATE stride of a real run-up
    # culminates in one pronounced "plant" step near the end of the clip.
    delivery = 0.0
    if n >= 10:
        k = max(2, n // 5 * 2)  # last 40%
        late_mags = steps[-k:]
        overall_med = float(np.median(steps))
        peak = float(late_mags.max())
        if overall_med > 1e-6 and peak > 1.4 * overall_med:
            peak_frac = float(np.argmax(steps)) / (n - 1)
            if peak_frac >= 0.6:
                delivery = float(np.clip((peak / overall_med - 1.4) / 1.0, 0.0, 1.0))

    return {
        "motion": float(motion),
        "active": float(active),
        "span": float(span),
        "growth": float(growth),
        "straight": float(straight),
        "range": float(range_cov),
        "band": float(band),
        "decel": float(decel),
        "directed": float(directed),
        "delivery": float(delivery),
        "vert": float(vert),
        "relative": float(relative),
    }


def _weighted_score(comps: dict) -> float:
    w = {
        "motion": BOWLER_W_MOTION,
        "relative": BOWLER_W_RELATIVE,
        "active": BOWLER_W_ACTIVE,
        "growth": BOWLER_W_GROWTH,
        "span": BOWLER_W_SPAN,
        "straight": BOWLER_W_STRAIGHT,
        "range": BOWLER_W_RANGE,
        "band": BOWLER_W_BAND,
        "decel": BOWLER_W_DECEL,
        "directed": BOWLER_W_DIRECTED,
        "delivery": BOWLER_W_DELIVERY,
        "vert": BOWLER_W_VERT,
    }
    total = float(sum(w[k] * comps[k] for k in w))
    return total / float(sum(w.values()))


def _confidence(score: float) -> float:
    return float(np.clip(1.0 - np.exp(-3.0 * score), 0.0, 1.0))


def _last_speed(tr: Track, window: int = 5) -> float:
    """Mean center velocity (px/frame) of the bowler's most recent contiguous
    steps -- used to project where the bowler SHOULD be after a short gap."""
    idxs = list(range(max(0, len(tr.frames) - window - 1), len(tr.frames)))
    total_d = 0.0
    total_t = 0
    for a, b in zip(idxs[:-1], idxs[1:]):
        dt = tr.frames[b] - tr.frames[a]
        if dt <= 2:
            ca = np.asarray(_bbox_center(tr.bboxes[a]))
            cb = np.asarray(_bbox_center(tr.bboxes[b]))
            total_d += float(np.hypot(*(cb - ca)))
            total_t += dt
    if total_t <= 0:
        return 0.0
    return total_d / total_t


def _same_person(winner_tr: Track, cand_box, cand_frame: int,
                 frame_diag: float) -> bool:
    """Spatial-continuity test for stitching a continuation back to a bowler.

    Accepts the candidate when (a) the boxes overlap, or (b) the candidate's
    center lies inside a window projected from the bowler's own motion: last
    center + velocity x gap, with a tolerance that scales with the bowler's
    box diagonal (a static bowler) or with its running speed (a run-up bowler
    that keeps moving while occluded)."""
    w_last_box = winner_tr.bboxes[-1]
    w_last_frame = winner_tr.frames[-1]
    if _iou(w_last_box, cand_box) >= BOWLER_STITCH_MIN_IOU:
        return True
    own_diag = np.hypot(w_last_box[2] - w_last_box[0], w_last_box[3] - w_last_box[1])
    gap = cand_frame - w_last_frame
    speed = _last_speed(winner_tr)
    ca = np.asarray(_bbox_center(w_last_box))
    cb = np.asarray(_bbox_center(cand_box))
    predicted = ca + speed * gap
    dist = float(np.hypot(*(cb - predicted)))
    tol = max(BOWLER_STITCH_CENTER_MULT * own_diag, 0.04 * frame_diag)
    if speed > 0 and gap > 0:
        tol = max(tol, BOWLER_STITCH_VEL_MULT * speed * gap)
    return dist <= tol


def _crop_frame_region(frame, bbox, fx1=0.20, fy1=0.10, fx2=0.80, fy2=0.75):
    """Grab the upper-body (torso) region of a bbox from a full frame.

    The torso band is used for the appearance check because the full bbox is
    dominated by background (pitch, sky, stumps) that correlates across
    completely different people."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx1, cy1 = int(x1 + fx1 * bw), int(y1 + fy1 * bh)
    cx2, cy2 = int(x1 + fx2 * bw), int(y1 + fy2 * bh)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]


def _hsv_hist_corr(img_a, img_b) -> float:
    """Correlation (0..1, 1 = identical colour layout) of 3-channel HSV
    histograms of two crops. Robust enough to reject a visually distinct person
    even when the backgrounds are similar."""
    if img_a is None or img_b is None or img_a.size == 0 or img_b.size == 0:
        return 0.0
    try:
        ha = cv2.calcHist([cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)], [0, 1], None,
                          [16, 16], [0, 180, 0, 256])
        hb = cv2.calcHist([cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)], [0, 1], None,
                          [16, 16], [0, 180, 0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    except cv2.error:
        return 0.0


def _appearance_match(winner: Track, cand_frame: int, cand_box, frame_provider) -> bool:
    """HSV-histogram consistency between the bowler's last seen crop and the
    candidate continuation's first crop. Requires `frame_provider(frame_idx,
    bbox) -> np.ndarray` (the caller supplies the graded frames)."""
    last_box = winner.bboxes[-1]
    last_frame = winner.frames[-1]
    img_a = frame_provider(last_frame, last_box)
    img_b = frame_provider(cand_frame, cand_box)
    if img_a is None or img_b is None:
        return True  # no visual evidence -> keep the spatial test's verdict
    corr = _hsv_hist_corr(_crop_frame_region(img_a, last_box),
                          _crop_frame_region(img_b, cand_box))
    return corr >= BOWLER_STITCH_APPEARANCE_MIN


def _stitch_continuations(winner: Track, tracks: Dict[int, Track],
                          frame_diag: float,
                          frame_provider: Optional[Callable] = None) -> Track:
    """Identity lock keeps one bowler; a *continuation of the same person*
    (born after a brief occlusion or a tracker ID reset) is re-stitched to it.
    A different person -- fielder, keeper, umpire, batsman -- is never adopted,
    even briefly; if the bowler is lost we prefer a gap over a wrong box.
    When `frame_provider` is supplied the spatial test is tightened with an
    appearance (HSV-histogram) check so a visually distinct person that merely
    happens to be spatially continuous is NOT stitched in."""
    if winner is None or not tracks:
        return winner
    w_index = winner.track_id
    w_frames = set(winner.frames)
    changed = True
    while changed:
        changed = False
        w_last = winner.frames[-1]
        for tid, tr in tracks.items():
            if tid == w_index or not tr.frames or not tr.bboxes:
                continue
            if tr.frames[-1] <= w_last:
                continue
            gap = tr.frames[0] - w_last
            if not (-1 <= gap <= BOWLER_STITCH_MAX_GAP):
                continue
            if not _same_person(winner, tr.bboxes[0], tr.frames[0], frame_diag):
                continue
            if frame_provider is not None and not _appearance_match(
                    winner, tr.frames[0], tr.bboxes[0], frame_provider):
                continue
            extra = [(f, b) for f, b in zip(tr.frames, tr.bboxes) if f not in w_frames and f > w_last]
            if extra:
                for f, b in extra:
                    winner.frames.append(f)
                    winner.bboxes.append(b)
                w_frames.update(f for f, _ in extra)
                changed = True
    return winner


def score_bowler_tracks(tracks: Dict[int, Track], frame_dims=None,
                        total_frames: Optional[int] = None) -> List[dict]:
    """Score every track by cricket evidence. Returns a ranked (descending)
    list of dicts: track_id, score, confidence, per-component scores, and
    coverage metadata -- used both for selection and for debug overlays."""
    if not tracks:
        return []
    frame_h, frame_w = frame_dims if frame_dims is not None else _FRAME_DIMS_DEFAULT
    frame_diag = float(np.hypot(frame_w, frame_h))
    n_total = total_frames if (total_frames and total_frames > 0) else \
        max((len(tr.frames) for tr in tracks.values()), default=1)
    # Shared camera-drift estimate: mean step vector over every track that has
    # a step ending at the same frame. A sustained pan moves everyone equally;
    # a bowler running against/along the pan keeps a large residual, while a
    # static batsman that merely "rides" the pan is cancelled out.
    pair_steps: Dict[int, list] = {}
    for _tid, tr in tracks.items():
        ef, dxs, dys, _m, _s = _step_sequences(tr)
        for i, f in enumerate(ef):
            pair_steps.setdefault(int(f), []).append((dxs[i], dys[i]))
    drift = {}
    for f, vecs in pair_steps.items():
        if len(vecs) >= 2:
            drift[f] = tuple(np.mean(np.asarray(vecs), axis=0))
    ranked = []
    for tid, tr in sorted(tracks.items()):
        ef, dxs, dys, _m, scales = _step_sequences(tr)
        rel_mags = None
        if len(ef):
            resid = []
            for i, f in enumerate(ef):
                d = drift.get(int(f))
                rd = (dxs[i] - d[0]) if d is not None else dxs[i]
                rdy = (dys[i] - d[1]) if d is not None else dys[i]
                resid.append(float(np.hypot(rd, rdy)) / scales[i])
            rel_mags = np.asarray(resid, dtype=float)
        comps = _score_components(tr, frame_diag, n_total, float(frame_w),
                                  float(frame_h), rel_mags=rel_mags)
        score = _weighted_score(comps)
        ranked.append({
            "track_id": tid,
            "score": float(score),
            "confidence": float(_confidence(score)),
            "n_frames": len(tr),
            "start_frame": tr.frames[0] if tr.frames else None,
            "end_frame": tr.frames[-1] if tr.frames else None,
            **comps,
        })
    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked


def count_bowler_identity_switches(tracks: Dict[int, Track], final_track_id: int,
                                   frame_dims=None,
                                   total_frames: Optional[int] = None) -> int:
    """How many half-clip windows produced a different top bowler candidate than
    the final lock -- an observable, honest measure of bowler <-> batsman
    switching (returns 0 for a stable lock, >0 when the top scorer churns)."""
    fs = [f for tr in tracks.values() for f in tr.frames] or [0]
    n_total = total_frames if (total_frames and total_frames > 0) else int(max(fs)) + 1
    edges = np.linspace(0, n_total, 3)
    switches = 0
    for a, b in zip(edges[:-1], edges[1:]):
        sub = {tid: Track(track_id=tid,
                          frames=[f for f in tr.frames if a <= f < b],
                          bboxes=[bb for f, bb in zip(tr.frames, tr.bboxes)
                                  if a <= f < b])
               for tid, tr in tracks.items() if tr.frames}
        sub = {tid: tr for tid, tr in sub.items() if tr.frames}
        if not sub:
            continue
        ranked = score_bowler_tracks(sub, frame_dims, n_total)
        if ranked and ranked[0]["track_id"] != final_track_id:
            switches += 1
    return switches


def select_bowler_track_with_meta(tracks: Dict[int, Track], frame_dims=None,
                                  total_frames: Optional[int] = None,
                                  frame_provider: Optional[Callable] = None
                                  ) -> "tuple[Optional[Track], Optional[dict]]":
    """Choose the bowler. Returns (best_track, meta).

    The winner is the highest cricket-evidence track, then its identity is
    locked: continuations of the same physical person are re-stitched to it,
    and nobody else is ever adopted. If nobody in the clip moves like a bowler
    (all-static scene), returns (None, None) so the caller does NOT fall back
    to boxing a random (batsman/keeper) person.

    `frame_provider(frame_idx, bbox) -> np.ndarray` optionally grades the
    stitched candidate with an appearance check (G3); without it the stitch
    stays spatial-only for backwards compatibility.
    """
    if not tracks:
        return None, None
    frame_h, frame_w = frame_dims if frame_dims is not None else _FRAME_DIMS_DEFAULT
    frame_diag = float(np.hypot(frame_w, frame_h))
    ranked = score_bowler_tracks(tracks, frame_dims, total_frames)
    best = ranked[0]
    # Confirmation gate (G-margin): auto-lock only when the winner is strong
    # ENOUGH and clearly ahead. A tight top-1 vs top-2 margin is exactly the
    # reported BOWLER <-> BATSMAN tie -> report NOT CONFIRMED so the UI asks
    # the user instead of silently analysing the wrong person.
    confirmed = True
    confirm_reason = None
    if best["score"] < BOWLER_CONFIRM_MIN_SCORE:
        confirmed, confirm_reason = False, "low_evidence"
    elif len(ranked) >= 2 and best["score"] - ranked[1]["score"] < BOWLER_CONFIRM_MIN_MARGIN:
        confirmed, confirm_reason = False, "ambiguous_margin"
    best["confirmed"] = confirmed
    best["confirm_reason"] = confirm_reason
    best["identity_switch_count"] = count_bowler_identity_switches(
        tracks, best["track_id"], frame_dims, total_frames)
    best["candidates"] = [
        {"track_id": r["track_id"], "score": float(r["score"]),
         "confidence": float(r["confidence"]), "motion": float(r["motion"]),
         "active": float(r["active"]), "n_frames": int(r["n_frames"])}
        for r in ranked]
    # Debug: log top-3 candidates so we can see why the wrong person won
    import logging
    _log = logging.getLogger(__name__)
    if len(ranked) > 1:
        _log.info(
            "Bowler selection top-%d: %s",
            min(3, len(ranked)),
            [(r["track_id"], f"score={r['score']:.3f}",
              f"motion={r['motion']:.3f}", f"active={r['active']:.3f}",
              f"relative={r['relative']:.3f}", f"growth={r['growth']:.3f}",
              f"dir={r['directed']:.3f}", f"delivery={r['delivery']:.3f}",
              f"frames={r['n_frames']}")
             for r in ranked[:3]])
    if best["motion"] < BOWLER_MIN_MOTION_FRACTION and best["active"] < BOWLER_MIN_ACTIVE_FRACTION:
        _log.info("No bowler: best motion=%.3f active=%.3f below thresholds",
                  best["motion"], best["active"])
        return None, None
    if not confirmed:
        _log.info("Bowler NOT confirmed: reason=%s (top1=%.3f top2=%.3f margin=%.3f)",
                  confirm_reason, ranked[0]["score"],
                  ranked[1]["score"] if len(ranked) > 1 else 0.0,
                  (ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else 0.0)
    winner = _stitch_continuations(tracks[best["track_id"]], tracks, frame_diag,
                                   frame_provider=frame_provider)
    best["track_id"] = winner.track_id
    best["n_frames"] = len(winner)
    best["start_frame"] = winner.frames[0] if winner.frames else None
    best["end_frame"] = winner.frames[-1] if winner.frames else None
    _log.info("Selected bowler track #%d (score=%.3f, frames=%d, confirmed=%s)",
              winner.track_id, best["score"], len(winner), confirmed)
    return winner, best


def select_bowler_track(tracks: Dict[int, Track], frame_dims=None,
                        total_frames: Optional[int] = None,
                        frame_provider: Optional[Callable] = None) -> Optional[Track]:
    """Backwards-compatible wrapper: the selected bowler Track or None."""
    winner, _meta = select_bowler_track_with_meta(
        tracks, frame_dims, total_frames, frame_provider=frame_provider)
    return winner


# --------------------------------------------------------------------------- #
# Cricket role classification for non-bowler persons
#
# After the bowler is locked, remaining tracks are scored for three roles:
#   batsman       -- stationary/stable person at the far end (striker's end)
#   wicketkeeper  -- crouching person behind the stumps (bowler's end)
#   umpire        -- upright person standing behind the stumps
#
# Scoring uses four signals computed from the track:
#   motion      -- low motion = batsman; high = not batsman
#   aspect      -- height/width ratio: crouching (< 1.6) = keeper candidate
#   position    -- vertical center relative to frame: far = low Y, near = high Y
#   size        -- relative bbox area: batsman is typically largest non-bowler
# --------------------------------------------------------------------------- #

ROLE_BATSMAN = "batsman"
ROLE_WICKETKEEPER = "wicketkeeper"
ROLE_UMPIRE = "umpire"
ROLE_FIELDER = "fielder"
ROLE_UNKNOWN = "unknown"

# Aspect ratio thresholds (height / width)
_KEEPER_MAX_ASPECT = 1.6    # crouching: short & wide
_UMPIRE_MIN_ASPECT = 2.0    # standing upright: tall & narrow
_BATSMAN_MIN_ASPECT = 1.4   # batting stance: slightly wider than tall

# Motion thresholds (same scale as bowler scoring: tanh of path/diagonal)
_BATSMAN_MAX_MOTION = 0.15  # batsman stays mostly in place
_KEEPER_MAX_MOTION = 0.20   # keeper shuffles slightly

# Position: vertical center of bbox as fraction of frame height (0=top, 1=bottom)
# In behind-the-bowler view: top=far (batsman end), bottom=near (bowler end)
_KEEPER_MIN_Y_FRAC = 0.45   # keeper is in the lower half (near camera)
_BATSMAN_MAX_Y_FRAC = 0.55  # batsman is in the upper half (far from camera)


def _track_aspect_ratio(track: Track) -> float:
    """Mean height/width ratio across the track's bounding boxes."""
    if not track.bboxes:
        return 0.0
    ratios = []
    for x1, y1, x2, y2 in track.bboxes:
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        ratios.append(h / w)
    return float(np.mean(ratios))


def _track_motion_fraction(track: Track, frame_diag: float) -> float:
    """Total displacement / frame diagonal, same metric as bowler scoring."""
    steps = _consecutive_steps(track)
    path = float(np.sum(steps))
    return float(np.tanh(path / max(frame_diag, 1e-6)))


def _track_center_y_fraction(track: Track, frame_h: float) -> float:
    """Mean vertical center of bbox as fraction of frame height (0=top, 1=bottom)."""
    if not track.bboxes or frame_h <= 0:
        return 0.5
    centers = _track_centers(track)
    return float(np.mean(centers[:, 1])) / frame_h


def _track_median_area(track: Track) -> float:
    """Median bounding box area across the track."""
    if not track.bboxes:
        return 0.0
    areas = [_bbox_area(b) for b in track.bboxes]
    return float(np.median(areas))


def _score_role(track: Track, frame_diag: float, frame_h: float,
                frame_w: float, bowler_track_id: int) -> dict:
    """Score a non-bowler track for each cricket role. Returns dict with
    role scores and the best role assignment."""
    motion = _track_motion_fraction(track, frame_diag)
    aspect = _track_aspect_ratio(track)
    y_frac = _track_center_y_fraction(track, frame_h)
    area = _track_median_area(track)
    frame_area = max(1.0, frame_w * frame_h)
    size_frac = area / frame_area

    # --- Batsman score ---
    # Low motion, at far end (low Y), reasonably sized, upright stance
    batsman_motion_score = max(0.0, 1.0 - motion / _BATSMAN_MAX_MOTION)
    batsman_position_score = max(0.0, 1.0 - y_frac / _BATSMAN_MAX_Y_FRAC)
    batsman_aspect_score = 1.0 if aspect >= _BATSMAN_MIN_ASPECT else aspect / _BATSMAN_MIN_ASPECT
    batsman_size_score = min(1.0, size_frac / 0.05)  # penalize very small detections
    batsman = (0.35 * batsman_motion_score +
               0.25 * batsman_position_score +
               0.20 * batsman_aspect_score +
               0.20 * batsman_size_score)

    # --- Wicketkeeper score ---
    # Crouching (low aspect), near camera (high Y), low motion
    keeper_aspect_score = max(0.0, 1.0 - abs(aspect - 1.5) / 1.0)  # peak at ~1.5
    keeper_position_score = max(0.0, (y_frac - _KEEPER_MIN_Y_FRAC) / (1.0 - _KEEPER_MIN_Y_FRAC))
    keeper_motion_score = max(0.0, 1.0 - motion / _KEEPER_MAX_MOTION)
    keeper_size_score = min(1.0, size_frac / 0.04)
    keeper = (0.30 * keeper_aspect_score +
              0.30 * keeper_position_score +
              0.20 * keeper_motion_score +
              0.20 * keeper_size_score)

    # --- Umpire score ---
    # Standing upright (high aspect), mid-position, low motion
    umpire_aspect_score = max(0.0, min(1.0, (aspect - 1.5) / 1.0))  # grows above 1.5
    umpire_position_score = max(0.0, 1.0 - abs(y_frac - 0.55) / 0.20)  # peak mid-frame, narrow band
    umpire_motion_score = max(0.0, 1.0 - motion / 0.10)  # very still
    umpire_size_score = min(1.0, size_frac / 0.03)
    umpire = (0.30 * umpire_aspect_score +
              0.25 * umpire_position_score +
              0.25 * umpire_motion_score +
              0.20 * umpire_size_score)

    scores = {
        ROLE_BATSMAN: float(np.clip(batsman, 0.0, 1.0)),
        ROLE_WICKETKEEPER: float(np.clip(keeper, 0.0, 1.0)),
        ROLE_UMPIRE: float(np.clip(umpire, 0.0, 1.0)),
    }
    best_role = max(scores, key=scores.get)
    best_score = scores[best_role]
    if best_score < 0.20:
        best_role = ROLE_FIELDER
    return {
        "role": best_role,
        "confidence": float(np.clip(best_score, 0.0, 1.0)),
        "scores": scores,
    }


def classify_player_roles(tracks: Dict[int, Track], bowler_track_id: int,
                           frame_dims=None,
                           total_frames: Optional[int] = None) -> dict:
    """Classify all non-bowler tracks into cricket roles.

    Returns a dict mapping track_id to a role dict:
        {track_id: {"role": "batsman"|"wicketkeeper"|"umpire"|"fielder"|"unknown",
                     "confidence": float, "scores": {role: score}}}

    The bowler track is excluded. Only tracks with >= 2 frames are classified
    (single-frame detections are too unreliable for role assignment).
    """
    if not tracks:
        return {}
    frame_h, frame_w = frame_dims if frame_dims is not None else _FRAME_DIMS_DEFAULT
    frame_diag = float(np.hypot(frame_w, frame_h))
    result = {}
    for tid, tr in tracks.items():
        if tid == bowler_track_id:
            continue
        if len(tr) < 3:
            continue
        result[tid] = _score_role(tr, frame_diag, frame_h, frame_w, bowler_track_id)

    # --- Spatial deduplication: merge overlapping tracks of same person ---
    if len(result) > 1:
        tids = list(result.keys())
        to_remove = set()
        for i in range(len(tids)):
            if tids[i] in to_remove:
                continue
            tr_i = tracks.get(tids[i])
            if tr_i is None or not tr_i.bboxes:
                continue
            for j in range(i + 1, len(tids)):
                if tids[j] in to_remove:
                    continue
                tr_j = tracks.get(tids[j])
                if tr_j is None or not tr_j.bboxes:
                    continue
                fi_set = set(tr_i.frames)
                fj_set = set(tr_j.frames)
                common = fi_set & fj_set
                if len(common) < 2:
                    continue
                overlaps = 0
                for fi in common:
                    bi = tr_i.bboxes[tr_i.frames.index(fi)]
                    bj = tr_j.bboxes[tr_j.frames.index(fi)]
                    if _iou(bi, bj) > 0.3:
                        overlaps += 1
                if overlaps / max(len(common), 1) > 0.5:
                    if len(tr_i) >= len(tr_j):
                        to_remove.add(tids[j])
                    else:
                        to_remove.add(tids[i])
                        break
        for tid in to_remove:
            result.pop(tid, None)

    # --- Cap umpire count to at most 1 (highest confidence) ---
    umpire_tids = [tid for tid, r in result.items() if r["role"] == ROLE_UMPIRE]
    if len(umpire_tids) > 1:
        umpire_tids.sort(key=lambda t: result[t]["confidence"], reverse=True)
        for tid in umpire_tids[1:]:
            result[tid]["role"] = ROLE_FIELDER
            result[tid]["scores"][ROLE_UMPIRE] = 0.0

    return result


ROLE_STRIKER = "striker"
ROLE_NON_STRIKER = "non_striker"


def classify_batting_stances(tracks: Dict[int, Track], bowler: Optional[Track],
                             frame_dims=None,
                             total_frames: Optional[int] = None) -> dict:
    """Identify the two batting-end players from view-agnostic geometry.

    Uses the bowler's OWN run axis (start -> end of its center path), which is
    the only cricket geometry available without a pitch/camera model:

      * STRIKER     -- the *static* non-bowler person farthest ALONG the run
                       direction (the crease the bowler runs towards);
      * NON-STRIKER -- the static non-bowler person farthest BACK (bowler's end).

    "Static" means body-normalized speed below the locomotion bar, so a running
    fielder or moving keeper is rejected. Returns
        {track_id: {"role": "striker"|"non_striker", "confidence": float}}
    with 0-2 entries -- a clean BOWLER / STRIKER / NON-STRIKER picture for the
    main pipeline; everything else is ignored.
    """
    result = {}
    if bowler is None or not bowler.bboxes or len(bowler) < 6:
        return result
    frame_h, frame_w = frame_dims if frame_dims is not None else _FRAME_DIMS_DEFAULT
    centers = _track_centers(bowler)
    if len(centers) < 6:
        return result
    start = centers[0]
    run_vec = centers[-1] - start
    axis_len = float(np.hypot(*run_vec))
    if axis_len < 1e-6:
        return result
    axis = run_vec / axis_len

    cands = []
    for tid, tr in tracks.items():
        if tid == bowler.track_id or len(tr) < 3:
            continue
        if _mean_body_speed(tr) > BOWLER_ACTIVE_BODY_MIN:
            continue  # actively moving -> not a stationary batting-end player
        mid = len(tr.bboxes) // 2
        c = np.asarray(_bbox_center(tr.bboxes[mid]), dtype=float)
        proj = float(np.dot(c - start, axis))
        cands.append({"track_id": tid, "proj": proj})
    if not cands:
        return result
    cands.sort(key=lambda d: d["proj"])
    sep = float(cands[-1]["proj"] - cands[0]["proj"]) if len(cands) > 1 else 0.0
    min_sep = BOWLER_STANCE_MIN_SEP_FRAC * frame_w
    conf = float(np.clip(sep / max(min_sep, 1e-6), 0.0, 1.0)) if len(cands) > 1 else 0.5

    striker = cands[-1]
    result[striker["track_id"]] = {"role": ROLE_STRIKER, "confidence": conf}
    if len(cands) >= 2:
        result[cands[0]["track_id"]] = {"role": ROLE_NON_STRIKER, "confidence": conf}
    return result


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
