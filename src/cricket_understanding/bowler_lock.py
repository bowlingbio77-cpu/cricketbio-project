"""
Phase 12 -- BOWLER LOCK.

```
person detection -> tracking -> candidate generation
                -> cricket role evidence -> bowling-action evidence
                -> combined score -> bowler confirmation -> BOWLER LOCK
                -> pose extraction -> biomechanics
```

Rules enforced (from the brief)
--------------------------------
1. **Never silently switch** from the confirmed bowler to another person.  Once
   locked, only continuations that pass the identity gate may be re-stitched;
   if that fails the lock is *held* and the result is flagged, not swapped.
2. **Temporarily lost tracks require recovery.**  ``recover_lock`` re-attaches a
   later track fragment to the lock, but only if the appearance/spatial gate
   passes.  Failure is reported, never silently accepted.
3. **Pose must use the locked bowler track.**  :meth:`BowlerLock.pose_crop`
   refuses any frame whose track id is not the lock.
4. **Biomechanics must use the locked bowler track.**  :meth:`BowlerLock.
   biomechanics_window` refuses a non-locked track.
5. **Replay overlays must use the same identity.**
   :meth:`BowlerLock.replay_boxes` is the only sanctioned source of boxes for
   the overlay.
6. **If confirmation fails: BOWLER NOT CONFIRMED** and normal bowler
   biomechanics are withheld (:func:`not_confirmed_result`).

The lock is *inert by default*: with no validated role/action model it falls
back to the existing heuristic selection and says so, rather than pretending a
learned model ran.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import confidence as conf_mod

NOT_CONFIRMED = conf_mod.NOT_CONFIRMED_MESSAGE

#: Identity-gate thresholds. These bound *admission of a candidate*, they are
#: not a model-quality knob, and they are reported with every decision.
MAX_RECOVERY_FRAME_GAP = 30
MIN_RECOVERY_CENTROID_PX = 220.0
MIN_RECOVERY_IOU = 0.0
MIN_RECOVERY_HIST_CORR = 0.10
CROP_PAD_FRAC = 0.30


@dataclass
class LockCandidate:
    track_id: int
    evidence: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    role_confidence: Optional[float] = None
    action_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id, "combined_score": self.combined_score,
            "role_confidence": self.role_confidence,
            "action_confidence": self.action_confidence, "evidence": dict(self.evidence),
        }


@dataclass
class BowlerLock:
    track_id: Optional[int] = None
    confirmed: bool = False
    reason: str = ""
    candidates: List[LockCandidate] = field(default_factory=list)
    boxes: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    source: str = "none"                 # "role_model" | "action_model" | "heuristic" | "user"
    recovered_segments: List[Dict[str, Any]] = field(default_factory=list)
    identity_switch_attempts: int = 0
    warnings: List[str] = field(default_factory=list)
    confidence: Optional[conf_mod.ConfidenceBundle] = None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_existing_selection(cls, track, meta: Optional[Dict[str, Any]] = None,
                                source: str = "heuristic") -> "BowlerLock":
        """Adopt the EXISTING pipeline selection without changing its verdict.

        This is the compatibility path: with no validated cricket model the
        lock mirrors ``src.tracking.select_bowler_track_with_meta`` exactly,
        including its confirmation gate and its refusal to invent a bowler.
        """
        meta = meta or {}
        bboxes = _track_boxes(track)
        return cls(
            track_id=track.track_id if track is not None else None,
            confirmed=bool(meta.get("confirmed", False)) if track is not None else False,
            reason=meta.get("confirm_reason") or ("" if track is not None else "no_bowler_track"),
            source=source,
            boxes=bboxes,
            candidates=[
                LockCandidate(
                    track_id=int(c["track_id"]),
                    combined_score=float(c.get("score", 0.0)),
                    role_confidence=None,
                    action_confidence=None,
                    evidence={"motion": float(c.get("motion", 0.0)),
                              "active": float(c.get("active", 0.0)),
                              "n_frames": float(c.get("n_frames", 0))},
                )
                for c in (meta.get("candidates") or [])
            ],
            identity_switch_attempts=int(meta.get("identity_switch_count", 0) or 0),
        )

    @classmethod
    def from_role_and_action(cls, candidates: Sequence[LockCandidate],
                             confirm_min: float = 0.35,
                             confirm_min_margin: float = 0.06) -> "BowlerLock":
        """Build the lock from LEARNED role + action evidence."""
        ranked = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
        lock = cls(candidates=ranked, source="role_model+action_model")
        if not ranked:
            lock.reason = "no_candidates"
            return lock
        best = ranked[0]
        lock.track_id = best.track_id
        if best.combined_score < confirm_min:
            lock.confirmed, lock.reason = False, "low_evidence"
        elif len(ranked) >= 2 and best.combined_score - ranked[1].combined_score < confirm_min_margin:
            lock.confirmed, lock.reason = False, "ambiguous_margin"
        else:
            lock.confirmed, lock.reason = True, "confirmed"
        return lock

    @classmethod
    def from_user(cls, track_id: int, boxes: Optional[Dict[int, Sequence[float]]] = None,
                  reason: str = "user confirmation") -> "BowlerLock":
        return cls(track_id=track_id, confirmed=True, reason=reason, source="user",
                   boxes={int(k): tuple(float(v) for v in b)  # type: ignore[misc]
                          for k, b in (boxes or {}).items()})

    # -- rules 1 & 2 : identity stability and recovery ---------------------- #

    def is_locked(self) -> bool:
        return self.track_id is not None

    def is_locked_track(self, track_id: Optional[int]) -> bool:
        return self.track_id is not None and track_id == self.track_id

    def recover_lock(self, fragment_track_id: int,
                     fragment_boxes: Dict[int, Sequence[float]],
                     appearance_check: Optional[Callable[[], float]] = None) -> bool:
        """Re-attach a later track fragment to the locked identity.

        Returns ``True`` only when EVERY gate passes:

        * the fragment is not a *different* currently-confirmed lock,
        * the frame gap from the last locked observation is bounded,
        * the extrapolated centroid is within :data:`MIN_RECOVERY_CENTROID_PX`,
        * the appearance correlation is at least :data:`MIN_RECOVERY_HIST_CORR`
          (skipped with an explicit warning when no checker is supplied --
          never silently treated as a pass).

        A failure leaves the lock untouched and records why.
        """
        self.identity_switch_attempts += 1
        if self.track_id is None:
            self.warnings.append("recovery refused: no lock to recover")
            return False
        if fragment_track_id == self.track_id:
            for f, b in fragment_boxes.items():
                self.boxes[int(f)] = tuple(float(v) for v in b)  # type: ignore[assignment]
            return True
        if self.confirmed and self.source in ("role_model", "role_model+action_model", "user"):
            self.warnings.append(
                f"recovery refused: lock #{self.track_id} is confirmed from "
                f"{self.source}; switching to #{fragment_track_id} is never silent")
            return False
        if not fragment_boxes:
            self.warnings.append("recovery refused: empty fragment")
            return False

        frames = sorted(fragment_boxes)
        locked_frames = sorted(self.boxes)
        if not locked_frames:
            self.warnings.append("recovery refused: lock has no boxes to anchor from")
            return False
        gap = frames[0] - locked_frames[-1]
        if gap <= 0:
            self.warnings.append(
                f"recovery refused: fragment starts at f{frames[0]}, before the "
                f"lock's last observation f{locked_frames[-1]}")
            return False
        if gap > MAX_RECOVERY_FRAME_GAP:
            self.warnings.append(
                f"recovery refused: {gap}-frame gap exceeds "
                f"MAX_RECOVERY_FRAME_GAP={MAX_RECOVERY_FRAME_GAP}")
            return False

        b1, b2 = self.boxes[locked_frames[-1]], tuple(
            float(v) for v in fragment_boxes[frames[0]])  # type: ignore[assignment]
        pred = _extrapolate_centroid(self.boxes, locked_frames[-1], gap)
        dist = math.hypot((b2[0] + b2[2]) / 2 - pred[0], (b2[1] + b2[3]) / 2 - pred[1])
        if dist > MIN_RECOVERY_CENTROID_PX:
            self.warnings.append(
                f"recovery refused: fragment centroid {dist:.0f}px from the "
                f"extrapolated position (limit {MIN_RECOVERY_CENTROID_PX:.0f}px)")
            return False

        if appearance_check is None:
            self.warnings.append(
                "recovery ACCEPTED SPATIALLY ONLY: no appearance checker supplied, "
                "so identity is not visually confirmed. Flagged in the output.")
        else:
            corr = float(appearance_check())
            if corr < MIN_RECOVERY_HIST_CORR:
                self.warnings.append(
                    f"recovery refused: appearance correlation {corr:.3f} below "
                    f"MIN_RECOVERY_HIST_CORR={MIN_RECOVERY_HIST_CORR}")
                return False

        for f, b in fragment_boxes.items():
            self.boxes[int(f)] = tuple(float(v) for v in b)  # type: ignore[assignment]
        self.recovered_segments.append({
            "from_track_id": fragment_track_id, "into_track_id": self.track_id,
            "start_frame": frames[0], "end_frame": frames[-1],
            "frame_gap": gap, "centroid_error_px": round(dist, 2),
            "appearance_checked": appearance_check is not None,
        })
        return True

    # -- rules 3, 4, 5 : downstream consumers ------------------------------- #

    def pose_crop(self, frame_id: int, track_id: Optional[int],
                  frame_shape) -> Optional[Tuple[int, int, int, int]]:
        """Crop window for pose. Raises unless ``track_id`` is the lock.

        This is the guard that stops the pose stage from drifting onto the
        batsman, keeper or a nearby fielder.
        """
        if not self.is_locked():
            raise BowlerLockError(
                "BOWLER NOT CONFIRMED: cannot crop a pose window without a bowler "
                f"lock ({self.reason or 'no lock'}).")
        if not self.is_locked_track(track_id):
            raise BowlerLockError(
                f"identity violation: pose requested for track {track_id}, but the "
                f"lock is track {self.track_id}. Pose may only use the locked bowler.")
        b = self.boxes.get(int(frame_id))
        if b is None:
            return None
        h, w = frame_shape[:2]
        pw, ph = CROP_PAD_FRAC * (b[2] - b[0]), CROP_PAD_FRAC * (b[3] - b[1])
        x1, x2 = max(0, int(b[0] - pw)), min(w, int(b[2] + pw))
        y1, y2 = max(0, int(b[1] - ph)), min(h, int(b[3] + ph))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return (x1, y1, x2, y2)

    def biomechanics_window(self, start_frame: int, end_frame: int,
                            track_id: Optional[int]) -> Tuple[int, int]:
        """Validate that a biomechanics window belongs to the locked bowler."""
        if not self.confirmed:
            raise BowlerLockError(
                f"{NOT_CONFIRMED} ({self.reason or 'unconfirmed'}): no bowler "
                "biomechanics may be produced.")
        if not self.is_locked_track(track_id):
            raise BowlerLockError(
                f"identity violation: biomechanics requested for track {track_id}, "
                f"but the lock is track {self.track_id}.")
        if end_frame < start_frame:
            raise BowlerLockError(
                f"invalid window [{start_frame}, {end_frame}]")
        return (int(start_frame), int(end_frame))

    def replay_boxes(self, frame_id: int) -> Optional[Tuple[float, float, float, float]]:
        """The ONLY sanctioned box source for replay overlays.

        Returns the locked bowler's box, or ``None`` -- it never returns another
        person's box, so an overlay cannot silently switch identity.
        """
        if self.track_id is None:
            return None
        return self.boxes.get(int(frame_id))

    def replay_track_id(self) -> Optional[int]:
        return self.track_id

    # -- rule 6 : refusal --------------------------------------------------- #

    def gate(self) -> Dict[str, Any]:
        """The object the pipeline must branch on before producing results."""
        if not self.is_locked() or not self.confirmed:
            res = conf_mod.not_confirmed_result(self.reason or "unconfirmed")
            res["lock"] = self.to_dict()
            return res
        return {
            "status": "BOWLER LOCKED",
            "track_id": self.track_id,
            "biomechanics_available": True,
            "lock": self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "source": self.source,
            "n_boxed_frames": len(self.boxes),
            "box_frame_range": ([min(self.boxes), max(self.boxes)]
                                if self.boxes else None),
            "candidates": [c.to_dict() for c in self.candidates],
            "recovered_segments": self.recovered_segments,
            "identity_switch_attempts": self.identity_switch_attempts,
            "warnings": self.warnings,
            "confidence": self.confidence.to_dict() if self.confidence else None,
        }


class BowlerLockError(RuntimeError):
    """Raised on an identity violation or a not-confirmed lock."""


def _extrapolate_centroid(boxes: Dict[int, Tuple[float, float, float, float]],
                          last_frame: int, gap: int) -> Tuple[float, float]:
    """Constant-velocity centroid extrapolation from the last two observations."""
    frames = sorted(boxes)
    if len(frames) < 2:
        b = boxes[last_frame]
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    f1, f2 = frames[-2], frames[-1]
    b1, b2 = boxes[f1], boxes[f2]
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    dt = max(1, f2 - f1)
    vx, vy = (c2[0] - c1[0]) / dt, (c2[1] - c1[1]) / dt
    return (c2[0] + vx * gap, c2[1] + vy * gap)


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #

#: Set to a model path to activate learned role/action evidence. Left ``None``,
#: the lock mirrors the existing heuristic and SAYS SO.
ROLE_MODEL_PATH: Optional[str] = None
ACTION_MODEL_PATH: Optional[str] = None


def _track_boxes(track: Any) -> Dict[int, Tuple[float, float, float, float]]:
    """``Track`` -> ``{frame: bbox}``, defensively (duck-typed, no torch import)."""
    if track is None:
        return {}
    frames = list(getattr(track, "frames", []) or [])
    boxes = list(getattr(track, "bboxes", []) or [])
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for f, b in zip(frames, boxes):
        try:
            out[int(f)] = tuple(float(v) for v in b)  # type: ignore[assignment]
        except (TypeError, ValueError):
            continue
    return out


def build_lock_from_pipeline(tracks: Dict[int, Any], bowler, bowler_meta: Optional[Dict],
                             user_track_id: Optional[int] = None) -> BowlerLock:
    """Create the lock for one clip.

    Priority (highest first):
      1. explicit user confirmation (a human decided -- always wins);
      2. a learned role+action model, if one is configured and loads;
      3. the existing heuristic selection, unchanged.
    """
    if user_track_id is not None and user_track_id in tracks:
        return BowlerLock.from_user(user_track_id, boxes=_track_boxes(tracks[user_track_id]))

    learned = _try_learned_candidates(tracks)
    if learned:
        lock = BowlerLock.from_role_and_action(learned)
        for c in lock.candidates:
            lock.boxes.update(_track_boxes(tracks.get(c.track_id)))
        return lock

    lock = BowlerLock.from_existing_selection(bowler, bowler_meta, source="heuristic")
    lock.warnings.append(
        "No validated cricket role/action model is configured "
        "(cricket_understanding.bowler_lock.ROLE_MODEL_PATH is None). The lock "
        "mirrors the existing heuristic bowler selection. It is NOT a learned "
        "cricket-role decision and must not be described as one.")
    return lock


def _try_learned_candidates(tracks: Dict[int, Any]) -> List[LockCandidate]:
    """Load the learned role/action models if configured. Returns [] otherwise.

    Kept lazy and defensive: a missing or broken artifact degrades to the
    heuristic path with a warning rather than crashing the analysis.
    """
    if not (ROLE_MODEL_PATH and ACTION_MODEL_PATH):
        return []
    try:
        from . import role_model  # noqa: F401  (import guarded)
        from . import action_model  # noqa: F401
    except Exception:
        return []
    return []  # not wired yet; the heuristic path is used and is labelled as such
