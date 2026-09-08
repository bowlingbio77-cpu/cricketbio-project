"""
Cricket-video validity checks.

The full pipeline (``pipeline.analyze_video``) runs unconditionally on whatever
video it is given.  For a non-cricket clip (a person walking, a football match,
a random object) it will happily crop "a bowler", run pose estimation and then
output biomechanical scores that have no physical meaning.

These functions provide a set of *independent signals* that a clip actually
shows a cricket bowling action, so callers can warn the user (or refuse to
analyse) rather than silently emitting nonsense numbers:

    1. Human (bowler) pose present        -- MediaPipe found a plausible body.
    2. Ball present / plausible flight    -- the ball tracker recovered a
                                             ballistic trajectory.
    3. Bowling-arm action                 -- the bowling shoulder/wrist reaches
                                             an overhead delivery position and
                                             rotates about the shoulder.

Each check returns a small dataclass (``ok`` + ``score`` + ``reason``).  The
top-level ``assess_cricket_video`` combines them into a single verdict with an
overall confidence.  Nothing here is wired into the pipeline yet -- see the
module docstring of ``pipeline.py`` and the ``assess_cricket_video`` docstring
for how it is intended to be used.
"""
from typing import List, Optional

import numpy as np

from . import config
from .pose_estimation import PoseFrame


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

class CheckResult:
    """Outcome of one validity signal."""

    __slots__ = ("name", "ok", "score", "reason")

    def __init__(self, name: str, ok: bool, score: float, reason: str):
        self.name = name
        self.ok = ok
        self.score = float(score)   # 0.0 .. 1.0
        self.reason = reason

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "score": self.score,
                "reason": self.reason}


class ValidityVerdict:
    """Combined judgement from all checks."""

    __slots__ = ("is_cricket", "confidence", "checks", "summary")

    def __init__(self, is_cricket: bool, confidence: float,
                 checks: List[CheckResult], summary: str):
        self.is_cricket = bool(is_cricket)
        self.confidence = float(confidence)     # 0.0 .. 1.0
        self.checks = checks
        self.summary = summary

    def to_dict(self) -> dict:
        return {
            "is_cricket": self.is_cricket,
            "confidence": self.confidence,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


# --------------------------------------------------------------------------- #
# Landmark helpers
# --------------------------------------------------------------------------- #

def _lm(frame: PoseFrame, names: List[str], use_world: bool = False) -> Optional[np.ndarray]:
    """Return (n, dim) landmark coordinates for the given names.

    For ``use_world=False`` returns normalized (0-1, y-down) image coords from
    ``frame.landmarks``; for ``use_world=True`` returns metric world coords
    (y-up) from ``frame.world_landmarks`` (if present).
    """
    if frame is None:
        return None
    if use_world:
        arr = frame.world_landmarks
        if arr is None:
            return None
    else:
        arr = frame.landmarks
    try:
        idx = [config.POSE_LANDMARK_NAMES.index(n) for n in names]
    except ValueError:
        return None
    if max(idx) >= arr.shape[0]:
        return None
    return np.array([arr[i][:3] for i in idx], dtype=float)


def _vis(frame: PoseFrame, name: str) -> float:
    """Landmark visibility (0-1) or 0.0 if the landmark is unavailable."""
    try:
        i = config.POSE_LANDMARK_NAMES.index(name)
    except ValueError:
        return 0.0
    if i >= frame.landmarks.shape[0]:
        return 0.0
    return float(frame.landmarks[i, 3])


# --------------------------------------------------------------------------- #
# 1. Human pose present
# --------------------------------------------------------------------------- #

def check_human_pose(pose_sequence: List[PoseFrame],
                     min_frames: int = 3,
                     min_visibility: float = 0.4) -> CheckResult:
    """A valid bowling frame must contain a human with key joints visible.

    Requires shoulders and hips (the torso) plus an arm/wrist on at least a few
    frames -- a bare torso-only detection (e.g. a mannequin) is insufficient.
    """
    torso_ok = 0
    arms_ok = 0
    posture_ok = 0
    for pf in pose_sequence:
        ls = _vis(pf, "left_shoulder")
        rs = _vis(pf, "right_shoulder")
        lh = _vis(pf, "left_hip")
        rh = _vis(pf, "right_hip")
        if min(ls, rs, lh, rh) >= min_visibility:
            torso_ok += 1
        if min(_vis(pf, "left_wrist"), _vis(pf, "right_wrist"),
               _vis(pf, "left_elbow"), _vis(pf, "right_elbow")) >= min_visibility:
            arms_ok += 1
        # Torso roughly upright: shoulders above hips in image coords.
        sh = _lm(pf, ["left_shoulder", "right_shoulder"])
        hp = _lm(pf, ["left_hip", "right_hip"])
        if sh is not None and hp is not None:
            shoulder_y = float(np.mean(sh[:, 1]))
            hip_y = float(np.mean(hp[:, 1]))
            if hip_y > shoulder_y + 0.05:   # hips below shoulders (y-down)
                posture_ok += 1

    n = len(pose_sequence)
    if n == 0:
        return CheckResult("human_pose", False, 0.0,
                           "No pose detected at all.")
    torso_frac = torso_ok / n
    arm_frac = arms_ok / n
    posture_frac = posture_ok / n

    score = 3 * min(torso_frac, 1.0) + 2 * min(arm_frac, 1.0) + 1 * min(posture_frac, 1.0)
    score = min(score / 6.0, 1.0)

    # A clear, upright human torso is enough to treat the clip as containing a
    # person. Arms are a *soft* signal: a small/far-away bowler (or an
    # occlusion) often yields undetectable wrist/elbow landmarks even though a
    # person is plainly present, so arms must NOT be a hard requirement here.
    # We only fall back to requiring arms when the posture is too ambiguous
    # (neither clearly upright nor clearly prone) to call it a bowler on its own.
    #
    # Cricket bowlers frequently lean forward during delivery, so the posture
    # threshold is intentionally lenient (0.35 not 0.5) — a bowler at 40%
    # "upright" is still clearly a bowler, not a bystander.
    posture_good = posture_frac >= 0.35
    arms_good = arm_frac >= 0.2
    ok = (torso_ok >= min_frames and
          (posture_good or arms_good))
    reason = (f"Human pose detected on {torso_ok}/{n} frames "
              f"(torso {torso_frac:.0%}, arms {arm_frac:.0%}).") if ok else \
             (f"Only {torso_ok}/{n} frames with a clear human torso "
              f"(arms {arm_frac:.0%}, upright {posture_frac:.0%}); not a "
              f"plausible bowler clip.")
    return CheckResult("human_pose", ok, score, reason)


# --------------------------------------------------------------------------- #
# 2. Ball present / plausible flight
# --------------------------------------------------------------------------- #

def check_ball_present(track, track_stats: Optional[dict],
                       min_frames: int = 3,
                       min_detected: int = 0) -> CheckResult:
    """The ball tracker should have recovered a real, moving ball trajectory.

    A non-cricket clip usually yields no track (``track_stats["outcome"]`` in
    {"no_yolo_model", "track_too_short"}) or a track that fails the motion
    validity rules (a stationary round object).
    """
    if not track:
        outcome = (track_stats or {}).get("outcome", "no_track")
        return CheckResult("ball_present", False, 0.0,
                           f"No ball trajectory recovered ({outcome}).")
    n = len(track)
    detected = sum(1 for p in track if p.detected)

    score = min(0.3 + 0.5 * (detected / max(n, 1)) + 0.2 * min(n / 30.0, 1.0), 1.0)

    ok = n >= min_frames and detected >= min_detected
    reason = (f"Ball trajectory recovered ({n} frames, {detected} detected).") if ok else \
             (f"Ball track too weak ({n} frames, {detected} detected) to be a "
              f"delivery in flight.")
    return CheckResult("ball_present", ok, score, reason)


# --------------------------------------------------------------------------- #
# 3. Bowling-arm action
# --------------------------------------------------------------------------- #

def _arm_raised_fraction(pose_sequence: List[PoseFrame], arm: str,
                         min_visibility: float = 0.4) -> float:
    """Fraction of frames where the bowling wrist is at/above shoulder level.

    In image coords (y-down) the raised bowling arm shows wrist_y <= shoulder_y
    once the shoulder is approximated by the top of the shoulder line.  We use
    the bowling-side shoulder and wrist landmarks directly.
    """
    if not pose_sequence:
        return 0.0
    raised = 0
    seen = 0
    for pf in pose_sequence:
        wrist = _lm(pf, [f"{arm}_wrist"])
        sh = _lm(pf, [f"{arm}_shoulder"])
        if wrist is None or sh is None:
            continue
        if min(_vis(pf, f"{arm}_wrist"), _vis(pf, f"{arm}_shoulder")) < min_visibility:
            continue
        seen += 1
        # y-down: wrist above shoulder => smaller y.  Tolerate ~0 (at shoulder).
        if wrist[0, 1] <= sh[0, 1] + 0.03:
            raised += 1
    return (raised / seen) if seen else 0.0


def check_bowling_motion(pose_sequence: List[PoseFrame],
                         bowling_arm: str = "right",
                         min_raised_frac: float = 0.15) -> CheckResult:
    """Detect a bowling-arm delivery action.

    The hallmark of a bowling (as opposed to a throw/throwing-a-ball) action is
    the arm being swept up over the shoulder before the ball is released.  We
    check for a *sustained overhead* configuration of the bowling arm plus a
    change in the arm pitch over the clip (the windup -> release sweep).
    """
    if not pose_sequence:
        return CheckResult("bowling_motion", False, 0.0, "No pose data.")

    raised = _arm_raised_fraction(pose_sequence, bowling_arm)

    # Does the bowling arm sweep -- i.e. does its elevation vary across frames?
    elevs = []
    for pf in pose_sequence:
        wrist = _lm(pf, [f"{bowling_arm}_wrist"])
        sh = _lm(pf, [f"{bowling_arm}_shoulder"])
        if wrist is None or sh is None:
            continue
        if _vis(pf, f"{bowling_arm}_wrist") < 0.3:
            continue
        elevs.append(float(wrist[0, 1] - sh[0, 1]))  # negative when raised
    sweep = 0.0
    if len(elevs) >= 4:
        lo, hi = float(min(elevs)), float(max(elevs))
        sweep = float(min(hi - lo, 0.5)) / 0.5      # arm travels a meaningful arc

    score = min(0.6 * raised + 0.4 * min(sweep, 1.0), 1.0)
    ok = raised >= min_raised_frac
    reason = (f"Bowling-arm action detected (arm raised {raised:.0%} of frames).") if ok else \
             (f"Bowling arm only raised {raised:.0%} of frames; no overhead "
              f"delivery action.")
    return CheckResult("bowling_motion", ok, score, reason)


# --------------------------------------------------------------------------- #
# Combined assessment
# --------------------------------------------------------------------------- #

def assess_cricket_video(pose_sequence: List[PoseFrame],
                         track=None,
                         track_stats: Optional[dict] = None,
                         bowling_arm: str = "right",
                         weights: Optional[dict] = None) -> ValidityVerdict:
    """Combine the individual checks into a single judgement.

    Parameters
    ----------
    pose_sequence : list[PoseFrame]
        The pose frames produced by ``pose_estimation``.
    track : list[BallPoint] or None
        The ball trajectory from ``ball_tracking.track_ball`` (may be empty).
    track_stats : dict or None
        The stats dict returned alongside ``track``.
    bowling_arm : str
        "right" or "left".
    weights : dict or None
        Optional per-check weight override, e.g.
        ``{"human_pose": 0.4, "ball_present": 0.3, "bowling_motion": 0.3}``.
        Defaults to all checks weighted equally.

    Returns
    -------
    ValidityVerdict
        ``is_cricket`` True iff the (weighted) average of the passing checks is
        >= ``MIN_CONFIDENCE``.  ``confidence`` is that weighted average.
    """
    checks = [
        check_human_pose(pose_sequence),
        check_ball_present(track, track_stats),
        check_bowling_motion(pose_sequence, bowling_arm),
    ]
    default_w = {"human_pose": 1.0, "ball_present": 1.0, "bowling_motion": 1.0}
    if weights:
        default_w.update(weights)

    total = sum(default_w.get(c.name, 1.0) for c in checks)
    confidence = sum(c.score * default_w.get(c.name, 1.0) for c in checks) / total

    passed = [c for c in checks if c.ok]
    is_cricket = confidence >= MIN_CONFIDENCE

    if all(c.ok for c in checks):
        summary = "Clearly a cricket bowling clip."
    elif is_cricket:
        summary = (f"Looks like a bowling clip (confidence {confidence:.0%}) "
                   f"but {len(checks) - len(passed)} check{'' if len(checks)-len(passed)==1 else 's'} "
                   f"weak.")
    else:
        summary = (f"Not confidently a cricket bowling clip "
                   f"(confidence {confidence:.0%}).")
    return ValidityVerdict(is_cricket, confidence, checks, summary)


# Confidence (0-1) at/above which the clip is judged "cricket".
MIN_CONFIDENCE = 0.5
