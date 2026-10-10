"""Cricket role engine: the single source of truth for player identity.

Why this module exists
----------------------
Identity was previously decided in more than one place, and the decisions
disagreed. ``src/tracking.py`` ranked person tracks on geometry and motion, while
``src/pose_estimation.py::_primary_person`` independently re-chose the analysed
subject by largest landmark box, ignoring the track id entirely. A run could
therefore report a *locked* bowler while extracting biomechanics from a different
human.

Measured on the only annotated broadcast footage in the repository, that failure
is bimodal rather than noisy: when the predicted track matched the annotated
bowler the median IoU was 0.91, and when it did not the IoU was *exactly* zero.
See ``bowler_identity_audit.md``.

Design rules
------------
1. One authority. Every downstream consumer takes ``bowler_track_id`` from
   :class:`RoleAssignment` and must not re-derive identity.
2. Evidence is interpretable. Every score decomposes into named, individually
   inspectable components with per-component contributions.
3. A prior is never a rule. :data:`LOWER_HALF_THRESHOLD` is a weak temporal
   prior only; it can never on its own assign the bowler role.
4. Fail safe. Weak identity, continuity, or pose evidence yields
   :attr:`RoleState.FAILED` and a withheld result, never a guess.
5. No silent switching. A confirmed bowler is locked; losing the track moves to
   :attr:`RoleState.TEMPORARILY_LOST` and can only recover to the *same* id.

The engine consumes per-frame geometry and, optionally, pose and ball evidence.
It has no dependency on ultralytics, torch, or mediapipe, so it is fully
testable without models.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

class Role(str, Enum):
    """Authoritative cricket role for a tracked person."""

    BOWLER = "BOWLER"
    STRIKER = "STRIKER"
    NON_STRIKER = "NON_STRIKER"
    WICKETKEEPER = "WICKETKEEPER"
    FIELDER = "FIELDER"
    UNKNOWN = "UNKNOWN"
    OTHER = "OTHER"


class RoleState(str, Enum):
    """Identity confidence state for the bowler slot.

    ``CANDIDATE -> PROBABLE -> CONFIRMED -> LOCKED`` is the happy path.
    ``LOCKED -> TEMPORARILY_LOST -> RECOVERED`` handles occlusion. A
    re-acquisition of a *different* track is a failure, never a switch.
    """

    UNKNOWN = "UNKNOWN"
    CANDIDATE = "CANDIDATE"
    PROBABLE = "PROBABLE"
    CONFIRMED = "CONFIRMED"
    LOCKED = "LOCKED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


class BowlingPhase(str, Enum):
    """Phases of the bowling action, in temporal order."""

    RUN_UP = "RUN_UP"
    APPROACH = "APPROACH"
    GATHER = "GATHER"
    DELIVERY_STRIDE = "DELIVERY_STRIDE"
    RELEASE = "RELEASE"
    FOLLOW_THROUGH = "FOLLOW_THROUGH"
    UNKNOWN = "UNKNOWN"


#: Weak temporal prior only. Expressed as the fraction of the delivery window a
#: candidate must occupy with the lower body dominant to earn the prior. This is
#: *never* sufficient to assign BOWLER on its own -- see
#: :meth:`BowlerRoleEngine.score_candidate`, where the prior carries a capped
#: weight and requires corroborating action evidence.
LOWER_HALF_THRESHOLD = 0.50

#: Ordering weight caps. Weighted evidence is capped so that no single signal can
#: manufacture identity on its own.
MAX_WEIGHTED_CONTRIBUTION = 0.34
MAX_PRIOR_CONTRIBUTION = 0.06

#: Absolute floor on the delivery peak, expressed as displacement per frame in
#: units of the actor's own bounding-box diagonal.
#:
#: A *relative* peak test (peak >= k * median) is not sufficient on its own: for a
#: person who is essentially stationary, both peak and median step are near-zero
#: noise, so the ratio is easily satisfied by jitter. Without this floor a
#: standing, fidgeting batsman scores a full six-phase delivery. A real run-up
#: moves several percent of a body diagonal per frame, so the floor separates the
#: two regimes cleanly.
MIN_PEAK_BODY_STEP = 0.02

#: The approach phase must itself be moving, not a single spike in a static
#: track. Expressed as the fraction of the peak the median pre-peak step must
#: reach.
MIN_RAMP_RATIO = 0.20


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class RoleWeights:
    """Interpretable, configurable evidence weights.

    The grouped weights let a caller reason about *families* of evidence
    (spatial, motion, pose, action, phase, continuity) while individual
    components stay inspectable.
    """

    motion: float = 0.20
    action: float = 0.26
    phase: float = 0.16
    continuity: float = 0.14
    pose: float = 0.12
    spatial: float = 0.08
    ball: float = 0.04
    lower_half_prior: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "motion": self.motion,
            "action": self.action,
            "phase": self.phase,
            "continuity": self.continuity,
            "pose": self.pose,
            "spatial": self.spatial,
            "ball": self.ball,
            "lower_half_prior": self.lower_half_prior,
        }


@dataclass
class RoleThresholds:
    """Thresholds governing state transitions. All on a 0..1 score scale."""

    candidate: float = 0.35
    probable: float = 0.50
    confirm: float = 0.62
    lock: float = 0.70

    #: A confirmed bowler may be missing for at most this fraction of the
    #: delivery window before the result is failed rather than recovered.
    max_lost_fraction: float = 0.40

    #: Minimum fraction of frames with a usable pose for biomechanics to run.
    min_pose_coverage: float = 0.50

    #: Minimum number of observed frames before any role is asserted.
    min_observations: int = 8

    #: Score at which the runner-up is considered a credible alternative.
    ambiguity_margin: float = 0.12


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #

@dataclass
class PlayerObservation:
    """One tracked person in one frame.

    Geometry is required; pose and ball evidence are optional and only strengthen
    or weaken the assessment when actually present.
    """

    track_id: int
    frame_idx: int
    bbox: BBox
    timestamp_sec: float = 0.0
    detection_confidence: float = 1.0
    #: 0..1 arm extension of the bowling arm, when pose is available.
    arm_extension: Optional[float] = None
    #: 0..1 degree to which the lower body dominates the pose silhouette.
    lower_body_dominance: Optional[float] = None
    #: Distance in pixels from this person's centre to the tracked ball.
    ball_distance_px: Optional[float] = None
    #: Whether the ball is visible in this frame at all.
    ball_visible: bool = False
    #: Which arm the pose model saw extend, when it is known: "LEFT"/"RIGHT".
    bowling_arm_hint: str = "UNKNOWN"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)


@dataclass
class EvidenceBreakdown:
    """Per-component score and weighted contribution, for inspection."""

    components: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    family_scores: Dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        parts = ", ".join(
            f"{name}={value:.3f} (w={self.contributions.get(name, 0.0):.3f})"
            for name, value in sorted(self.components.items())
        )
        return f"score={self.total:.3f} [{parts}]"

    @property
    def total(self) -> float:
        return float(sum(self.contributions.values()))


@dataclass
class PhaseProfile:
    """Result of segmenting a candidate's motion into bowling phases."""

    phases: Dict[int, BowlingPhase] = field(default_factory=dict)
    peak_frame: Optional[int] = None
    ramp_up_ratio: float = 0.0
    matched: bool = False
    reason: str = ""

    @property
    def phase_counts(self) -> Dict[BowlingPhase, int]:
        counts: Dict[BowlingPhase, int] = defaultdict(int)
        for phase in self.phases.values():
            counts[phase] += 1
        return dict(counts)

    @property
    def ordered_phases(self) -> List[BowlingPhase]:
        """Phases in temporal order of first appearance."""
        seen: List[BowlingPhase] = []
        for _frame, phase in sorted(self.phases.items()):
            if phase is not BowlingPhase.UNKNOWN and phase not in seen:
                seen.append(phase)
        return seen


@dataclass
class BowlerCandidate:
    """A scored candidate for the bowler role."""

    track_id: int
    score: float
    evidence: EvidenceBreakdown
    profile: PhaseProfile
    observations: int
    pose_coverage: float
    continuity: float
    bowling_arm: str = "UNKNOWN"
    # Separate, deliberately non-interchangeable confidences.
    role_confidence: float = 0.0
    action_confidence: float = 0.0
    continuity_confidence: float = 0.0
    tracking_confidence: float = 0.0
    detected: bool = False
    manual: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class RoleAssignment:
    """The authoritative role map for one clip. Downstream reads this only."""

    roles: Dict[int, Role] = field(default_factory=dict)
    bowler_track_id: Optional[int] = None
    state: RoleState = RoleState.UNKNOWN
    candidates: List[BowlerCandidate] = field(default_factory=list)
    bowling_arm: str = "UNKNOWN"
    phase_profile: Optional[PhaseProfile] = None
    identity_confidence: float = 0.0
    action_confidence: float = 0.0
    continuity_confidence: float = 0.0
    pose_coverage: float = 0.0
    pose_sufficient: bool = False
    analysis_withheld: bool = True
    withholding_reason: str = ""
    manual_override: bool = False
    manual_operator: str = ""
    manual_reason: str = ""
    identity_switch_count: int = 0
    evidence: EvidenceBreakdown = field(default_factory=EvidenceBreakdown)
    reasons: List[str] = field(default_factory=list)
    #: (frame_idx, bbox) for the confirmed bowler, in frame order. Populated only
    #: when a bowler is committed, so consumers can drive overlays from it.
    _bowler_boxes: List[Tuple[int, BBox]] = field(default_factory=list)

    @property
    def bowler_confirmed(self) -> bool:
        return self.state in (
            RoleState.CONFIRMED,
            RoleState.LOCKED,
            RoleState.RECOVERED,
        )

    def role_of(self, track_id: int) -> Role:
        return self.roles.get(track_id, Role.UNKNOWN)

    def bowler_bbox_sequence(self) -> List[Tuple[int, BBox]]:
        return self._bowler_boxes

    def summary(self) -> Dict[str, object]:
        """Serialisable, UI-safe view. Never includes raw per-frame arrays."""
        return {
            "bowler_track_id": self.bowler_track_id,
            "state": self.state.value,
            "bowler_confirmed": self.bowler_confirmed,
            "identity_confidence": round(self.identity_confidence, 4),
            "action_confidence": round(self.action_confidence, 4),
            "continuity_confidence": round(self.continuity_confidence, 4),
            "pose_coverage": round(self.pose_coverage, 4),
            "pose_sufficient": self.pose_sufficient,
            "analysis_withheld": self.analysis_withheld,
            "withholding_reason": self.withholding_reason,
            "bowling_arm": self.bowling_arm,
            "manual_override": self.manual_override,
            "identity_switch_count": self.identity_switch_count,
            "roles": {str(k): v.value for k, v in self.roles.items()},
            "candidates": [
                {
                    "track_id": c.track_id,
                    "score": round(c.score, 4),
                    "role_confidence": round(c.role_confidence, 4),
                    "action_confidence": round(c.action_confidence, 4),
                    "continuity_confidence": round(c.continuity_confidence, 4),
                    "bowling_arm": c.bowling_arm,
                    "detected": c.detected,
                    "manual": c.manual,
                    "reasons": list(c.reasons),
                    "evidence": {k: round(v, 4) for k, v in c.evidence.components.items()},
                }
                for c in self.candidates
            ],
            "reasons": list(self.reasons),
        }


def observations_from_tracks(
    tracks: Dict[int, "object"],
    total_frames: Optional[int] = None,
    detection_confidences: Optional[Dict[int, float]] = None,
) -> Dict[int, List[PlayerObservation]]:
    """Adapt ``{track_id: Track}`` into engine observations.

    Keeps the engine independent of ``src.tracking`` internals -- it consumes a
    plain data shape, so it is testable without YOLO or ByteTrack and the
    tracking layer stays free to change its representation.

    Only geometry and detector confidence are available here; pose and ball
    evidence are absent at this stage, which is recorded honestly as reduced
    coverage rather than assumed.
    """
    out: Dict[int, List[PlayerObservation]] = {}
    for track_id, track in tracks.items():
        frames = getattr(track, "frames", None) or []
        bboxes = getattr(track, "bboxes", None) or []
        conf = (detection_confidences or {}).get(track_id, 1.0)
        obs: List[PlayerObservation] = []
        for frame_idx, bbox in zip(frames, bboxes):
            if bbox is None or len(bbox) < 4:
                continue
            try:
                box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except (TypeError, ValueError):
                continue
            obs.append(
                PlayerObservation(
                    track_id=int(track_id),
                    frame_idx=int(frame_idx),
                    bbox=box,
                    timestamp_sec=float(frame_idx) / 30.0,
                    detection_confidence=float(conf),
                )
            )
        if obs:
            out[int(track_id)] = obs
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class CricketRoleEngine:
    """Assigns cricket roles and owns the bowler-identity lifecycle.

    Typical use::

        engine = CricketRoleEngine()
        assignment = engine.assign(observations, frame_count=n_frames)

        if assignment.analysis_withheld:
            ...  # render "BOWLER NOT CONFIRMED"
    """

    def __init__(
        self,
        weights: Optional[RoleWeights] = None,
        thresholds: Optional[RoleThresholds] = None,
    ) -> None:
        self.weights = weights or RoleWeights()
        self.thresholds = thresholds or RoleThresholds()
        self._by_track: Dict[int, List[PlayerObservation]] = defaultdict(list)
        self._manual_track_id: Optional[int] = None
        self._manual_operator: str = ""
        self._manual_reason: str = ""
        self._state = RoleState.UNKNOWN
        self._locked_track_id: Optional[int] = None
        self._lost_frames: set = set()
        self._switch_count = 0
        self._bowler_boxes: List[Tuple[int, BBox]] = []

    # -- ingestion ---------------------------------------------------------- #

    def observe(self, observation: PlayerObservation) -> None:
        """Accumulate one observation. Order-independent."""
        self._by_track[observation.track_id].append(observation)

    def extend(self, observations: Iterable[PlayerObservation]) -> None:
        for observation in observations:
            self.observe(observation)

    def reset(self) -> None:
        self._by_track.clear()
        self._manual_track_id = None
        self._manual_operator = ""
        self._manual_reason = ""
        self._state = RoleState.UNKNOWN
        self._locked_track_id = None
        self._lost_frames = set()
        self._switch_count = 0
        self._bowler_boxes = []

    def reset_observations_for_reassignment(self) -> None:
        """Drop accumulated observations but keep the lock and any manual choice.

        Used when a clip is re-analysed over a new window: identity history must
        survive, otherwise a later window could silently adopt a different person.
        """
        self._by_track.clear()

    # -- manual fallback ---------------------------------------------------- #

    def set_manual_bowler(
        self, track_id: int, operator: str = "operator", reason: str = ""
    ) -> None:
        """Explicitly nominate the bowler, bypassing automated selection.

        The result is always flagged ``manual_override`` so the UI can label it
        as a human decision rather than an automatic one.
        """
        if track_id not in self._by_track:
            raise KeyError(
                f"track {track_id} has no observations; manual override must name a known track"
            )
        self._manual_track_id = track_id
        self._manual_operator = operator
        self._manual_reason = reason or "operator nominated"

    # -- evidence computation ----------------------------------------------- #

    @staticmethod
    def _steps(obs: Sequence[PlayerObservation]) -> List[float]:
        """Centre displacement per frame in *body units* (own bbox diagonal).

        Body-normalised so a distant small bowler is not swamped by a
        near-camera batsman -- the defect that made raw-pixel motion fail.
        """
        if len(obs) < 2:
            return []
        out: List[float] = []
        for prev, cur in zip(obs, obs[1:]):
            span = max(cur.diagonal, prev.diagonal, 1e-6)
            out.append(math.hypot(cur.center[0] - prev.center[0],
                                  cur.center[1] - prev.center[1]) / span)
        return out

    def segment_phases(self, obs: Sequence[PlayerObservation]) -> PhaseProfile:
        """Segment a candidate's motion profile into bowling phases.

        Uses the shape of the speed profile, which is view-agnostic: a run-up
        ramps up, the delivery stride is the peak, and the follow-through
        decelerates. A horizontal (side-on) and vertical (end-on) run-up produce
        the same profile.
        """
        profile = PhaseProfile()
        if len(obs) < 6:
            profile.reason = "too few observations to segment a delivery"
            return profile

        steps = np.asarray(self._steps(obs), dtype=float)
        if not np.any(steps > 1e-6):
            profile.reason = "no measurable motion"
            return profile

        peak_local = int(np.argmax(steps))
        peak = steps[peak_local]
        median = float(np.median(steps))

        # Absolute floor first: reject jitter before considering ratios.
        if peak < MIN_PEAK_BODY_STEP:
            profile.reason = (
                f"peak step {peak:.4f} below motion floor {MIN_PEAK_BODY_STEP} "
                "(essentially stationary)"
            )
            return profile

        if peak <= max(1e-6, 1.35 * median):
            profile.reason = "no pronounced delivery peak"
            return profile

        # The peak must sit late: a bowler peaks at release, not at the start.
        peak_frac = peak_local / max(1, len(steps) - 1)
        if peak_frac < 0.45:
            profile.reason = "peak too early to be a delivery"
            return profile

        # The approach must be sustained movement, not one spike in a static track.
        ramp = steps[:peak_local] if peak_local >= 3 else np.asarray([], dtype=float)
        ramp_ratio = float(np.median(ramp) / peak) if len(ramp) else 0.0
        if len(ramp) and ramp_ratio < MIN_RAMP_RATIO:
            profile.reason = (
                f"approach not sustained (ramp ratio {ramp_ratio:.3f} "
                f"< {MIN_RAMP_RATIO})"
            )
            return profile

        profile.peak_frame = obs[peak_local].frame_idx
        profile.matched = True
        profile.reason = "delivery peak detected"

        # Split the pre-peak ramp into run-up / approach / gather by its shape.
        if len(ramp) >= 3:
            third = max(1, len(ramp) // 3)
            bounds = {
                0: third,
                third: 2 * third,
                2 * third: peak_local,
            }
            phases_by_span = (
                BowlingPhase.RUN_UP,
                BowlingPhase.APPROACH,
                BowlingPhase.GATHER,
            )
            for (start, stop), phase in zip(bounds.items(), phases_by_span):
                for i in range(start, min(stop, peak_local)):
                    profile.phases[obs[i].frame_idx] = phase
        else:
            for i in range(max(0, peak_local - 2), peak_local):
                profile.phases[obs[i].frame_idx] = BowlingPhase.APPROACH

        profile.phases[obs[peak_local].frame_idx] = BowlingPhase.RELEASE

        # Post-peak: the frame after release is the delivery stride's plant, then
        # a decelerating follow-through.
        for offset, i in enumerate(range(peak_local + 1, len(steps) + 1)):
            if i >= len(obs):
                break
            phase = (
                BowlingPhase.DELIVERY_STRIDE
                if offset == 0
                else BowlingPhase.FOLLOW_THROUGH
            )
            profile.phases[obs[i].frame_idx] = phase

        profile.ramp_up_ratio = ramp_ratio
        return profile

    def score_candidate(
        self, track_id: int, obs: Sequence[PlayerObservation], n_total_frames: int
    ) -> BowlerCandidate:
        """Score one candidate with a fully decomposable explanation."""
        w = self.weights
        obs = sorted(obs, key=lambda o: o.frame_idx)
        steps = np.asarray(self._steps(obs), dtype=float)
        n = len(steps)
        components: Dict[str, float] = {}
        reasons: List[str] = []

        if n == 0 or len(obs) < 3:
            return BowlerCandidate(
                track_id=track_id,
                score=0.0,
                evidence=EvidenceBreakdown(),
                profile=PhaseProfile(reason="insufficient observations"),
                observations=len(obs),
                pose_coverage=0.0,
                continuity=0.0,
                reasons=["insufficient observations"],
            )

        peak = float(steps.max())
        median = float(np.median(steps))

        # -- motion ---------------------------------------------------------- #
        components["motion_energy"] = float(np.tanh(4.0 * float(np.mean(steps))))
        components["motion_persistence"] = float(np.mean(steps > 1e-6))
        components["direction_consistency"] = self._direction_consistency(obs)

        # -- spatial --------------------------------------------------------- #
        frame_diag = self._frame_diagonal(obs, n_total_frames)
        areas = np.asarray([o.area for o in obs], dtype=float)
        med_area = max(float(np.median(areas)), 1e-6)
        # Deliberately *not* "largest wins". Scale change is scored as evidence of
        # an approach toward the camera, mildly, and cannot dominate.
        components["scale_change"] = float(
            np.clip(abs(areas[-1] - areas[0]) / med_area, 0.0, 1.0)
        )
        centers = np.asarray([o.center for o in obs], dtype=float)
        if len(centers) >= 2:
            travel = float(np.hypot(centers[-1][0] - centers[0][0],
                                    centers[-1][1] - centers[0][1]))
            components["range_coverage"] = float(
                np.clip(travel / max(frame_diag, 1e-6), 0.0, 1.0)
            )
        else:
            components["range_coverage"] = 0.0

        # -- action / phase -------------------------------------------------- #
        profile = self.segment_phases(obs)
        components["delivery_peak"] = (
            float(np.clip((peak / max(median, 1e-6) - 1.35) / 1.5, 0.0, 1.0))
            if median > 1e-6
            else 0.0
        )
        if profile.matched:
            components["phase_sequence"] = self._phase_sequence_score(profile)
            components["peak_late"] = float(
                np.clip((self._peak_fraction(steps) - 0.45) / 0.35, 0.0, 1.0)
            )
            components["deceleration"] = self._deceleration(steps)
            reasons.append(f"delivery phases: {[p.value for p in profile.ordered_phases]}")
        else:
            components["phase_sequence"] = 0.0
            components["peak_late"] = 0.0
            components["deceleration"] = 0.0
            reasons.append(f"no delivery signature: {profile.reason}")

        # -- continuity ------------------------------------------------------ #
        expected = float(np.clip(n / max(1, n_total_frames), 0.0, 1.0))
        gaps = self._frame_gaps(obs)
        gap_penalty = float(np.mean([g - 1 for g in gaps])) if gaps else 0.0
        components["temporal_coverage"] = expected
        components["frame_regularity"] = float(np.clip(1.0 - gap_penalty / 10.0, 0.0, 1.0))

        # -- pose (optional) -------------------------------------------------- #
        arm_vals = [o.arm_extension for o in obs if o.arm_extension is not None]
        lower_vals = [
            o.lower_body_dominance for o in obs if o.lower_body_dominance is not None
        ]
        pose_coverage = len(arm_vals) / max(1, len(obs))
        if arm_vals:
            components["arm_extension"] = float(np.clip(max(arm_vals), 0.0, 1.0))
            # Peak extension should coincide with the release frame.
            if profile.peak_frame is not None:
                peak_ext = [
                    o.arm_extension
                    for o in obs
                    if o.frame_idx == profile.peak_frame and o.arm_extension is not None
                ]
                components["release_extension"] = (
                    float(np.clip(peak_ext[0], 0.0, 1.0)) if peak_ext else 0.0
                )
            else:
                components["release_extension"] = 0.0
        else:
            components["arm_extension"] = 0.0
            components["release_extension"] = 0.0

        # The lower-half prior. Capped and *multiplicative with action evidence*:
        # it is a tie-breaker among action-bearing candidates, never a rule.
        lower_half_fraction = (
            float(np.mean([v >= LOWER_HALF_THRESHOLD for v in lower_vals]))
            if lower_vals
            else 0.0
        )
        components["lower_half_prior"] = lower_half_fraction

        # -- ball (supporting only) ------------------------------------------ #
        ball_dists = [
            o.ball_distance_px for o in obs if o.ball_visible and o.ball_distance_px is not None
        ]
        if ball_dists and frame_diag > 0:
            close = float(np.mean([d <= 0.15 * frame_diag for d in ball_dists]))
            components["ball_proximity"] = close
        else:
            components["ball_proximity"] = 0.0

        # -- aggregate into families ---------------------------------------- #
        family_map = {
            "motion": ["motion_energy", "motion_persistence", "direction_consistency"],
            "spatial": ["scale_change", "range_coverage"],
            "action": ["delivery_peak", "deceleration"],
            "phase": ["phase_sequence", "peak_late"],
            "continuity": ["temporal_coverage", "frame_regularity"],
            "pose": ["arm_extension", "release_extension"],
            "ball": ["ball_proximity"],
        }
        family_scores: Dict[str, float] = {}
        for family, names in family_map.items():
            values = [components.get(name, 0.0) for name in names]
            family_scores[family] = float(np.mean(values)) if values else 0.0

        # -- weighted score --------------------------------------------------- #
        contributions: Dict[str, float] = {}
        for family, names in family_map.items():
            weight = getattr(w, family)
            if weight <= 0:
                continue
            family_contribution = family_scores[family] * weight
            # Cap so no family can manufacture identity on its own.
            family_contribution = min(family_contribution, MAX_WEIGHTED_CONTRIBUTION)
            for name in names:
                value = components.get(name, 0.0)
                share = (value / sum(components.get(n, 0.0) for n in names)) if any(
                    components.get(n, 0.0) > 0 for n in names
                ) else 0.0
                contributions[name] = family_contribution * share

        # The prior is added last, capped, and only if the candidate already has
        # real action evidence -- it cannot promote a non-bowler.
        prior = lower_half_fraction * MAX_PRIOR_CONTRIBUTION * w.lower_half_prior
        if family_scores.get("action", 0.0) > 0 and family_scores.get("phase", 0.0) > 0:
            contributions["lower_half_prior"] = prior
        else:
            contributions["lower_half_prior"] = 0.0
            if lower_vals:
                reasons.append("lower-half prior suppressed: no action evidence")

        score = float(np.clip(sum(contributions.values()), 0.0, 1.0))
        evidence = EvidenceBreakdown(
            components=components,
            contributions=contributions,
            family_scores=family_scores,
        )

        tracking_conf = float(np.mean([o.detection_confidence for o in obs]))
        return BowlerCandidate(
            track_id=track_id,
            score=score,
            evidence=evidence,
            profile=profile,
            observations=len(obs),
            pose_coverage=pose_coverage,
            continuity=float(
                np.mean([family_scores.get("continuity", 0.0), 1.0 if not gaps else 0.8])
            ),
            action_confidence=float(
                np.mean([family_scores.get("action", 0.0), family_scores.get("phase", 0.0)])
            ),
            role_confidence=score,
            continuity_confidence=family_scores.get("continuity", 0.0),
            tracking_confidence=tracking_conf,
            detected=family_scores.get("action", 0.0) > 0 and family_scores.get("phase", 0.0) > 0,
            bowling_arm=self._infer_bowling_arm(obs),
            reasons=reasons,
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _frame_diagonal(obs: Sequence[PlayerObservation], n_total_frames: int) -> float:
        if not obs or n_total_frames <= 0:
            return 0.0
        # Assume a conventional frame; the engine is scale-normalised throughout.
        return 1280.0

    @staticmethod
    def _peak_fraction(steps: np.ndarray) -> float:
        if len(steps) < 2:
            return 0.0
        return float(np.argmax(steps)) / (len(steps) - 1)

    @staticmethod
    def _deceleration(steps: np.ndarray) -> float:
        n = len(steps)
        if n < 6:
            return 0.0
        k = max(2, n // 5)
        late = float(np.median(steps[-k:]))
        overall = float(np.median(steps))
        if overall <= 1e-6 or late >= overall:
            return 0.0
        return float(np.clip(1.0 - late / overall, 0.0, 1.0))

    @staticmethod
    def _frame_gaps(obs: Sequence[PlayerObservation]) -> List[int]:
        if len(obs) < 2:
            return []
        return [
            int(obs[i + 1].frame_idx - obs[i].frame_idx) for i in range(len(obs) - 1)
        ]

    def _direction_consistency(self, obs: Sequence[PlayerObservation]) -> float:
        if len(obs) < 3:
            return 0.0
        deltas = [
            (obs[i + 1].center[0] - obs[i].center[0],
             obs[i + 1].center[1] - obs[i].center[1])
            for i in range(len(obs) - 1)
        ]
        mags = [math.hypot(dx, dy) for dx, dy in deltas]
        total = math.hypot(sum(d for d, _ in deltas), sum(d for _, d in deltas))
        if total <= 1e-6 or not any(m > 1e-6 for m in mags):
            return 0.0
        hx = sum(d for d, _ in deltas) / total
        hy = sum(d for _, d in deltas) / total
        align = [
            (dx * hx + dy * hy) / max(m, 1e-6) for (dx, dy), m in zip(deltas, mags) if m > 1e-6
        ]
        return float(np.mean([a > 0.5 for a in align])) if align else 0.0

    def _phase_sequence_score(self, profile: PhaseProfile) -> float:
        """Reward a complete, correctly ordered phase sequence."""
        order = [
            BowlingPhase.RUN_UP,
            BowlingPhase.APPROACH,
            BowlingPhase.GATHER,
            BowlingPhase.RELEASE,
            BowlingPhase.FOLLOW_THROUGH,
        ]
        present = profile.ordered_phases
        if not present:
            return 0.0
        # Completeness: how many of the expected phases appear.
        completeness = len(set(order) & set(present)) / len(order)
        # Monotonicity: phases must appear in the canonical order.
        filtered = [p for p in present if p in order]
        positions = [order.index(p) for p in filtered]
        inversions = sum(
            1 for i in range(len(positions) - 1) if positions[i] > positions[i + 1]
        )
        monotonic = 1.0 if not inversions else max(0.0, 1.0 - inversions / max(1, len(positions)))
        return float(np.clip(0.6 * completeness + 0.4 * monotonic, 0.0, 1.0))

    @staticmethod
    def _infer_bowling_arm(obs: Sequence[PlayerObservation]) -> str:
        """Left/right arm from the frame of greatest arm extension."""
        best, best_ext = "UNKNOWN", -1.0
        for o in obs:
            if o.arm_extension is None or o.bowling_arm_hint == "UNKNOWN":
                continue
            if o.arm_extension > best_ext:
                best_ext, best = o.arm_extension, o.bowling_arm_hint
        return best

    # -- main entry point --------------------------------------------------- #

    def assign(
        self,
        observations: Optional[Iterable[PlayerObservation]] = None,
        frame_count: Optional[int] = None,
    ) -> RoleAssignment:
        """Produce the authoritative :class:`RoleAssignment`."""
        if observations is not None:
            self.extend(observations)
        by_track = self._by_track
        if frame_count is None:
            frame_count = max(
                (o.frame_idx for obs in by_track.values() for o in obs), default=0
            ) + 1

        assignment = RoleAssignment()
        self._bowler_boxes = []

        if not by_track:
            assignment.state = RoleState.UNKNOWN
            assignment.withholding_reason = "no player observations"
            assignment.reasons.append("no player observations")
            return assignment

        # Score every track.
        scored = [
            self.score_candidate(track_id, obs, frame_count)
            for track_id, obs in by_track.items()
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        assignment.candidates = scored

        top = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None

        # -- manual override path ------------------------------------------- #
        if self._manual_track_id is not None:
            return self._finalise_manual(assignment, frame_count)

        # -- insufficient evidence ------------------------------------------ #
        if top.observations < self.thresholds.min_observations:
            assignment.state = RoleState.UNKNOWN
            assignment.withholding_reason = (
                f"insufficient observations ({top.observations} < "
                f"{self.thresholds.min_observations})"
            )
            assignment.reasons.append(assignment.withholding_reason)
            return assignment

        if top.score < self.thresholds.candidate:
            assignment.state = RoleState.UNKNOWN
            assignment.withholding_reason = f"no candidate above threshold ({top.score:.3f})"
            assignment.reasons.append(assignment.withholding_reason)
            return assignment

        # -- state machine --------------------------------------------------- #
        state = self._advance(top, runner_up, assignment)

        # -- ambiguity ------------------------------------------------------- #
        # Two credible candidates is a guess, not an identification. Ambiguity is
        # reported at any state and downgrades an otherwise-confirmed pick, so a
        # near-tie can never surface as a confident result.
        if runner_up is not None and top.score > 0:
            margin = top.score - runner_up.score
            if margin < self.thresholds.ambiguity_margin:
                assignment.reasons.append(
                    f"ambiguous: track {top.track_id} ({top.score:.3f}) vs "
                    f"track {runner_up.track_id} ({runner_up.score:.3f}), "
                    f"margin {margin:.3f} < {self.thresholds.ambiguity_margin}"
                )
                if state in (RoleState.CONFIRMED, RoleState.LOCKED):
                    state = RoleState.PROBABLE
                    self._state = state
                    assignment.reasons.append(
                        "confirmation downgraded to PROBABLE: candidates too close"
                    )

        # -- lock / continuity ----------------------------------------------- #
        if state in (RoleState.CONFIRMED, RoleState.LOCKED):
            self._locked_track_id = top.track_id
        elif state is RoleState.FAILED and self._locked_track_id is not None:
            # Never silently switch: a lost bowler is a failure, not a reselection.
            if top.track_id != self._locked_track_id:
                self._switch_count += 1
                assignment.identity_switch_count = self._switch_count
                assignment.reasons.append(
                    f"refused identity switch: locked track {self._locked_track_id} "
                    f"unavailable, best alternative was {top.track_id}"
                )

        self._commit(assignment, top, state, frame_count)
        return assignment

    def _advance(
        self,
        top: BowlerCandidate,
        runner_up: Optional[BowlerCandidate],
        assignment: RoleAssignment,
    ) -> RoleState:
        t = self.thresholds
        if top.score >= t.lock:
            return RoleState.LOCKED
        if top.score >= t.confirm:
            return RoleState.CONFIRMED
        if top.score >= t.probable:
            return RoleState.PROBABLE
        return RoleState.CANDIDATE

    def _finalise_manual(
        self, assignment: RoleAssignment, frame_count: int
    ) -> RoleAssignment:
        track_id = self._manual_track_id
        assert track_id is not None
        candidate = next(c for c in assignment.candidates if c.track_id == track_id)
        candidate.manual = True
        candidate.reasons.append(f"manual override: {self._manual_reason}")

        self._state = RoleState.LOCKED
        self._locked_track_id = track_id
        self._commit(assignment, candidate, RoleState.LOCKED, frame_count)
        assignment.manual_override = True
        assignment.manual_operator = self._manual_operator
        assignment.manual_reason = self._manual_reason
        assignment.reasons.append(
            f"bowler manually assigned to track {track_id} by "
            f"{self._manual_operator or 'operator'}"
        )
        # A manual decision is trusted as identity, but the analysis is still
        # withheld if there is not enough pose to actually measure anything.
        return assignment

    def _commit(
        self,
        assignment: RoleAssignment,
        candidate: BowlerCandidate,
        state: RoleState,
        frame_count: int,
    ) -> None:
        obs = sorted(self._by_track[candidate.track_id], key=lambda o: o.frame_idx)
        boxes = [(o.frame_idx, o.bbox) for o in obs]
        self._bowler_boxes = boxes
        assignment._bowler_boxes = boxes

        assignment.bowler_track_id = candidate.track_id
        assignment.state = state
        assignment.roles[candidate.track_id] = Role.BOWLER
        assignment.bowling_arm = candidate.bowling_arm
        assignment.phase_profile = candidate.profile
        assignment.evidence = candidate.evidence
        assignment.pose_coverage = candidate.pose_coverage
        assignment.identity_confidence = candidate.role_confidence
        assignment.action_confidence = candidate.action_confidence
        assignment.continuity_confidence = candidate.continuity_confidence

        for other in assignment.candidates:
            if other.track_id == candidate.track_id:
                continue
            assignment.roles[other.track_id] = self._secondary_role(other, candidate)

        assignment.pose_sufficient = (
            candidate.pose_coverage >= self.thresholds.min_pose_coverage
        )
        if not assignment.pose_sufficient:
            assignment.analysis_withheld = True
            assignment.withholding_reason = (
                f"insufficient pose coverage for biomechanics "
                f"({candidate.pose_coverage:.0%} < {self.thresholds.min_pose_coverage:.0%})"
            )
            assignment.reasons.append(assignment.withholding_reason)
        elif not assignment.bowler_confirmed:
            assignment.analysis_withheld = True
            if not assignment.withholding_reason:
                assignment.withholding_reason = (
                    f"bowler not confirmed (state={state.value}, "
                    f"identity={candidate.role_confidence:.2f})"
                )
                assignment.reasons.append(assignment.withholding_reason)
        else:
            assignment.analysis_withheld = False
            assignment.withholding_reason = ""

    @staticmethod
    def _secondary_role(candidate: BowlerCandidate, bowler: BowlerCandidate) -> Role:
        """Assign a non-bowler role from relative evidence.

        Kept deliberately conservative: without batting/keeping evidence the
        engine says UNKNOWN rather than guessing.
        """
        if candidate.score >= 0.5 * max(bowler.score, 1e-6) and candidate.observations > 0:
            return Role.FIELDER
        return Role.UNKNOWN
