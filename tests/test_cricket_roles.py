"""Tests for :mod:`src.cricket_roles`, the authoritative role engine.

These tests encode the safety contract from ``bowler_identity_audit.md``:

* a wrong bowler with a plausible analysis is an invalid result;
* identity is decided in exactly one place;
* weak evidence fails safe rather than guessing;
* a locked bowler is never silently switched.
"""

from __future__ import annotations

import math

import pytest

from src.cricket_roles import (
    LOWER_HALF_THRESHOLD,
    MAX_PRIOR_CONTRIBUTION,
    BowlingPhase,
    CricketRoleEngine,
    PlayerObservation,
    Role,
    RoleAssignment,
    RoleState,
    RoleThresholds,
    RoleWeights,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #

def delivery_track(
    track_id: int,
    n: int = 40,
    x0: float = 500.0,
    travel: float = -180.0,
    y0: float = 260.0,
    peak_at: float = 0.72,
    scale: float = 1.0,
    arm: str = "RIGHT",
    with_pose: bool = True,
    with_ball: bool = False,
    pose_fraction: float = 1.0,
) -> list[PlayerObservation]:
    """Synthesise run-up -> delivery stride -> release -> follow-through.

    A speed profile that ramps to ``peak_at`` and then decelerates is integrated
    into positions, so the per-frame displacement genuinely peaks mid-sequence.
    That peak is the signature the phase segmenter looks for; a monotonic
    constant-velocity track would not be a delivery.
    """
    # 1. Velocity profile: linear ramp up, linear ramp down.
    speeds = []
    for i in range(n):
        frac = i / max(1, n - 1)
        if frac < peak_at:
            s = frac / peak_at
        else:
            s = max(0.0, 1.0 - (frac - peak_at) / (1.0 - peak_at))
        speeds.append(s)

    # 2. Integrate to positions so total displacement equals `travel`.
    xs, x = [], x0
    for i in range(n):
        if i:
            x += (travel / n) * speeds[i]
        xs.append(x)

    obs = []
    for i in range(n):
        speed = speeds[i]
        px = xs[i]
        py = y0 - 10.0 * speed
        w, h = 46.0 * scale, 108.0 * scale
        # `pose_fraction` < 1.0 simulates frames where the pose model found no
        # usable subject, which must withhold biomechanics.
        has_pose = with_pose and (i / max(1, n - 1)) < pose_fraction
        arm_ext = max(0.0, min(1.0, (speed - 0.45) / 0.55)) if has_pose else None
        obs.append(
            PlayerObservation(
                track_id=track_id,
                frame_idx=i,
                bbox=(px, py, px + w, py + h),
                timestamp_sec=i / 30.0,
                detection_confidence=0.9,
                arm_extension=arm_ext,
                lower_body_dominance=0.62 if has_pose else None,
                bowling_arm_hint=arm if has_pose else "UNKNOWN",
                ball_distance_px=90.0 * (1.0 - speed) if with_ball else None,
                ball_visible=with_ball,
            )
        )
    return obs


def idle_track(track_id: int, n: int = 40, x: float = 200.0, y: float = 300.0,
               jitter: float = 1.5) -> list[PlayerObservation]:
    """A person who stands still and twitches -- a batsman at the crease."""
    obs = []
    for i in range(n):
        dx = jitter * math.sin(i * 0.9)
        dy = jitter * math.cos(i * 1.3)
        obs.append(
            PlayerObservation(
                track_id=track_id,
                frame_idx=i,
                bbox=(x + dx, y + dy, x + dx + 92, y + dy + 190),
                timestamp_sec=i / 30.0,
                detection_confidence=0.95,
                arm_extension=0.12,
                lower_body_dominance=0.30,
                bowling_arm_hint="RIGHT",
            )
        )
    return obs


def big_near_camera_batsman(track_id: int = 99, n: int = 40) -> list[PlayerObservation]:
    """A very large, near-camera, high-confidence person.

    This is the adversarial case from ``scripts/debug_bowler_selection.py``: the
    old geometry-only scorer was drawn to exactly this distractor.
    """
    return idle_track(track_id, n=n, x=40.0, y=120.0, jitter=3.0)


# --------------------------------------------------------------------------- #
# Enums / constants
# --------------------------------------------------------------------------- #

class TestVocabulary:
    def test_roles_are_authoritative(self):
        assert {r.name for r in Role} >= {
            "BOWLER", "STRIKER", "NON_STRIKER", "WICKETKEEPER",
            "FIELDER", "UNKNOWN", "OTHER",
        }

    def test_states_cover_lifecycle(self):
        assert {s.name for s in RoleState} >= {
            "UNKNOWN", "CANDIDATE", "PROBABLE", "CONFIRMED",
            "LOCKED", "TEMPORARILY_LOST", "RECOVERED", "FAILED",
        }

    def test_phase_order_is_canonical(self):
        order = [
            BowlingPhase.RUN_UP, BowlingPhase.APPROACH, BowlingPhase.GATHER,
            BowlingPhase.DELIVERY_STRIDE, BowlingPhase.RELEASE,
            BowlingPhase.FOLLOW_THROUGH,
        ]
        assert [p.value for p in order] == [
            "RUN_UP", "APPROACH", "GATHER", "DELIVERY_STRIDE",
            "RELEASE", "FOLLOW_THROUGH",
        ]

    def test_lower_half_prior_value(self):
        assert LOWER_HALF_THRESHOLD == 0.50

    def test_prior_cannot_dominate(self):
        assert MAX_PRIOR_CONTRIBUTION < 0.10


# --------------------------------------------------------------------------- #
# Observation geometry
# --------------------------------------------------------------------------- #

class TestPlayerObservation:
    def test_center_and_size(self):
        o = PlayerObservation(1, 0, (10, 20, 40, 100))
        assert o.center == (25.0, 60.0)
        assert o.width == 30.0
        assert o.height == 80.0
        assert o.area == 2400.0
        assert o.diagonal == pytest.approx(math.hypot(30, 80))

    def test_degenerate_bbox_is_safe(self):
        o = PlayerObservation(1, 0, (10, 20, 10, 20))
        assert o.width == 0.0 and o.height == 0.0 and o.area == 0.0


# --------------------------------------------------------------------------- #
# Phase segmentation
# --------------------------------------------------------------------------- #

class TestPhaseSegmentation:
    def test_segments_a_real_delivery(self):
        engine = CricketRoleEngine()
        profile = engine.segment_phases(delivery_track(1))
        assert profile.matched, profile.reason
        assert profile.peak_frame is not None
        phases = profile.ordered_phases
        assert BowlingPhase.RELEASE in phases
        assert BowlingPhase.FOLLOW_THROUGH in phases

    def test_refuses_when_no_peak(self):
        engine = CricketRoleEngine()
        profile = engine.segment_phases(idle_track(1))
        assert not profile.matched
        assert "peak" in profile.reason.lower()

    def test_refuses_on_too_few_frames(self):
        engine = CricketRoleEngine()
        profile = engine.segment_phases(delivery_track(1, n=3))
        assert not profile.matched
        assert "too few" in profile.reason

    def test_refuses_early_peak(self):
        """A person who lunges immediately is not mid-delivery."""
        engine = CricketRoleEngine()
        early = []
        for i in range(30):
            v = math.exp(-((i / 29.0) ** 2) * 9.0)  # peaks at frame 0
            x = 100.0 * (1.0 - v)
            early.append(
                PlayerObservation(1, i, (300 + x, 250, 346 + x, 358))
            )
        profile = engine.segment_phases(early)
        assert not profile.matched
        assert "early" in profile.reason.lower()

    def test_refuses_subpixel_jitter(self):
        """Regression: a near-stationary person must not score a full delivery.

        Both peak and median step are near-zero noise for a fidgeting person, so
        a purely *relative* peak test is trivially satisfied. This is why
        ``MIN_PEAK_BODY_STEP`` exists.
        """
        engine = CricketRoleEngine()
        jitter = [
            PlayerObservation(
                3, i,
                (200 + 1.5 * math.sin(i), 300, 292 + 1.5 * math.sin(i), 480),
                arm_extension=0.1,
                lower_body_dominance=0.95,
            )
            for i in range(40)
        ]
        profile = engine.segment_phases(jitter)
        assert not profile.matched
        assert "motion floor" in profile.reason or "sustained" in profile.reason

    def test_refuses_single_spike_in_static_track(self):
        """One large step surrounded by stillness is not a delivery."""
        engine = CricketRoleEngine()
        obs = []
        for i in range(40):
            x = 360.0 if i == 25 else 300.0
            obs.append(PlayerObservation(1, i, (x, 250, x + 46, 358)))
        profile = engine.segment_phases(obs)
        assert not profile.matched

    def test_real_delivery_clears_the_motion_floor(self):
        """The floor must not reject a genuine run-up."""
        from src.cricket_roles import MIN_PEAK_BODY_STEP

        engine = CricketRoleEngine()
        peak = max(engine._steps(delivery_track(1)))
        assert peak > MIN_PEAK_BODY_STEP
        assert engine.segment_phases(delivery_track(1)).matched


# --------------------------------------------------------------------------- #
# Candidate scoring
# --------------------------------------------------------------------------- #

class TestCandidateScoring:
    def test_delivery_outscores_idle(self):
        engine = CricketRoleEngine()
        bowler = engine.score_candidate(1, delivery_track(1), 40)
        idle = engine.score_candidate(2, idle_track(2), 40)
        assert bowler.score > idle.score

    def test_bowler_beats_large_near_camera_batsman(self):
        """The regression that motivated the engine."""
        engine = CricketRoleEngine()
        bowler = engine.score_candidate(1, delivery_track(1), 40)
        distractor = engine.score_candidate(99, big_near_camera_batsman(99, 40), 40)
        assert bowler.score > distractor.score, (
            f"bowler {bowler.score:.3f} did not beat distractor "
            f"{distractor.score:.3f}"
        )

    def test_largest_bbox_alone_is_not_the_rule(self):
        """Area must not dominate: a huge idle person must not win."""
        engine = CricketRoleEngine()
        huge = [
            PlayerObservation(7, i, (0, 0, 400, 800)) for i in range(40)
        ]
        cand = engine.score_candidate(7, huge, 40)
        assert cand.score < 0.5, "size alone must not manufacture a bowler"

    def test_evidence_is_fully_decomposable(self):
        engine = CricketRoleEngine()
        cand = engine.score_candidate(1, delivery_track(1), 40)
        assert cand.evidence.components
        assert set(cand.evidence.contributions) <= set(cand.evidence.components)
        # Contributions must reconstruct the score.
        assert sum(cand.evidence.contributions.values()) == pytest.approx(cand.score, abs=1e-6)
        assert "score=" in cand.evidence.explain()

    def test_no_single_family_can_manufacture_identity(self):
        """Every family's weighted contribution is capped."""
        engine = CricketRoleEngine()
        # A perfectly still, full-frame, high-confidence person with perfect pose.
        perfect = [
            PlayerObservation(
                5, i, (0, 0, 500, 900),
                detection_confidence=1.0,
                arm_extension=1.0,
                lower_body_dominance=1.0,
                ball_visible=True,
                ball_distance_px=0.0,
            )
            for i in range(40)
        ]
        cand = engine.score_candidate(5, perfect, 40)
        for family, value in cand.evidence.family_scores.items():
            weight = getattr(RoleWeights(), family)
            assert value * weight <= 0.34 + 1e-9, f"{family} exceeded the cap"

    def test_lower_half_prior_requires_action_evidence(self):
        """The prior is suppressed without action -- it can never assign BOWLER."""
        engine = CricketRoleEngine()
        # Idle person whose lower body is dominant the whole time.
        idle = [
            PlayerObservation(
                3, i,
                (200 + 1.5 * math.sin(i), 300, 292 + 1.5 * math.sin(i), 480),
                arm_extension=0.1,
                lower_body_dominance=0.95,
            )
            for i in range(40)
        ]
        cand = engine.score_candidate(3, idle, 40)
        assert cand.evidence.contributions.get("lower_half_prior", 0.0) == 0.0
        assert any("suppressed" in r for r in cand.reasons)

    def test_lower_half_prior_applies_with_action(self):
        engine = CricketRoleEngine()
        cand = engine.score_candidate(1, delivery_track(1), 40)
        assert cand.evidence.contributions.get("lower_half_prior", 0.0) > 0.0

    def test_left_arm_bowler_supported(self):
        engine = CricketRoleEngine()
        cand = engine.score_candidate(1, delivery_track(1, arm="LEFT"), 40)
        assert cand.bowling_arm == "LEFT"
        assert cand.score > 0.5

    def test_ball_evidence_is_supporting_only(self):
        """Ball proximity alone must not create a bowler."""
        engine = CricketRoleEngine()
        with_ball = engine.score_candidate(1, delivery_track(1, with_ball=True), 40)
        without = engine.score_candidate(1, delivery_track(1, with_ball=False), 40)
        # Adding ball evidence may nudge the score, but the ball family weight is
        # small and the difference must stay bounded.
        assert abs(with_ball.score - without.score) < 0.10

    def test_separate_confidences_are_not_collapsed(self):
        engine = CricketRoleEngine()
        cand = engine.score_candidate(1, delivery_track(1), 40)
        assert 0.0 <= cand.tracking_confidence <= 1.0
        assert 0.0 <= cand.action_confidence <= 1.0
        assert 0.0 <= cand.continuity_confidence <= 1.0
        # Tracking confidence is detector confidence, independent of role evidence.
        assert cand.tracking_confidence == pytest.approx(0.9)

    def test_score_is_bounded(self):
        engine = CricketRoleEngine()
        for track, obs in ((1, delivery_track(1)), (2, idle_track(2))):
            cand = engine.score_candidate(track, obs, 40)
            assert 0.0 <= cand.score <= 1.0


# --------------------------------------------------------------------------- #
# Assignment + safe failure
# --------------------------------------------------------------------------- #

class TestAssignment:
    def test_assigns_bowler_and_confirms(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2) + big_near_camera_batsman(99))
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id == 1
        assert result.roles[1] == Role.BOWLER
        assert result.bowler_confirmed
        assert not result.analysis_withheld

    def test_never_chooses_the_large_batsman(self):
        engine = CricketRoleEngine()
        engine.extend(idle_track(2) + big_near_camera_batsman(99) + delivery_track(1))
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id == 1
        assert result.bowler_track_id != 99

    def test_withholds_when_no_motion(self):
        """Fail safe: two people standing still must not produce a bowler."""
        engine = CricketRoleEngine()
        engine.extend(idle_track(1) + idle_track(2))
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id is None
        assert result.analysis_withheld
        assert result.withholding_reason

    def test_withholds_on_insufficient_observations(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1, n=4))
        result = engine.assign(frame_count=40)
        assert result.analysis_withheld
        assert "insufficient observations" in result.withholding_reason

    def test_withholds_with_no_observations(self):
        result = CricketRoleEngine().assign([])
        assert result.state is RoleState.UNKNOWN
        assert result.analysis_withheld
        assert result.bowler_track_id is None

    def test_withholds_when_pose_insufficient(self):
        """Identity may be known but biomechanics still must not run."""
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1, with_pose=False) + idle_track(2))
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id == 1
        assert not result.pose_sufficient
        assert result.analysis_withheld
        assert "pose coverage" in result.withholding_reason

    def test_pose_coverage_threshold_is_configurable(self):
        """A partial pose is acceptable at a low threshold, not at a high one.

        Note identity confidence legitimately drops here too: with no pose at the
        release frame, ``release_extension`` is 0, so this is not merely a
        coverage-gating effect.
        """
        lenient = CricketRoleEngine(thresholds=RoleThresholds(min_pose_coverage=0.30))
        lenient.extend(delivery_track(1, pose_fraction=0.4) + idle_track(2))
        ok = lenient.assign(frame_count=40)
        assert ok.pose_coverage == pytest.approx(0.4, abs=0.05)
        assert ok.pose_sufficient

        strict = CricketRoleEngine(thresholds=RoleThresholds(min_pose_coverage=0.80))
        strict.extend(delivery_track(1, pose_fraction=0.4) + idle_track(2))
        blocked = strict.assign(frame_count=40)
        assert blocked.bowler_track_id == 1
        assert not blocked.pose_sufficient
        assert blocked.analysis_withheld
        assert "pose coverage" in blocked.withholding_reason

    def test_candidates_are_ranked_and_exposed(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        result = engine.assign(frame_count=40)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)
        assert result.candidates[0].track_id == 1

    def test_bowler_bbox_sequence_is_frame_ordered(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        result = engine.assign(frame_count=40)
        frames = [f for f, _ in result.bowler_bbox_sequence()]
        assert frames == sorted(frames)
        assert len(frames) == 40

    def test_summary_is_serialisable_and_safe(self):
        import json

        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        result = engine.assign(frame_count=40)
        payload = json.dumps(result.summary())
        assert "bowler_track_id" in payload
        # Raw per-frame arrays must not leak into the UI payload.
        assert "bowler_bboxes" not in payload

    def test_assignment_is_deterministic(self):
        scores = []
        for _ in range(3):
            engine = CricketRoleEngine()
            engine.extend(delivery_track(1) + idle_track(2) + big_near_camera_batsman(99))
            scores.append(engine.assign(frame_count=40).bowler_track_id)
        assert len(set(scores)) == 1


# --------------------------------------------------------------------------- #
# Manual fallback
# --------------------------------------------------------------------------- #

class TestManualOverride:
    def test_manual_selection_is_honoured_and_flagged(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        engine.set_manual_bowler(2, operator="coach", reason="reviewed footage")
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id == 2
        assert result.manual_override
        assert result.manual_operator == "coach"
        assert result.manual_reason == "reviewed footage"
        assert result.state is RoleState.LOCKED

    def test_manual_override_of_unknown_track_raises(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1))
        with pytest.raises(KeyError):
            engine.set_manual_bowler(42)

    def test_manual_candidate_is_marked(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        engine.set_manual_bowler(2, operator="coach")
        result = engine.assign(frame_count=40)
        manual = next(c for c in result.candidates if c.track_id == 2)
        assert manual.manual
        assert any("manual override" in r for r in manual.reasons)

    def test_manual_still_requires_pose_for_analysis(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1, with_pose=False) + idle_track(2))
        engine.set_manual_bowler(1, operator="coach")
        result = engine.assign(frame_count=40)
        assert result.bowler_track_id == 1
        assert result.analysis_withheld
        assert "pose coverage" in result.withholding_reason

    def test_reset_clears_manual_state(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        engine.set_manual_bowler(2, operator="coach")
        engine.reset()
        # reset() also drops observations, so re-supply them.
        engine.extend(delivery_track(1) + idle_track(2))
        result = engine.assign(frame_count=40)
        assert not result.manual_override
        assert result.bowler_track_id == 1


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #

class TestLocking:
    def test_high_confidence_locks(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1, n=60) + idle_track(2))
        result = engine.assign(frame_count=60)
        assert result.state in (RoleState.CONFIRMED, RoleState.LOCKED)

    def test_locked_identity_is_not_silently_switched(self):
        """Re-assigning without the locked bowler must not adopt another track."""
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2))
        first = engine.assign(frame_count=40)
        assert first.bowler_track_id == 1

        # The bowler disappears entirely; only two idle people remain.
        engine.reset_observations_for_reassignment()
        result = engine.assign(observations=idle_track(2) + idle_track(3), frame_count=40)
        assert result.bowler_track_id != 2
        assert result.analysis_withheld

    def test_thresholds_control_state(self):
        strict = CricketRoleEngine(
            thresholds=RoleThresholds(confirm=0.99, lock=0.995)
        )
        strict.extend(delivery_track(1) + idle_track(2))
        result = strict.assign(frame_count=40)
        assert result.state in (RoleState.PROBABLE, RoleState.CANDIDATE)
        assert not result.bowler_confirmed
        assert result.analysis_withheld


# --------------------------------------------------------------------------- #
# Regression fixtures
# --------------------------------------------------------------------------- #

class TestAmbiguityReporting:
    def test_ambiguity_is_reported_when_scores_are_close(self):
        engine = CricketRoleEngine()
        # Two near-identical deliveries -> genuinely ambiguous.
        engine.extend(delivery_track(1) + delivery_track(2))
        result = engine.assign(frame_count=40)
        assert any("ambiguous" in r for r in result.reasons)

    def test_no_ambiguity_when_one_candidate_is_clearly_ahead(self):
        engine = CricketRoleEngine()
        engine.extend(delivery_track(1) + idle_track(2) + big_near_camera_batsman(99))
        result = engine.assign(frame_count=40)
        assert not any("ambiguous" in r for r in result.reasons)
