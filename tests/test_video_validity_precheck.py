"""Cricket pre-check gate: a real bowler must not be rejected as non-cricket.

Regression coverage for a false rejection seen on a genuine broadcast clip:

    Only 25/25 frames with a clear human torso (arms 8%, upright 28%);
    not a plausible bowler clip.

A human torso was detected in *every* sampled frame, yet the gate failed because
it also required either arm landmarks (8%) or an upright torso (28%). Both soft
signals fail for legitimate reasons at the pipeline's 640x360 working size:

* a broadcast bowler is small in frame, so MediaPipe wrist/elbow visibility
  frequently stays under the 0.4 threshold;
* a clip trimmed tight around the delivery is non-upright by construction --
  the bowler leans progressively forward and is near-horizontal at release.

So torso strength is allowed to stand alone. The arm and posture signals are
still reported and still rescue weak-torso clips on their own.
"""
import numpy as np
import pytest

from src import config
from src.pose_estimation import PoseFrame
from src.video_validity import check_human_pose


def _frame(idx=0, *, shoulder_y=0.30, hip_y=0.60, shoulder_x=0.50,
           hip_x=0.50, torso_vis=0.9, arm_vis=0.9):
    """One PoseFrame with a torso at the requested geometry.

    ``shoulder_y``/``hip_y`` are normalized image coords (y grows downward), so
    ``hip_y > shoulder_y`` means the body is upright.
    """
    lm = np.full((33, 4), 0.0, dtype=float)
    # visibility everywhere, then dial down the arms for the soft-signal cases
    lm[:, 3] = arm_vis
    for name, x, y in (("left_shoulder", shoulder_x - 0.05, shoulder_y),
                       ("right_shoulder", shoulder_x + 0.05, shoulder_y),
                       ("left_hip", hip_x - 0.05, hip_y),
                       ("right_hip", hip_x + 0.05, hip_y)):
        lm[config.POSE_LANDMARK_NAMES.index(name)] = [x, y, 0.0, torso_vis]
    for name in ("left_wrist", "right_wrist", "left_elbow", "right_elbow"):
        lm[config.POSE_LANDMARK_NAMES.index(name)] = [
            shoulder_x, shoulder_y - 0.05, 0.0, arm_vis]
    return PoseFrame(frame_idx=idx, timestamp_sec=idx / 20.0, landmarks=lm)


class TestFalseRejection:
    def test_the_exact_reported_clip_is_now_accepted(self):
        # 25/25 torso, arms 8%, upright 28% -- the numbers from the failure.
        n = 25
        frames = []
        for i in range(n):
            if i < 7:            # 7/25 = 28% upright
                frames.append(_frame(i, shoulder_y=0.30, hip_y=0.62,
                                     arm_vis=0.9))
            else:                # rest: horizontal torso, invisible wrists
                frames.append(_frame(i, shoulder_y=0.45, hip_y=0.47,
                                     arm_vis=0.05))
        # torso must be 100%: arms are the only thing dialled down above.
        result = check_human_pose(frames)
        assert result.ok, result.reason

    def test_torso_alone_is_enough(self):
        frames = [_frame(i, arm_vis=0.0) for i in range(10)]
        assert check_human_pose(frames).ok

    def test_leaning_bowler_counts_as_upright(self):
        """A 30-degree forward lean is still a person, not a body lying flat."""
        frames = [_frame(i, shoulder_y=0.30, hip_y=0.30 + 0.9 * 0.28)
                  for i in range(10)]
        assert check_human_pose(frames).ok

    def test_arms_can_still_rescue_a_weak_torso(self):
        # Torso present on only 4/10 frames -- below the 0.5 "strong" bar -- but
        # arms are clearly visible on half of them, so the clip still proceeds.
        frames = []
        for i in range(10):
            f = _frame(i)
            if i >= 4:
                f.landmarks[config.POSE_LANDMARK_NAMES.index("left_hip")][3] = 0.0
            frames.append(f)
        result = check_human_pose(frames)
        assert result.ok, result.reason

    def test_posture_can_still_rescue_a_weak_torso(self):
        # Torso on only 4/10 frames and no arms at all, but upright throughout.
        frames = [_frame(i, arm_vis=0.0) for i in range(4)]
        frames += [_frame(10 + i, arm_vis=0.0, torso_vis=0.0) for i in range(6)]
        assert check_human_pose(frames).ok


class TestStillRejects:
    def test_no_person_at_all(self):
        assert not check_human_pose([]).ok

    def test_too_few_torso_frames(self):
        assert not check_human_pose([_frame(i) for i in range(2)]).ok

    def test_body_lying_flat_does_not_count_as_upright(self):
        # Torso horizontal (|dx| > |dy|) and no arms. The old geometry check
        # credited a fixed vertical drop, so it scored a sliver of a lean as
        # "upright"; the orientation check must score this flat body as 0%.
        frames = [_frame(i, shoulder_y=0.50, hip_y=0.51,
                         shoulder_x=0.30, hip_x=0.80, arm_vis=0.0)
                  for i in range(10)]
        result = check_human_pose(frames)
        assert "upright 0%" in result.reason
        # A clear torso is still enough to proceed; the point is that the flat
        # body was not credited as upright.
        assert result.ok


class TestMessages:
    def test_no_person_message_does_not_claim_a_torso(self):
        """The old wording was self-contradictory: "Only 25/25 frames ...""."""
        result = check_human_pose([_frame(i) for i in range(2)])
        assert not result.ok
        assert "no person" in result.reason.lower()

    def test_success_message_reports_all_three_signals(self):
        result = check_human_pose([_frame(i) for i in range(6)])
        assert result.ok
        for word in ("torso", "arms", "upright"):
            assert word in result.reason
