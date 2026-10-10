"""Regression test for the feature-provenance source contract.

The producer (``feature_engineering._resolve_source``) and the consumer
(``result_view._source_for`` / ``assess_delivery``) must agree on the documented
source tags in ``docs/RESULT_REDESIGN.md``. A prior mismatch meant every feature
from a real video run fell through to "not available" while completeness could
still read "complete". Both sides now share ``config.PROV_*`` constants; this
test fails if either side drifts again.
"""
from types import SimpleNamespace

import numpy as np

from src import config, feature_engineering as fe, result_view as rv


class _Pose:
    """Minimal PoseFrame stand-in: world (33, 3) and/or normalized (33, 4)."""

    def __init__(self, world=None, landmarks=None):
        if world is not None:
            self.world_landmarks = world
        if landmarks is not None:
            self.landmarks = landmarks


def test_producer_emits_documented_tags():
    assert fe._resolve_source(_Pose(world=np.zeros((33, 3)))) == config.PROV_WORLD_3D
    assert fe._resolve_source(_Pose(landmarks=np.zeros((33, 4)))) == config.PROV_NORMALIZED_2D
    assert fe._resolve_source(None) == config.PROV_MISSING
    assert fe._resolve_source(np.zeros((33, 3))) == config.PROV_3D_ARRAY
    assert fe._resolve_source(np.zeros((33, 2))) == config.PROV_2D_ARRAY


def _result(prov):
    return SimpleNamespace(
        feature_provenance=prov,
        subject_verified=True,
        delivery_reliable=True,
        scoring_blocked_reason=None,
        bowler_confirmed=True,
        identity_switch_count=0,
        role_engine=None,
        injury_risk={},
        feature_vector={},
        landmark_source_summary={},
    )


def test_consumer_recognizes_every_producer_tag():
    # Any tag the producer can emit must be counted, never silently dropped.
    for tag, field in (
        (config.PROV_WORLD_3D, "measured"),
        (config.PROV_NORMALIZED_2D, "degraded"),
        (config.PROV_MISSING, "missing"),
    ):
        status = rv.assess_delivery(
            _result({"elbow_flexion_deg": {"source": tag}}), is_video=True)
        assert getattr(status, field) == 1, (tag, field)
        assert status.state in ("PARTIAL", "NORMAL"), status.state


def test_world_3d_only_is_complete_and_normal():
    prov = {k: {"source": config.PROV_WORLD_3D}
            for k in ("shoulder_rotation_deg", "elbow_flexion_deg")}
    status = rv.assess_delivery(_result(prov), is_video=True)
    assert status.state == "NORMAL"
    assert status.completeness == "complete"
    assert status.measured == 2


def test_2d_fallback_is_partial_not_complete():
    prov = {"elbow_flexion_deg": {"source": config.PROV_NORMALIZED_2D}}
    status = rv.assess_delivery(_result(prov), is_video=True)
    assert status.state == "PARTIAL"
    assert status.completeness == "partial"
    assert status.degraded == 1


def test_missing_is_limited():
    prov = {"elbow_flexion_deg": {"source": config.PROV_MISSING}}
    status = rv.assess_delivery(_result(prov), is_video=True)
    assert status.completeness == "limited"
    assert status.missing == 1
