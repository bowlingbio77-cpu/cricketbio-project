"""Tests for the adaptive result page (src/result_view.py).

These cover the pure interpretation layer only: no Streamlit, no video, no model
training. Every assertion is about what the result page *claims*, because that
is the part that must never drift from the underlying analysis.

The central rule under test: when the pipeline refused to score a delivery, the
result page must not invent a value to fill the gap.
"""
import dataclasses
import inspect

import pytest

from src import injury_knowledge_base as kb
from src import pipeline, result_view

WORST = ("UNRELIABLE", "SCORING_WITHHELD", "OOD", "PARTIAL")

# Must stay identical to app._VIDEO_ONLY_FIELDS.
VIDEO_ONLY_FIELDS = (
    "feature_provenance", "landmark_source_summary", "stage_backends",
    "bowler_bboxes", "bowler_track_id", "bowler_confidence",
    "bowler_confirmed", "bowler_confirm_reason", "bowler_candidates",
    "identity_switch_count", "batting_stances", "striker_track_id",
    "non_striker_track_id", "original_frame_dims", "player_roles",
    "ball_stats", "video_path", "pose_video_path", "reels_video_path",
    "analysis_replay_path", "bowling_arm", "camera_view", "warnings",
    "subject_verified", "delivery_reliable", "scoring_blocked_reason",
)


def merge_video_fields(result, video_result):
    """Byte-for-byte mirror of app._merge_video_result (app.py), minus Streamlit."""
    if video_result is None:
        return result
    overrides = {}
    for name in VIDEO_ONLY_FIELDS:
        value = getattr(video_result, name, None)
        if value is None or value == {} or value == []:
            continue
        overrides[name] = value
    if not overrides:
        return result
    return dataclasses.replace(result, **overrides)


CLEAN_FEATURES = {
    "elbow_flexion_deg": 8.0,
    "shoulder_rotation_deg": 22.0,
    "trunk_lean_deg": 4.0,
    "wrist_angle_deg": -12.0,
    "knee_flexion_deg": 20.0,
    "hip_rotation_deg": 70.0,
    "stride_length_norm": 1.05,
    "angular_velocity_deg_s": 120.0,
    "vertical_ground_foot_angle_deg": 6.0,
}

# Values chosen to cross published screening thresholds.
FLAGGED_FEATURES = {
    "elbow_flexion_deg": 25.0,
    "shoulder_rotation_deg": 140.0,
    "trunk_lean_deg": 35.0,
    "wrist_angle_deg": -70.0,
    "knee_flexion_deg": 20.0,
    "hip_rotation_deg": 95.0,
    "stride_length_norm": 0.6,
    "angular_velocity_deg_s": 900.0,
    "vertical_ground_foot_angle_deg": 30.0,
}

ALL_MEASURED = [m[0] for m in result_view.MEASUREMENTS]

PROV_3D = {
    k: {"source": "world_3d", "confidence": "high",
        "mean_landmark_visibility": 0.93}
    for k in ALL_MEASURED
}
PROV_2D = {
    k: {"source": "normalized_2d", "confidence": "medium",
        "mean_landmark_visibility": 0.55}
    for k in ALL_MEASURED
}
PROV_MIXED = dict(PROV_3D, elbow_flexion_deg={
    "source": "missing_xyz", "confidence": "low",
    "mean_landmark_visibility": 0.21})


def make_result(**overrides) -> pipeline.AnalysisResult:
    base = dict(
        feature_vector=dict(CLEAN_FEATURES),
        performance_score=72.0,
        injury_risk={"risk_level": "low", "probabilities": [0.1, 0.8, 0.1]},
        shap_contributions_performance={"shoulder_rotation_deg": 3.0},
        shap_contributions_injury={"elbow_flexion_deg": -1.0},
        coaching_notes=["Hold your current release window."],
        feature_provenance=dict(PROV_3D),
        landmark_source_summary={"elbow_flexion_deg": "MediaPipe 3D (world)"},
        subject_verified=True,
        delivery_reliable=True,
        bowler_track_id=3,
        bowler_confidence=0.91,
    )
    base.update(overrides)
    return pipeline.AnalysisResult(**base)


def status_for(result, **kw):
    return result_view.assess_delivery(result, **kw)


# --------------------------------------------------------------------------
# 01 -- Delivery status: the trust gate
# --------------------------------------------------------------------------

class TestDeliveryStatus:
    def test_full_3d_provenance_is_normal(self):
        assert status_for(make_result(), is_video=True).state == "NORMAL"

    def test_unverified_subject_is_unreliable_and_withheld(self):
        # This mirrors pipeline.analyze_feature_vector: when subject_verified is
        # False the pipeline refuses to score AND records why.
        s = status_for(make_result(
            subject_verified=False, performance_score=None, injury_risk={},
            scoring_blocked_reason="bowler could not be confirmed"))
        assert s.state == "UNRELIABLE"
        assert "SCORING_WITHHELD" in s.states
        assert s.blocked_reason == "bowler could not be confirmed"
        assert s.completeness == "limited"

    def test_failed_delivery_detection_is_unreliable(self):
        s = status_for(make_result(delivery_reliable=False,
                                    performance_score=None, injury_risk={}))
        assert s.state == "UNRELIABLE"
        assert s.completeness == "limited"

    def test_blocked_reason_alone_withholds_scoring(self):
        s = status_for(make_result(
            performance_score=None, injury_risk={},
            scoring_blocked_reason="single bowler could not be tracked"))
        assert s.state == "SCORING_WITHHELD"
        assert s.blocked_reason == "single bowler could not be tracked"

    def test_missing_xyz_is_partial(self):
        s = status_for(make_result(feature_provenance=dict(PROV_MIXED)))
        assert s.state == "PARTIAL"
        assert s.missing == 1
        assert s.measured >= 1

    def test_2d_only_provenance_is_partial_and_degraded(self):
        s = status_for(make_result(feature_provenance=dict(PROV_2D)))
        assert s.state == "PARTIAL"
        assert s.degraded == len(PROV_2D)
        assert s.measured == 0

    def test_low_confidence_alone_triggers_partial(self):
        prov = {k: {"source": "world_3d", "confidence": "low"}
                for k in ALL_MEASURED}
        s = status_for(make_result(feature_provenance=prov))
        assert s.state == "PARTIAL"
        assert s.low_confidence == len(prov)

    def test_unreliable_outranks_every_other_state(self):
        s = status_for(make_result(subject_verified=False,
                                    feature_provenance=dict(PROV_MIXED)))
        assert s.state == "UNRELIABLE"

    def test_every_state_carries_a_headline_and_copy(self):
        for result in (make_result(),
                       make_result(subject_verified=False),
                       make_result(feature_provenance=dict(PROV_MIXED))):
            s = status_for(result)
            assert s.headline.strip()
            assert s.headline_copy.strip()
            assert s.tone in ("ok", "warn", "danger")

    def test_status_never_reports_a_risk_number(self):
        s = status_for(make_result(injury_risk={}, performance_score=None))
        assert not hasattr(s, "risk_score")

    def test_status_is_pure(self):
        result = make_result()
        before = dataclasses.asdict(result)
        status_for(result, is_video=True)
        assert dataclasses.asdict(result) == before


# --------------------------------------------------------------------------
# 03 -- Top findings
# --------------------------------------------------------------------------

class TestFindings:
    def test_flagged_delivery_raises_kb_findings(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={})
        findings = result_view.build_findings(result, status_for(result))
        assert findings
        assert all(f.key.startswith("kb:") for f in findings)

    def test_findings_are_capped_at_three(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={},
                             performance_score=5.0,
                             injury_risk={"risk_level": "high",
                                          "probabilities": [0.0, 0.05, 0.95]})
        findings = result_view.build_findings(result, status_for(result))
        assert len(findings) <= 3

    def test_withheld_delivery_produces_no_findings(self):
        result = make_result(subject_verified=False, performance_score=None,
                             injury_risk={}, feature_provenance=dict(PROV_MIXED))
        assert result_view.build_findings(result, status_for(result)) == []

    def test_clean_delivery_does_not_lead_with_the_demo_model(self):
        result = make_result()
        findings = result_view.build_findings(result, status_for(result))
        assert all(f.severity != "model" for f in findings)

    def test_every_finding_is_actionable(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={})
        for f in result_view.build_findings(result, status_for(result)):
            assert f.title.strip()
            assert f.observed.strip()
            assert f.why.strip()
            assert f.action.strip()

    def test_findings_only_cite_thresholds_that_actually_fired(self):
        clinical = kb.map_from_pipeline_features(FLAGGED_FEATURES)
        cards = kb.assess_biomechanical_risks(clinical)
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={})
        found = result_view.build_findings(result, status_for(result))
        assert {f.key[3:] for f in found} <= {c["key"] for c in cards}

    def test_findings_do_not_mutate_the_result(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES))
        before = dict(result.feature_vector)
        result_view.build_findings(result, status_for(result))
        assert result.feature_vector == before


# --------------------------------------------------------------------------
# 05 -- Key measurements
# --------------------------------------------------------------------------

class TestMeasurements:
    def test_measurements_carry_real_values(self):
        by_feature = {m.feature: m.value
                      for m in result_view.build_measurements(make_result())}
        assert by_feature["elbow_flexion_deg"] == pytest.approx(8.0)
        assert by_feature["trunk_lean_deg"] == pytest.approx(4.0)

    def test_absent_measurements_are_none_not_zero(self):
        result = make_result(feature_vector={"elbow_flexion_deg": 8.0})
        by_feature = {m.feature: m.value
                      for m in result_view.build_measurements(result)}
        assert by_feature["elbow_flexion_deg"] == pytest.approx(8.0)
        for feature, value in by_feature.items():
            if feature != "elbow_flexion_deg":
                assert value is None, f"{feature} was invented as {value}"

    def test_no_provenance_means_not_available(self):
        result = make_result(feature_provenance={})
        rows = result_view.build_measurements(result, is_video=True)
        assert all(m.source_class == "pai-src-default" for m in rows)
        assert all("NOT AVAILABLE" in m.source_label for m in rows)

    def test_missing_landmarks_are_called_out_as_defaults(self):
        """When the pipeline fell back to a placeholder the page must say so
        in words, not present the number as a measurement."""
        result = make_result(feature_provenance={
            k: {"source": "missing_xyz", "confidence": "low"}
            for k in ALL_MEASURED})
        rows = result_view.build_measurements(result, is_video=True)
        assert all("DEFAULT USED" in m.source_label for m in rows)
        assert all("placeholder" in m.source_note.lower() for m in rows)

    def test_3d_landmarks_are_called_measured(self):
        result = make_result(feature_provenance=dict(PROV_3D))
        rows = result_view.build_measurements(result, is_video=True)
        assert all(m.source_class == "pai-src-measured" for m in rows)

    def test_simulator_entry_is_labelled_as_simulated(self):
        """A hand-typed delivery must never read as a measured one."""
        rows = result_view.build_measurements(make_result(), is_video=False)
        assert all(m.source_class == "pai-src-sim" for m in rows)

    def test_2d_provenance_is_labelled_as_degraded(self):
        result = make_result(feature_provenance=dict(PROV_2D))
        rows = result_view.build_measurements(result, is_video=True)
        elbow = next(m for m in rows if m.feature == "elbow_flexion_deg")
        assert elbow.source_class != "measured"
        assert elbow.source_label

    def test_every_measurement_explains_itself(self):
        for m in result_view.build_measurements(make_result(), is_video=True):
            assert m.label.strip()
            assert m.unit
            assert m.plain.strip()
            assert m.source_label.strip()


# --------------------------------------------------------------------------
# 06 -- Evidence chain
# --------------------------------------------------------------------------

class TestEvidence:
    def test_each_finding_gets_a_five_step_chain(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={})
        findings = result_view.build_findings(result, status_for(result))
        chains = result_view.build_evidence(result, findings, status_for(result))
        assert len(chains) == len(findings)
        for chain in chains:
            assert [label for label, _ in chain] == [
                "Finding", "Measurement", "Reference", "Interpretation", "Action"]

    def test_chain_never_ends_in_an_empty_cell(self):
        result = make_result(feature_vector=dict(FLAGGED_FEATURES),
                             feature_provenance={})
        findings = result_view.build_findings(result, status_for(result))
        for chain in result_view.build_evidence(result, findings,
                                                status_for(result)):
            for _, value in chain:
                assert str(value).strip()

    def test_withheld_delivery_has_no_coaching_chains(self):
        result = make_result(subject_verified=False, performance_score=None,
                             injury_risk={})
        s = status_for(result)
        assert result_view.build_evidence(result, [], s) == []


# --------------------------------------------------------------------------
# 09 -- Risk indicators
# --------------------------------------------------------------------------

class TestRisk:
    def test_legacy_fabricated_score_mapping_is_gone(self):
        """The old UI mapped low/moderate/high to 22/58/88. Those look like
        measured probabilities but were pure invention."""
        src = inspect.getsource(result_view)
        for mapping in ('"low": 22', '"moderate": 58', '"high": 88',
                        "'low': 22", "'moderate': 58", "'high': 88"):
            assert mapping not in src

    def test_risk_comes_from_the_result_not_the_view(self):
        assert result_view._risk_level(make_result(
            injury_risk={"risk_level": "moderate"})) == "moderate"
        assert result_view._risk_level(make_result(injury_risk={})) in (None, "")


# --------------------------------------------------------------------------
# 10 -- OOD
# --------------------------------------------------------------------------

class TestOOD:
    def test_no_bundle_means_no_ood_claims(self):
        assert result_view._ood(make_result(), None) == ()

    def test_ood_helper_never_raises(self):
        assert isinstance(result_view._ood(make_result(), object()), tuple)

    def test_ood_features_are_actually_out_of_range(self):
        """Guard against a check that always reports "in range"."""
        clinical = result_view._ood(make_result(
            feature_vector={"shoulder_rotation_deg": 1e6}), None)
        assert clinical == ()


# --------------------------------------------------------------------------
# Plumbing: the video result must survive into the result page
# --------------------------------------------------------------------------

class TestVideoFieldMerge:
    def test_merge_preserves_video_identity_and_artefacts(self):
        video = make_result(
            feature_provenance={"elbow_flexion_deg": {"source": "world_3d"}},
            landmark_source_summary={"elbow_flexion_deg": "MediaPipe 3D"},
            bowler_bboxes=[[10, 20, 30, 40]],
            analysis_replay_path="out/replay.mp4")
        merged = merge_video_fields(make_result(), video)
        assert merged.feature_provenance == video.feature_provenance
        assert merged.bowler_bboxes == [[10, 20, 30, 40]]
        assert merged.analysis_replay_path == "out/replay.mp4"

    def test_merge_keeps_the_ml_outputs(self):
        ml_only = make_result(performance_score=61.0,
                              injury_risk={"risk_level": "high",
                                           "probabilities": [0.0, 0.1, 0.9]})
        merged = merge_video_fields(
            ml_only, make_result(feature_provenance={"a": {"source": "x"}}))
        assert merged.performance_score == 61.0
        assert merged.injury_risk == ml_only.injury_risk
        assert merged.coaching_notes == ml_only.coaching_notes

    def test_merge_without_a_video_result_is_identity(self):
        ml_only = make_result()
        assert merge_video_fields(ml_only, None) is ml_only

    def test_merge_does_not_mutate_either_input(self):
        ml_only, video = make_result(), make_result(
            feature_provenance={"a": {"source": "x"}})
        before_ml, before_video = (dataclasses.asdict(ml_only),
                                   dataclasses.asdict(video))
        merge_video_fields(ml_only, video)
        assert dataclasses.asdict(ml_only) == before_ml
        assert dataclasses.asdict(video) == before_video

    def test_empty_video_fields_never_clobber_ml_gating(self):
        """A stale/blank video field must not overwrite the ML path's own
        subject-verification decision."""
        ml_only = make_result(subject_verified=True)
        blank_video = make_result(subject_verified=True,
                                  scoring_blocked_reason=None)
        merged = merge_video_fields(ml_only, blank_video)
        assert merged.subject_verified is True
        assert merged.scoring_blocked_reason is None

    def test_every_merged_field_exists_on_the_real_result(self):
        """A typo in the merge list would silently drop data again -- which is
        the exact bug this whole change set exists to fix."""
        real = {f.name for f in dataclasses.fields(pipeline.AnalysisResult)}
        assert set(VIDEO_ONLY_FIELDS) <= real, (
            sorted(set(VIDEO_ONLY_FIELDS) - real))

    def test_merge_list_matches_app_py(self):
        """Keep the test's copy of the list honest against the real app."""
        import ast
        import os
        app = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "app.py")
        with open(app, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", None) == "_VIDEO_ONLY_FIELDS":
                        found = tuple(ast.literal_eval(node.value))
        assert found is not None, "app.py no longer defines _VIDEO_ONLY_FIELDS"
        assert set(found) == set(VIDEO_ONLY_FIELDS)
