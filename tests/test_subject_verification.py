"""Tests for the subject-verification gate (G1): ML predictions, SHAP and
coaching must be REFUSED whenever the pose subject could not be confirmed as
the bowler (full-frame fallback with multiple people detected in frame)."""
import pytest

from src import pipeline, config


def _bundle():
    class _M:
        def predict(self, X):
            raise AssertionError("model.predict must never run when scoring is refused")
    class Bund:
        feature_names = ["elbow_flexion_deg"]
        model = _M()
        scaler = None
        task = "performance"
        label_map = {0: "low", 1: "moderate", 2: "high"}
    return Bund()


FEATURES = {"elbow_flexion_deg": 10.0}


def _install_sinks(monkeypatch):
    """Replace ML/SHAP/coaching with counters so we can assert they did/didn't run."""
    calls = {"predict": 0, "shap": 0, "coaching": 0}

    def _predict(bundle, fv):
        calls["predict"] += 1
        return 60.0

    def _shap(bundle, fv):
        calls["shap"] += 1
        return {}

    def _coaching(fv, perf, risk, shap_contributions=None):
        calls["coaching"] += 1
        return ["Coaching note"]

    monkeypatch.setattr(pipeline.ml_models, "predict", _predict)
    monkeypatch.setattr(pipeline.explainability, "explain_prediction", _shap)
    monkeypatch.setattr(pipeline.coaching, "generate_recommendations", _coaching)
    return calls


def test_result_carries_verification_fields():
    result = pipeline.AnalysisResult(feature_vector={}, performance_score=None,
                                     injury_risk=None,
                                     shap_contributions_performance=None,
                                     shap_contributions_injury=None,
                                     coaching_notes=[])
    assert result.subject_verified is None
    assert result.scoring_blocked_reason is None


def test_scoring_withheld_when_subject_unverified(monkeypatch):
    calls = _install_sinks(monkeypatch)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), subject_verified=False)

    assert result.performance_score is None
    assert result.injury_risk is None
    assert result.shap_contributions_performance is None
    assert result.shap_contributions_injury is None
    assert result.subject_verified is False
    assert result.scoring_blocked_reason is not None
    assert calls["predict"] == 0
    assert calls["shap"] == 0
    assert calls["coaching"] == 0
    withheld = [n for n in result.coaching_notes if "withheld" in n.lower()]
    assert withheld, "coaching notes must explain that scoring was withheld"


def test_scoring_runs_when_subject_verified(monkeypatch):
    calls = _install_sinks(monkeypatch)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), subject_verified=True)

    assert result.performance_score == 60.0
    assert result.subject_verified is True
    assert result.scoring_blocked_reason is None
    assert calls["predict"] == 2  # perf + injury
    assert calls["coaching"] == 1


def test_scoring_runs_for_manual_entry_no_video(monkeypatch):
    # Manual-entry mode passes subject_verified=None (no video to verify) and
    # MUST still score -- the gate only applies to the video pipeline.
    calls = _install_sinks(monkeypatch)
    result = pipeline.analyze_feature_vector(dict(FEATURES), _bundle(), _bundle())
    assert result.performance_score == 60.0
    assert result.scoring_blocked_reason is None
    assert calls["predict"] == 2


def test_config_override_rescinds_the_gate(monkeypatch):
    _install_sinks(monkeypatch)
    monkeypatch.setattr(config, "SUBJECT_VERIFICATION_REQUIRED", False)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), subject_verified=False)
    assert result.performance_score == 60.0
    assert result.scoring_blocked_reason is None


def test_unreliable_delivery_blocks_scoring_when_gate_on(monkeypatch):
    # G6: with REFUSE_ML_ON_UNRELIABLE_DELIVERY enabled, an unreliable delivery
    # window must withhold ML/SHAP/coaching just like an unverified subject.
    calls = _install_sinks(monkeypatch)
    monkeypatch.setattr(config, "REFUSE_ML_ON_UNRELIABLE_DELIVERY", True)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), reliable=False)

    assert result.performance_score is None
    assert result.injury_risk is None
    assert result.delivery_reliable is False
    assert result.scoring_blocked_reason is not None
    assert "release" in result.scoring_blocked_reason
    assert calls["predict"] == 0 and calls["coaching"] == 0
    withheld = [n for n in result.coaching_notes if "withheld" in n.lower()]
    assert withheld


def test_unreliable_delivery_still_scores_when_gate_off(monkeypatch):
    # Default config: the reliability flag is surfaced but does not gate scoring.
    calls = _install_sinks(monkeypatch)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), reliable=False)
    assert result.performance_score == 60.0
    assert result.scoring_blocked_reason is None
    assert result.delivery_reliable is False       # flag still carried through
    assert calls["predict"] == 2


def test_reliable_delivery_scores_when_gate_on(monkeypatch):
    calls = _install_sinks(monkeypatch)
    monkeypatch.setattr(config, "REFUSE_ML_ON_UNRELIABLE_DELIVERY", True)
    result = pipeline.analyze_feature_vector(
        dict(FEATURES), _bundle(), _bundle(), reliable=True)
    assert result.performance_score == 60.0
    assert result.scoring_blocked_reason is None
    assert calls["predict"] == 2