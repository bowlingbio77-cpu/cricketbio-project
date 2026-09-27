"""Kinetic workspace rail: identity claims must never outrun the evidence.

The rail sits next to the video and states who was measured. The result page
withholds scoring when the subject is unverified (result_view.assess_delivery),
so the rail must not assert a role or a confidence in that state -- and it must
disappear entirely when no track was ever locked.
"""
import inspect
from pathlib import Path

from src import kinetic_ui


def _rail(meta: dict) -> str:
    return kinetic_ui.rail_html(meta)


def _base(**over) -> dict:
    meta = dict(track=5, conf=0.69, role="bowler", verified=True)
    meta.update(over)
    return meta


class TestBowlerIdentityCard:
    def test_hidden_when_no_track_was_locked(self):
        html = _rail(_base(track=None))
        assert "Bowler Identity" not in html

    def test_hidden_when_track_is_absent_even_if_a_role_exists(self):
        # A role with no track id is not an identity: nothing was locked.
        html = _rail(_base(track=None, role="bowler"))
        assert "Bowler Identity" not in html

    def test_shown_with_confidence_when_verified(self):
        html = _rail(_base())
        assert "Bowler Identity" in html
        assert "Track #5" in html
        assert "confidence 0.69" in html
        assert "BOWLER" in html

    def test_unverified_subject_does_not_assert_a_role(self):
        html = _rail(_base(verified=False))
        assert "Bowler Identity" in html
        assert "UNCONFIRMED" in html
        assert "identity not confirmed" in html
        assert "BOWLER" not in html
        assert "0.69" not in html

    def test_unverified_subject_does_not_leak_confidence(self):
        # The confidence number itself must not appear anywhere in the rail.
        html = _rail(_base(verified=False, conf=0.95))
        assert "confidence" not in html
        assert "0.95" not in html

    def test_missing_confidence_reports_lock_not_a_number(self):
        html = _rail(_base(conf=None))
        assert "identity locked" in html
        assert "confidence" not in html

    def test_absent_confidence_still_withheld_when_unverified(self):
        html = _rail(_base(verified=False, conf=None))
        assert "identity not confirmed" in html
        assert "identity locked" not in html


class TestRiskWithheldAgreesWithResultPage:
    def test_donut_still_gated_independently_of_identity(self):
        # Suppressing the identity must not suppress (or invent) the risk donut.
        meta = _base(risk_pct=81.0, risk_level="high")
        assert "Block Load Tolerance" in _rail(meta)
        assert "Block Load Tolerance" not in _rail(dict(meta, risk_pct=None))


class TestNoFabricatedRoleFallback:
    def test_app_py_passes_no_bare_bowler_role_fallback(self):
        # app.py used to pass role=getattr(result, "bowler_role", None) or
        # "bowler", which fed rail_html a role for every clip and made the
        # identity card impossible to withhold. The default must not return.
        app_src = Path(__file__).resolve().parents[1] / "app.py"
        src = app_src.read_text(encoding="utf-8")
        assert 'bowler_role", None) or "bowler"' not in src

    def test_identity_gate_depends_on_track_not_role(self):
        # The card may only appear when a track id was actually locked.
        src = inspect.getsource(kinetic_ui)
        assert 'if meta.get("track") is not None:' in src
