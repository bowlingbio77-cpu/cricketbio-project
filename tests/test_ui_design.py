"""Customize UI config: default, validation, persistence, reset, export/import."""
import json
import os
import subprocess
import sys

import pytest

from src import ui_design


@pytest.fixture(autouse=True)
def _isolated_path(tmp_path, monkeypatch):
    """Every test points the module at its own config file and restores it."""
    monkeypatch.setattr(ui_design, "CONFIG_PATH", str(tmp_path / "ui_design.json"))
    yield
    for suffix in ("", ".bak", ".corrupt", ".tmp"):
        p = str(tmp_path / ("ui_design.json" + suffix))
        if os.path.exists(p):
            os.remove(p)


def test_default_loads_when_no_file_exists(tmp_path):
    cfg = ui_design.load()
    assert cfg == ui_design.default_config()
    # A plain rerun must not create or clobber a file.
    assert not os.path.exists(ui_design.CONFIG_PATH)


def test_default_is_valid():
    ok, err = ui_design.validate(ui_design.default_config())
    assert ok, err


def test_valid_roundtrip_persists():
    cfg = ui_design.default_config()
    cfg["hero"]["headline"] = "Edited headline"
    cfg["theme"]["accent"] = "#112233"
    cfg["section_gap_px"] = 40
    ok, err = ui_design.save(cfg)
    assert ok, err
    assert ui_design.load()["hero"]["headline"] == "Edited headline"
    assert ui_design.load()["theme"]["accent"] == "#112233"
    assert ui_design.load()["hero"]["headline"] != ui_design.default_config()["hero"]["headline"]


@pytest.mark.parametrize("mutate,expected", [
    (lambda c: c.update({"injected": "x"}), "unknown keys"),
    (lambda c: c.update({"theme": {"accent": "red"}}), "exactly"),
    (lambda c: c["theme"].update({"accent": "red"}), "#RRGGBB"),
    (lambda c: c.update({"section_gap_px": "big"}), "integer"),
    (lambda c: c.update({"section_gap_px": -1}), "integer"),
    (lambda c: c.update({"section_gap_px": 500}), "integer"),
    (lambda c: c["hero"].update({"headline": "x" * 200}), "exceeds"),
    (lambda c: c["brand"].update({"name": ""}), "cannot be empty"),
    (lambda c: c.update({"schema_version": 99}), "schema_version"),
    (lambda c: c["hero"].update({"enabled": "yes"}), "boolean"),
])
def test_invalid_configs_are_rejected(mutate, expected):
    cfg = ui_design.default_config()
    mutate(cfg)
    ok, err = ui_design.validate(cfg)
    assert not ok and expected in err


def test_save_rejects_invalid_and_keeps_previous():
    cfg = ui_design.default_config()
    assert ui_design.save(cfg)[0]
    tampered = ui_design.default_config()
    tampered["omega"] = "whoop"
    ok, _ = ui_design.save(tampered)
    assert not ok
    # The last good design is untouched.
    assert ui_design.load() == ui_design.default_config()


def test_corrupt_file_recovers_to_defaults_and_parks_file():
    with open(ui_design.CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write("this is not json{")
    cfg = ui_design.load()
    assert cfg == ui_design.default_config()
    assert os.path.exists(ui_design.CONFIG_PATH + ".corrupt")


def test_saved_design_survives_a_real_restart(tmp_path):
    """Write the design, then load it in a brand-new Python process."""
    cfg = ui_design.default_config()
    cfg["brand"]["name"] = "PROTO"
    ok, _ = ui_design.save(cfg)
    assert ok

    env = dict(os.environ)
    env["PACEAI_UI_DESIGN_FILE"] = str(tmp_path / "ui_design.json")
    code = (
        "import src.ui_design as u; "
        "print(u.load()['brand']['name'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "PROTO" in out.stdout


def test_renderer_applies_config_to_css():
    cfg = ui_design.default_config()
    cfg["theme"]["accent"] = "#123456"
    cfg["section_gap_px"] = 42
    css = ui_design.css_overrides(cfg)
    assert "--pai-accent:#123456" in css
    assert "margin-top:42px" in css


def test_export_import_roundtrip_and_validation():
    cfg = ui_design.default_config()
    cfg["hero"]["subtitle"] = "round trip"
    blob = ui_design.export(cfg)
    loaded, err = ui_design.import_config(blob)
    assert not err
    assert loaded["hero"]["subtitle"] == "round trip"

    bad, err = ui_design.import_config('{"schema_version": 1, "nope": 1}')
    assert not bad and err

    not_json, err = ui_design.import_config("<not json>")
    assert not not_json and "JSON" in err


def test_reset_restores_default(tmp_path):
    cfg = ui_design.default_config()
    cfg["brand"]["name"] = "CHANGED"
    assert ui_design.save(cfg)[0]
    ok, _ = ui_design.reset_to_default()
    assert ok
    assert ui_design.load() == ui_design.default_config()


def test_analysis_values_cannot_be_set_by_design():
    """Presentation config has a closed schema: no pipeline/model keys exist."""
    assert ui_design.validate(ui_design.default_config())[0]
    cfg = ui_design.default_config()
    cfg["injury_risk"] = "high"  # any attempt adds an unknown key
    assert not ui_design.validate(cfg)[0]


def test_app_boots_and_renders_customize_ui_page(monkeypatch, tmp_path):
    """The real app boots, offers the Customize UI page, and renders every
    editor widget plus the saved-design CSS override (no click choreography, so
    it stays deterministic under AppTest)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    # Make sure the saved default (fixture-isolated tmp file) is injected.
    assert any("--pai-accent:" in m.value for m in at.markdown)
    # The shell palette tokens are shared so one design re-skins the whole app.
    assert any("--app-accent:" in m.value for m in at.markdown)
    # Clear page headings render (structured, not bare Streamlit markdown).
    assert any("pai-sec-head" in m.value for m in at.markdown)
    assert any("Delivery Kinematic Parameters" in m.value for m in at.markdown)

    at.radio(key="primary_nav").set_value("Customize UI")
    at.run()
    assert not at.exception, at.exception
    assert [t.label for t in at.text_input] == ["Product name", "Tagline", "Headline", "Subtitle"]
    assert [b.label for b in at.button] == ["Save design", "Discard edits", "Reset to default"]
    assert [c.label for c in at.color_picker] == ["Accent", "Background", "Surface", "Muted", "Line"]

    # A saved design shows up in the masthead of a fresh boot.
    edited = ui_design.default_config()
    edited["brand"]["name"] = "SAVED-BRAND"
    assert ui_design.save(edited)[0]
    at2 = AppTest.from_file("app.py", default_timeout=60)
    at2.run()
    assert not at2.exception, at2.exception
    assert any("SAVED-BRAND" in m.value for m in at2.markdown)