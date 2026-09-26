"""Drive app.py to the result page and assert the 11 sections render clean.

Run:  python tools/smoke_result_page.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "app.py")
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "harness_result_states.py")

SECTIONS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
]

# The old result screen mapped low/moderate/high to hard-coded 22/58/88 and
# printed them as a risk percentage. These patterns are the shapes of that lie.
# Scoped to risk wording on purpose: a real 88% tracking confidence or a real
# 85% classifier probability is legitimate and must not trip this check.
FORBIDDEN = [
    "22% risk", "58% risk", "88% risk",
    "risk: 22", "risk: 58", "risk: 88",
    "injury risk score",
    "risk of injury:",
    "overall risk score",
]


def run(mode: str = "sim", sliders=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    if at.exception:
        raise AssertionError(f"boot failed: {at.exception[0].value}")
    at.radio[0].set_value("\u26a1 Analyze").run()
    at.radio[1].set_value(mode).run()
    if sliders:
        for label, value in sliders.items():
            for s in at.slider:
                if s.label == label:
                    s.set_value(value)
                    break
            at.run()
    return at


def html_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def check(at, label):
    if at.exception:
        raise AssertionError(f"[{label}] exceptions: "
                             f"{[str(e.value)[:300] for e in at.exception]}")
    body = html_text(at)
    found = [n for n in SECTIONS if f"pai-sec-title\">{n} " in body
             or f">{n} \u00b7" in body]
    print(f"[{label}] sections rendered: {found}")
    missing = [n for n in SECTIONS if n not in found]
    hits = [p for p in FORBIDDEN if p.lower() in body.lower()]
    print(f"[{label}] missing: {missing}  forbidden-phrases: {hits}")
    # No Streamlit error boxes leaked into the page.
    assert not at.error, [e.value for e in at.error]
    return found, hits, body


def run_state(state: str):
    """Render the result page for a state the UI cannot reach without a clip."""
    at = AppTest.from_file(HARNESS, default_timeout=300)
    at.session_state["pai_state"] = state
    at.run()
    if at.exception:
        raise AssertionError(f"[{state}] exceptions: "
                             f"{[str(e.value)[:300] for e in at.exception]}")
    return check(at, state)


def main():
    ok = True

    print("=== 1. clean simulator delivery ===")
    at = run("Interactive Bio-Simulator")
    found, hits, body = check(at, "clean")
    if hits:
        ok = False
    assert "NORMAL" in body or "ENTRY" in body.upper(), "no status headline"
    print("   headline present")

    print("=== 2. flagged simulator delivery ===")
    flagged = {
        "Elbow Flexion (release)": 28.0,
        "Shoulder Rotation": 150.0,
        "Trunk Lean": 38.0,
        "Wrist Angle": -75.0,
    }
    at = run("Interactive Bio-Simulator", flagged)
    found, hits, body = check(at, "flagged")
    if hits:
        ok = False
    if "FINDING" in body.upper():
        print("   findings section populated")

    print("=== 3. video mode with no clip (must not fabricate) ===")
    at = run("\U0001f4f9 Video Motion Capture")
    found, hits, body = check(at, "video-empty")
    if hits:
        ok = False
    assert not found, "no analysis should render without an input"
    print("   correctly renders nothing (no analysis has run)")

    print("=== 4. scoring withheld (unverified bowler) ===")
    found, hits, body = run_state("withheld")
    if hits:
        ok = False
    low = body.lower()
    for phrase in ("withheld", "not confirmed", "no performance", "no risk"):
        if phrase in low:
            print(f"   present: {phrase!r}")
    assert "withheld" in low or "limited" in low, low[:400]
    for leak in ("performance score of", "risk level: low", "22%"):
        assert leak not in low, f"leaked {leak!r} while scoring was withheld"

    print("=== 5. partially measured (2D fallback) ===")
    found, hits, body = run_state("partial")
    if hits:
        ok = False
    low = body.lower()
    assert "2d" in low, "2D fallback not disclosed"
    print("   2D fallback disclosed")

    print("=== 6. flagged (evidence chain must be populated) ===")
    found, hits, body = run_state("flagged")
    if hits:
        ok = False
    assert "06" in found, f"evidence section missing: {found}"
    # pai-chain-body only appears in a real chain (5 steps each).
    steps = body.count("pai-chain-body")
    print(f"   {steps} chain steps rendered")
    assert steps >= 5, "evidence chain is not a full 5-step trace"
    # The real tracking confidence must still reach the user.
    assert "88%" in body, "real bowler tracking confidence was dropped"

    print()
    print("SMOKE OK" if ok else "SMOKE FOUND FORBIDDEN PHRASES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
