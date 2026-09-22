"""Tests for the "Players Detected" broadcast roster rendered in the app UI.

The roster is the presentation layer for the pipeline's player-role output:
it lists the locked bowler plus every classified non-bowler player with its
cricket-evidence confidence, using colors that match the Analysis Replay
overlays. These tests cover the HTML contract (structure, ordering, colors,
confidence formatting, honest-empty behavior).
"""
from src import analysis_ui
from src.tracking import (ROLE_BATSMAN, ROLE_WICKETKEEPER, ROLE_UMPIRE,
                          ROLE_FIELDER)


class TestRoleRoster:

    def test_empty_without_data(self):
        assert analysis_ui.render_role_roster(None, None, None) == ""
        assert analysis_ui.render_role_roster({}, None, None) == ""

    def test_bowler_only_shows_one_row(self):
        html = analysis_ui.render_role_roster({}, bowler_track_id=3,
                                              bowler_confidence=0.74)
        assert html != ""
        assert "BOWLER" in html
        assert "#3" in html
        assert "74%" in html
        assert html.count('<div class="paceai-roster-row') == 1

    def test_bowler_is_first_and_prominent(self):
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 0.90, "scores": {}},
        }
        html = analysis_ui.render_role_roster(player_roles,
                                              bowler_track_id=1,
                                              bowler_confidence=0.74)
        bowler_idx = html.find("BOWLER")
        batsman_idx = html.find("BATSMAN")
        assert bowler_idx != -1 and batsman_idx != -1
        assert bowler_idx < batsman_idx
        assert "bowler-row" in html

    def test_roles_sorted_by_confidence_descending(self):
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 0.90, "scores": {}},
            3: {"role": ROLE_WICKETKEEPER, "confidence": 0.94, "scores": {}},
            4: {"role": ROLE_UMPIRE, "confidence": 0.92, "scores": {}},
            5: {"role": ROLE_FIELDER, "confidence": 0.40, "scores": {}},
        }
        html = analysis_ui.render_role_roster(player_roles, None, None)
        # Sorted by confidence descending: keeper 0.94, umpire 0.92, batsman 0.90, fielder 0.40
        call_orders = [html.find(label) for label in
                       ("WICKETKEEPER", "UMPIRE", "BATSMAN", "FIELDER")]
        assert -1 not in call_orders
        assert call_orders == sorted(call_orders)

    def test_reference_image_scenario(self):
        # The exact detection set from the reference broadcast overlay.
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 0.90, "scores": {}},
            3: {"role": ROLE_WICKETKEEPER, "confidence": 0.94, "scores": {}},
            4: {"role": ROLE_UMPIRE, "confidence": 0.92, "scores": {}},
        }
        html = analysis_ui.render_role_roster(player_roles,
                                              bowler_track_id=1,
                                              bowler_confidence=0.74)
        for needle in ("BOWLER", "0.74" if False else "74%",
                       "BATSMAN", "90%", "WICKETKEEPER", "94%",
                       "UMPIRE", "92%"):
            assert needle in html

    def test_confidence_clipped_and_missing_shown_as_n_a(self):
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 1.7, "scores": {}},
            3: {"role": ROLE_UMPIRE, "confidence": None, "scores": {}},
        }
        html = analysis_ui.render_role_roster(player_roles, None, None)
        assert "100%" in html  # 1.7 clipped to 100
        assert "n/a" in html

    def test_colors_match_broadcast_palette(self):
        player_roles = {
            2: {"role": ROLE_BATSMAN, "confidence": 0.90, "scores": {}},
        }
        html = analysis_ui.render_role_roster(player_roles,
                                              bowler_track_id=1,
                                              bowler_confidence=0.74)
        # bowler orange + batsman red (hex values from the roster palette)
        assert "#ffc861" in html
        assert "#ff4d4d" in html

    def test_malformed_role_info_does_not_crash(self):
        player_roles = {
            2: "not-a-dict",
            3: {"role": ROLE_UMPIRE},  # missing confidence
            4: {"confidence": 0.5},    # missing role -> unknown
        }
        html = analysis_ui.render_role_roster(player_roles, None, None)
        assert html != ""
        assert "UNKNOWN" in html