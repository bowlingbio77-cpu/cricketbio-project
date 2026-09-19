"""
Acceptance tests for MangaForge schema validation.

Verifies that:
    1. The hand-authored sample project passes validation.
    2. Intentionally malformed documents are rejected with clear errors.
    3. Edge cases (empty dict, missing required fields, bad enums) are caught.
    4. load_project() works for valid files and raises on bad ones.
"""

import json
from pathlib import Path

import pytest

from mangaforge.schema import validate_project, load_project, ValidationError


SAMPLE_PATH = Path(__file__).parent / "sample_project.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_project() -> dict:
    """Return the smallest possible valid MangaProject."""
    return {
        "schema_version": "1.0",
        "metadata": {
            "project_name": "Test",
            "created_at": "2026-01-01T00:00:00Z",
            "authors": ["tester"],
        },
        "story": {
            "title": "Test Story",
            "synopsis": "A test.",
            "chapters": [],
        },
        "characters": [],
        "pages": [],
    }


# ---------------------------------------------------------------------------
# 1. Sample project acceptance test
# ---------------------------------------------------------------------------

class TestSampleProject:
    """The hand-authored sample must pass validation cleanly."""

    def test_sample_loads_without_errors(self):
        project = load_project(SAMPLE_PATH)
        assert project["schema_version"] == "1.0"
        assert project["metadata"]["project_name"] == "Echoes of Twilight"

    def test_sample_has_one_page_one_character(self):
        project = load_project(SAMPLE_PATH)
        assert len(project["pages"]) == 1
        assert len(project["characters"]) == 1
        assert project["characters"][0]["character_id"] == "kai"

    def test_sample_panels_have_dialogue_and_sfx(self):
        project = load_project(SAMPLE_PATH)
        panels = project["pages"][0]["panels"]
        assert len(panels) == 3
        # Panel 3 has dialogue
        assert len(panels[2]["dialogue"]) == 1
        assert panels[2]["dialogue"][0]["text"] == "...Who's there?"
        # Panel 1 has sound effects
        assert len(panels[0]["sound_effects"]) == 1


# ---------------------------------------------------------------------------
# 2. Malformed document rejection
# ---------------------------------------------------------------------------

class TestMalformedRejection:
    """Intentionally broken documents must be rejected."""

    def test_empty_dict_rejected(self):
        errors = validate_project({})
        paths = [e.path for e in errors]
        assert "schema_version" in paths
        assert "metadata" in paths
        assert "story" in paths
        assert "characters" in paths
        assert "pages" in paths

    def test_wrong_schema_version_rejected(self):
        doc = _minimal_project()
        doc["schema_version"] = "9.9"
        errors = validate_project(doc)
        assert any("schema_version" in e.path for e in errors)

    def test_missing_metadata_fields_rejected(self):
        doc = _minimal_project()
        doc["metadata"] = {"project_name": "X"}  # missing created_at, authors
        errors = validate_project(doc)
        paths = [e.path for e in errors]
        assert "metadata.created_at" in paths
        assert "metadata.authors" in paths

    def test_bad_character_role_rejected(self):
        doc = _minimal_project()
        doc["characters"] = [
            {
                "character_id": "x",
                "name": "X",
                "role": "superhero",  # not a valid role
            }
        ]
        errors = validate_project(doc)
        assert any("role" in e.path for e in errors)

    def test_bad_panel_camera_angle_rejected(self):
        doc = _minimal_project()
        doc["pages"] = [
            {
                "page_id": "p1",
                "chapter_id": "ch1",
                "panels": [
                    {
                        "panel_id": "p1_1",
                        "layout": {"x": 0, "y": 0, "width": 100, "height": 100},
                        "scene_description": "test",
                        "camera_angle": "selfie",  # invalid
                    }
                ],
            }
        ]
        errors = validate_project(doc)
        assert any("camera_angle" in e.path for e in errors)

    def test_bad_dialogue_bubble_type_rejected(self):
        doc = _minimal_project()
        doc["pages"] = [
            {
                "page_id": "p1",
                "chapter_id": "ch1",
                "panels": [
                    {
                        "panel_id": "p1_1",
                        "layout": {"x": 0, "y": 0, "width": 100, "height": 100},
                        "scene_description": "test",
                        "dialogue": [
                            {
                                "character_id": "x",
                                "text": "hello",
                                "bubble_type": "scream",  # invalid
                            }
                        ],
                    }
                ],
            }
        ]
        errors = validate_project(doc)
        assert any("bubble_type" in e.path for e in errors)

    def test_wrong_type_for_panel_layout_rejected(self):
        doc = _minimal_project()
        doc["pages"] = [
            {
                "page_id": "p1",
                "chapter_id": "ch1",
                "panels": [
                    {
                        "panel_id": "p1_1",
                        "layout": "bad",  # should be dict
                        "scene_description": "test",
                    }
                ],
            }
        ]
        errors = validate_project(doc)
        assert any("layout" in e.path and "dict" in e.message for e in errors)

    def test_wrong_type_for_characters_rejected(self):
        doc = _minimal_project()
        doc["characters"] = "not a list"
        errors = validate_project(doc)
        assert any("characters" in e.path for e in errors)


# ---------------------------------------------------------------------------
# 3. Valid minimal project
# ---------------------------------------------------------------------------

class TestMinimalProject:
    """A bare-bones project with only required fields must pass."""

    def test_minimal_project_valid(self):
        errors = validate_project(_minimal_project())
        assert errors == []

    def test_minimal_project_loads(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps(_minimal_project()), encoding="utf-8")
        project = load_project(path)
        assert project["metadata"]["project_name"] == "Test"


# ---------------------------------------------------------------------------
# 4. load_project edge cases
# ---------------------------------------------------------------------------

class TestLoadProject:
    """load_project() file-level error handling."""

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_project("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("NOT JSON {{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_project(path)

    def test_valid_json_but_invalid_schema_raises(self, tmp_path):
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid MangaProject"):
            load_project(path)


# ---------------------------------------------------------------------------
# 5. Agent stub importability
# ---------------------------------------------------------------------------

class TestAgentStubs:
    """All agent stub modules should be importable and raise NotImplementedError."""

    def test_story_director(self):
        from mangaforge.agents.story_director import generate_story
        with pytest.raises(NotImplementedError):
            generate_story({})

    def test_character_director(self):
        from mangaforge.agents.character_director import manage_characters
        with pytest.raises(NotImplementedError):
            manage_characters({})

    def test_world_director(self):
        from mangaforge.agents.world_director import manage_world
        with pytest.raises(NotImplementedError):
            manage_world({})

    def test_manga_director(self):
        from mangaforge.agents.manga_director import layout_pages
        with pytest.raises(NotImplementedError):
            layout_pages({})

    def test_visual_director(self):
        from mangaforge.agents.visual_director import direct_visuals
        with pytest.raises(NotImplementedError):
            direct_visuals({})

    def test_continuity_engine(self):
        from mangaforge.agents.continuity_engine import check_continuity
        with pytest.raises(NotImplementedError):
            check_continuity({})

    def test_dialogue_director(self):
        from mangaforge.agents.dialogue_director import direct_dialogue
        with pytest.raises(NotImplementedError):
            direct_dialogue({})

    def test_page_composer(self):
        from mangaforge.agents.page_composer import compose_pages
        with pytest.raises(NotImplementedError):
            compose_pages({})

    def test_quality_orchestrator(self):
        from mangaforge.agents.quality_orchestrator import orchestrate_quality
        with pytest.raises(NotImplementedError):
            orchestrate_quality({})

    def test_orchestrator(self):
        from mangaforge.agents.orchestrator import run_pipeline
        with pytest.raises(NotImplementedError):
            run_pipeline({})

    def test_story_checker(self):
        from mangaforge.agents.quality_checkers.story_checker import check_story
        with pytest.raises(NotImplementedError):
            check_story({})

    def test_character_checker(self):
        from mangaforge.agents.quality_checkers.character_checker import check_characters
        with pytest.raises(NotImplementedError):
            check_characters({})

    def test_color_checker(self):
        from mangaforge.agents.quality_checkers.color_checker import check_color
        with pytest.raises(NotImplementedError):
            check_color({})

    def test_composition_checker(self):
        from mangaforge.agents.quality_checkers.composition_checker import check_composition
        with pytest.raises(NotImplementedError):
            check_composition({})

    def test_dialogue_checker(self):
        from mangaforge.agents.quality_checkers.dialogue_checker import check_dialogue
        with pytest.raises(NotImplementedError):
            check_dialogue({})

    def test_continuity_checker(self):
        from mangaforge.agents.quality_checkers.continuity_checker import check_continuity
        with pytest.raises(NotImplementedError):
            check_continuity({})
