"""
MangaProject schema definition and validation.

Defines the canonical structure for a manga project document and provides
a validator that can read a JSON file and confirm it conforms to the schema.
No external dependencies -- validation is hand-rolled against a declarative
schema dict so the project stays dependency-light at this foundation phase.

Usage::

    from mangaforge.schema import validate_project, load_project

    project = load_project("my_manga.json")          # loads + validates
    errors = validate_project(raw_dict)               # returns [] if valid
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema definition (declarative dict -- each key describes a section)
# ---------------------------------------------------------------------------

_QUALITY_CHECK_SPEC: Dict[str, Any] = {
    "type": "list",
    "element": {
        "type": "dict",
        "required": ["check_name", "status"],
        "properties": {
            "check_name": {"type": "str"},
            "status": {
                "type": "str",
                "values": ["pass", "fail", "warning", "pending"],
            },
            "score": {"type": "number"},
            "message": {"type": "str"},
        },
    },
}

SCHEMA: Dict[str, Any] = {
    "required": ["schema_version", "metadata", "story", "characters", "pages"],
    "properties": {
        "schema_version": {"type": "str", "values": ["1.0"]},
        "metadata": {
            "type": "dict",
            "required": ["project_name", "created_at", "authors"],
            "properties": {
                "project_name": {"type": "str"},
                "created_at": {"type": "str"},
                "updated_at": {"type": "str"},
                "authors": {"type": "list", "element": {"type": "str"}},
                "genre": {"type": "str"},
                "target_audience": {"type": "str"},
                "total_pages_planned": {"type": "int"},
                "notes": {"type": "str"},
            },
        },
        "story": {
            "type": "dict",
            "required": ["title", "synopsis", "chapters"],
            "properties": {
                "title": {"type": "str"},
                "synopsis": {"type": "str"},
                "genre_tags": {"type": "list", "element": {"type": "str"}},
                "chapters": {
                    "type": "list",
                    "element": {
                        "type": "dict",
                        "required": ["chapter_id", "title", "summary"],
                        "properties": {
                            "chapter_id": {"type": "str"},
                            "title": {"type": "str"},
                            "summary": {"type": "str"},
                            "scene_breakdown": {
                                "type": "list",
                                "element": {
                                    "type": "dict",
                                    "required": ["scene_id", "description"],
                                    "properties": {
                                        "scene_id": {"type": "str"},
                                        "description": {"type": "str"},
                                        "setting_id": {"type": "str"},
                                        "characters_present": {
                                            "type": "list",
                                            "element": {"type": "str"},
                                        },
                                        "mood": {"type": "str"},
                                        "page_range": {
                                            "type": "dict",
                                            "properties": {
                                                "start": {"type": "int"},
                                                "end": {"type": "int"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "characters": {
            "type": "list",
            "element": {
                "type": "dict",
                "required": ["character_id", "name", "role"],
                "properties": {
                    "character_id": {"type": "str"},
                    "name": {"type": "str"},
                    "aliases": {"type": "list", "element": {"type": "str"}},
                    "role": {
                        "type": "str",
                        "values": [
                            "protagonist",
                            "antagonist",
                            "supporting",
                            "minor",
                            "background",
                        ],
                    },
                    "description": {"type": "str"},
                    "appearance": {
                        "type": "dict",
                        "properties": {
                            "age": {"type": "str"},
                            "gender": {"type": "str"},
                            "height": {"type": "str"},
                            "build": {"type": "str"},
                            "hair_color": {"type": "str"},
                            "hair_style": {"type": "str"},
                            "eye_color": {"type": "str"},
                            "skin_tone": {"type": "str"},
                            "distinguishing_marks": {
                                "type": "list",
                                "element": {"type": "str"},
                            },
                            "default_outfit": {"type": "str"},
                            "reference_image": {"type": "str"},
                        },
                    },
                    "personality": {
                        "type": "dict",
                        "properties": {
                            "traits": {"type": "list", "element": {"type": "str"}},
                            "motivation": {"type": "str"},
                            "backstory": {"type": "str"},
                        },
                    },
                    "relationships": {
                        "type": "list",
                        "element": {
                            "type": "dict",
                            "required": ["target_character_id", "relation"],
                            "properties": {
                                "target_character_id": {"type": "str"},
                                "relation": {"type": "str"},
                                "notes": {"type": "str"},
                            },
                        },
                    },
                },
            },
        },
        "world": {
            "type": "dict",
            "required": ["settings"],
            "properties": {
                "era": {"type": "str"},
                "technology_level": {"type": "str"},
                "rules": {"type": "list", "element": {"type": "str"}},
                "settings": {
                    "type": "list",
                    "element": {
                        "type": "dict",
                        "required": ["setting_id", "name"],
                        "properties": {
                            "setting_id": {"type": "str"},
                            "name": {"type": "str"},
                            "description": {"type": "str"},
                            "time_of_day": {"type": "str"},
                            "weather": {"type": "str"},
                            "visual_palette": {
                                "type": "dict",
                                "properties": {
                                    "primary_colors": {
                                        "type": "list",
                                        "element": {"type": "str"},
                                    },
                                    "mood": {"type": "str"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "pages": {
            "type": "list",
            "element": {
                "type": "dict",
                "required": ["page_id", "chapter_id", "panels"],
                "properties": {
                    "page_id": {"type": "str"},
                    "chapter_id": {"type": "str"},
                    "page_number": {"type": "int"},
                    "notes": {"type": "str"},
                    "panels": {
                        "type": "list",
                        "element": {
                            "type": "dict",
                            "required": ["panel_id", "layout", "scene_description"],
                            "properties": {
                                "panel_id": {"type": "str"},
                                "layout": {
                                    "type": "dict",
                                    "required": ["x", "y", "width", "height"],
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "width": {"type": "number"},
                                        "height": {"type": "number"},
                                    },
                                },
                                "scene_description": {"type": "str"},
                                "characters_present": {
                                    "type": "list",
                                    "element": {"type": "str"},
                                },
                                "art_style": {
                                    "type": "dict",
                                    "properties": {
                                        "line_weight": {"type": "str"},
                                        "detail_level": {
                                            "type": "str",
                                            "values": ["minimal", "standard", "detailed"],
                                        },
                                        "special_effects": {
                                            "type": "list",
                                            "element": {"type": "str"},
                                        },
                                    },
                                },
                                "camera_angle": {
                                    "type": "str",
                                    "values": [
                                        "eye_level",
                                        "low_angle",
                                        "high_angle",
                                        "birdseye",
                                        "dutch_angle",
                                        "close_up",
                                        "extreme_close_up",
                                        "wide_shot",
                                        "medium_shot",
                                    ],
                                },
                                "sound_effects": {
                                    "type": "list",
                                    "element": {
                                        "type": "dict",
                                        "required": ["text"],
                                        "properties": {
                                            "text": {"type": "str"},
                                            "style": {"type": "str"},
                                            "position": {"type": "str"},
                                        },
                                    },
                                },
                                "dialogue": {
                                    "type": "list",
                                    "element": {
                                        "type": "dict",
                                        "required": ["character_id", "text"],
                                        "properties": {
                                            "character_id": {"type": "str"},
                                            "text": {"type": "str"},
                                            "emotion": {"type": "str"},
                                            "position": {
                                                "type": "dict",
                                                "properties": {
                                                    "x": {"type": "number"},
                                                    "y": {"type": "number"},
                                                },
                                            },
                                            "bubble_type": {
                                                "type": "str",
                                                "values": [
                                                    "speech",
                                                    "thought",
                                                    "narration",
                                                    "shout",
                                                    "whisper",
                                                ],
                                            },
                                            "font_style": {"type": "str"},
                                        },
                                    },
                                },
                                "transitions": {
                                    "type": "dict",
                                    "properties": {
                                        "enter": {"type": "str"},
                                        "exit": {"type": "str"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "quality": {
            "type": "dict",
            "properties": {
                "stages": {
                    "type": "dict",
                    "properties": {
                        "story": _QUALITY_CHECK_SPEC,
                        "character": _QUALITY_CHECK_SPEC,
                        "color": _QUALITY_CHECK_SPEC,
                        "composition": _QUALITY_CHECK_SPEC,
                        "dialogue": _QUALITY_CHECK_SPEC,
                        "continuity": _QUALITY_CHECK_SPEC,
                    },
                },
                "overall_status": {
                    "type": "str",
                    "values": ["draft", "in_progress", "review", "approved", "rejected"],
                },
                "reviewer_notes": {"type": "str"},
                "last_checked_at": {"type": "str"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationError:
    """A single validation error with a path and message."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.path!r}, {self.message!r})"

    def __str__(self) -> str:
        return f"[{self.path}] {self.message}"


def _check_type(value: Any, type_name: str, path: str) -> Optional[str]:
    """Return an error string if *value* is not the expected type, else None."""
    type_map = {
        "str": str,
        "int": int,
        "number": (int, float),
        "bool": bool,
        "list": list,
        "dict": dict,
    }
    expected = type_map.get(type_name)
    if expected is None:
        return f"{path}: unknown type {type_name!r}"
    if not isinstance(value, expected):
        return f"{path}: expected {type_name}, got {type(value).__name__}"
    return None


def _validate_node(
    node: Any,
    spec: Dict[str, Any],
    path: str,
    errors: List[ValidationError],
) -> None:
    """Recursively validate *node* against *spec*, appending errors."""

    # Type check
    if "type" in spec:
        err = _check_type(node, spec["type"], path)
        if err:
            errors.append(ValidationError(path, err))
            return

    # Enum / allowed-values check
    if "values" in spec and node not in spec["values"]:
        errors.append(
            ValidationError(
                path,
                f"got {node!r}, expected one of {spec['values']}",
            )
        )

    # Dict validation
    if spec.get("type") == "dict" and isinstance(node, dict):
        for key in spec.get("required", []):
            if key not in node:
                errors.append(ValidationError(f"{path}.{key}", "required field missing"))
        for key, child_spec in spec.get("properties", {}).items():
            if key in node:
                _validate_node(node[key], child_spec, f"{path}.{key}", errors)

    # List validation
    if spec.get("type") == "list" and isinstance(node, list):
        elem_spec = spec.get("element", {})
        for i, item in enumerate(node):
            _validate_node(item, elem_spec, f"{path}[{i}]", errors)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_project(data: Dict[str, Any]) -> List[ValidationError]:
    """Validate a MangaProject dict against the schema.

    Returns a list of :class:`ValidationError` objects.  An empty list means
    the document is valid.
    """
    errors: List[ValidationError] = []

    # Top-level required keys
    for key in SCHEMA.get("required", []):
        if key not in data:
            errors.append(ValidationError(key, "required field missing"))

    # Validate each top-level section
    for key, spec in SCHEMA.get("properties", {}).items():
        if key in data:
            _validate_node(data[key], spec, key, errors)

    return errors


def load_project(path: str | Path) -> Dict[str, Any]:
    """Load a MangaProject JSON file, validate it, and return the dict.

    Raises ``ValueError`` if the file contains a malformed project.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    errors = validate_project(data)
    if errors:
        error_report = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"Invalid MangaProject in {path}:\n{error_report}"
        )

    return data
