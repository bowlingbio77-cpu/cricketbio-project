"""
Visual Director agent stub.

Responsibility:
    Owns the visual style direction across the manga. Manages art style
    parameters (line weight, detail level, special effects), camera angles,
    lighting moods, and visual palette enforcement per setting. Reads the
    ``world`` and ``pages`` sections; writes art_style and camera_angle
    fields on panels.

Phase 1 status: stub only -- no logic implemented yet.
"""


def direct_visuals(project: dict) -> dict:
    """Apply visual direction to the panels of a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        The updated project dict with visual direction applied.
    """
    raise NotImplementedError("Visual Director not yet implemented")
