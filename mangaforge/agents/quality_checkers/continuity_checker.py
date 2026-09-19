"""
Continuity Quality Checker agent stub.

Responsibility:
    Validates cross-panel and cross-page continuity. Checks that
    characters present in a scene are consistent, that setting
    properties (time, weather) don't contradict between adjacent panels,
    and that sequential action logic holds. Writes results to
    ``quality.stages.continuity``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_continuity(project: dict) -> list:
    """Run continuity quality checks on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        A list of quality check result dicts for the continuity stage.
    """
    raise NotImplementedError("Continuity Quality Checker not yet implemented")
