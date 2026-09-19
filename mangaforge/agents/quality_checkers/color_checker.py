"""
Color Quality Checker agent stub.

Responsibility:
    Validates color palette consistency across panels and settings.
    Checks that each setting's visual palette is applied consistently,
    that character colors match their reference definitions, and that
    mood-color alignment is maintained. Writes results to
    ``quality.stages.color``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_color(project: dict) -> list:
    """Run color quality checks on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        A list of quality check result dicts for the color stage.
    """
    raise NotImplementedError("Color Quality Checker not yet implemented")
