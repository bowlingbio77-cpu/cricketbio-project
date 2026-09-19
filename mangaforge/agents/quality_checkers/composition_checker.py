"""
Composition Quality Checker agent stub.

Responsibility:
    Validates visual composition of panels. Checks panel layout
    proportions, reading flow (right-to-left for manga), camera angle
    appropriateness, text-to-art balance, and gutter consistency.
    Writes results to ``quality.stages.composition``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_composition(project: dict) -> list:
    """Run composition quality checks on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        A list of quality check result dicts for the composition stage.
    """
    raise NotImplementedError("Composition Quality Checker not yet implemented")
