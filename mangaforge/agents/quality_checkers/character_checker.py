"""
Character Quality Checker agent stub.

Responsibility:
    Validates character consistency and completeness. Checks that every
    character referenced in panels exists in the roster, that appearance
    fields are consistent across pages, and that relationships are
    symmetric or explicitly asymmetric. Writes results to
    ``quality.stages.character``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_characters(project: dict) -> list:
    """Run character quality checks on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        A list of quality check result dicts for the character stage.
    """
    raise NotImplementedError("Character Quality Checker not yet implemented")
