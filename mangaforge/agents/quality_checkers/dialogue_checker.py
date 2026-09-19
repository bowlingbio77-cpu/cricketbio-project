"""
Dialogue Quality Checker agent stub.

Responsibility:
    Validates dialogue content and formatting. Checks character voice
    consistency, text length vs. bubble size feasibility, emotion
    alignment, bubble-type appropriateness, and reading order within
    panels. Writes results to ``quality.stages.dialogue``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_dialogue(project: dict) -> list:
    """Run dialogue quality checks on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        A list of quality check result dicts for the dialogue stage.
    """
    raise NotImplementedError("Dialogue Quality Checker not yet implemented")
