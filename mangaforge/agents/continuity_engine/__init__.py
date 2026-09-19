"""
Continuity Engine agent stub.

Responsibility:
    Cross-references the entire MangaProject to detect continuity errors.
    Checks character appearance consistency (hair, outfits, marks) across
    pages, setting consistency (time of day, weather, palette), character
    presence tracking, and sequential panel logic. Writes findings to the
    ``quality.stages.continuity`` section.

Phase 1 status: stub only -- no logic implemented yet.
"""


def check_continuity(project: dict) -> dict:
    """Run continuity checks on a MangaProject and record findings.

    Args:
        project: The full MangaProject dict.

    Returns:
        The updated project dict with continuity quality checks populated.
    """
    raise NotImplementedError("Continuity Engine not yet implemented")
