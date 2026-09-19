"""
Dialogue Director agent stub.

Responsibility:
    Owns all dialogue and sound-effect content within panels. Reviews
    dialogue for tone consistency, character voice distinctiveness,
    pacing (text-to-art ratio), bubble placement, and readability.
    Writes findings to ``quality.stages.dialogue`` and updates dialogue
    entries in ``pages[].panels[].dialogue``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def direct_dialogue(project: dict) -> dict:
    """Review and refine dialogue across all panels of a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        The updated project dict with dialogue refinements applied.
    """
    raise NotImplementedError("Dialogue Director not yet implemented")
