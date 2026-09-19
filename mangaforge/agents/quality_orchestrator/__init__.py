"""
Quality Orchestrator agent stub.

Responsibility:
    Coordinates all quality-checker agents and aggregates their results.
    Reads individual checker outputs from ``quality.stages.*``, computes
    an overall quality verdict, identifies blocking vs. non-blocking
    issues, and decides whether the project is ready for the next phase
    or needs rework. Writes to ``quality.overall_status`` and
    ``quality.reviewer_notes``.

Phase 1 status: stub only -- no logic implemented yet.
"""


def orchestrate_quality(project: dict) -> dict:
    """Run the quality orchestration pass on a MangaProject.

    Args:
        project: The full MangaProject dict.

    Returns:
        The updated project dict with aggregated quality verdict.
    """
    raise NotImplementedError("Quality Orchestrator not yet implemented")
