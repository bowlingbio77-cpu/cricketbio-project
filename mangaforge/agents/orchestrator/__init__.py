"""
Top-level Orchestrator agent stub.

Responsibility:
    Entry point for the full MangaForge pipeline. Reads the current
    MangaProject state, determines which agents need to run based on
    what has changed and what is stale, dispatches work to each agent
    in dependency order, and writes the final updated project. Manages
    the overall generation loop: draft -> quality check -> rework -> approve.

Phase 1 status: stub only -- no logic implemented yet.
"""

# Agent execution order (dependency-aware):
# 1. Story Director       -- narrative foundation
# 2. Character Director   -- character roster
# 3. World Director       -- settings and world rules
# 4. Manga Director       -- page/panel layout
# 5. Visual Director      -- art style and camera direction
# 6. Dialogue Director    -- dialogue and SFX content
# 7. Page Composer        -- final page assembly
# 8. Continuity Engine    -- cross-cutting consistency
# 9. Quality Checkers     -- story, character, color, composition, dialogue, continuity
# 10. Quality Orchestrator -- aggregate verdict


def run_pipeline(project: dict) -> dict:
    """Run the full MangaForge generation pipeline.

    Args:
        project: The full MangaProject dict.

    Returns:
        The updated project dict after all agents have run.
    """
    raise NotImplementedError("Orchestrator not yet implemented")
