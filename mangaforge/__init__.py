"""
MangaForge -- AI manga-page generation pipeline.

This package provides the shared state substrate and folder architecture
for a multi-agent manga generation system. Each sub-package corresponds
to a specialist agent (Story Director, Character Director, etc.) that
reads and writes a shared MangaProject JSON document.

Phase 1 deliverables:
    - MangaProject JSON schema (schema.py)
    - Folder architecture for all agents (stub modules)
    - Loader / validator for project files
"""

__version__ = "0.1.0"
