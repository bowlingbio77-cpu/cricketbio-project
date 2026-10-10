"""
Cricket Understanding v1 -- training-data and annotation subsystem.

Scope
-----
This package owns the **cricket-specific** learning layer: the label schema, the
human annotation session, dataset quality control, leakage-safe splitting,
deterministic training-data generation, pose-sequence export, bowler-lock
integration and confidence separation.

It is intentionally additive. The existing heuristic pipeline
(``src/pipeline.py``, ``src/tracking.py``, ``src/pose_estimation.py``) keeps
working unchanged; the learned models are wired in behind
:mod:`src.cricket_understanding.bowler_lock`, which is inert until a validated
model artifact is supplied.

Import policy
-------------
Nothing in this package imports ``torch``, ``ultralytics`` or ``catboost`` at
module scope, so annotation, QC, splitting and generation all run in a bare
Python environment.  Heavy dependencies are imported lazily and only where
strictly required.
"""
from . import schema  # noqa: F401  (re-exported for convenience)

__all__ = ["schema"]
__version__ = "1.0.0"
SCHEMA_VERSION = schema.SCHEMA_VERSION
