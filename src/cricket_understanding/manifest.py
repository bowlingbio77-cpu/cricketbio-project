"""
Phase 6 -- ``dataset_manifest.json``.

A machine-readable index of every registered clip, its annotation/review status,
its split, and its annotation counts.  Statuses are **derived from what is on
disk**, never asserted:

``annotation_status``
    NONE       no annotation files
    PARTIAL    some annotation kinds present, QC not run / not passing
    COMPLETE   all expected kinds present and no ERROR-severity QC issues
    QC_FAILED  QC ran and produced at least one ERROR
    QC_PASSED  QC ran, zero ERRORs

``review_status``
    UNREVIEWED / IN_PROGRESS / REVIEWED / REJECTED -- human-set only.  The
    manifest never promotes a clip to REVIEWED on its own; that requires a
    human note naming the reviewer (enforced in ``registry.VideoRecord``).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from . import qc, schema, splits as splits_mod
from .registry import VideoRecord

MANIFEST_PATH = schema.p("dataset_manifest.json")
MANIFEST_VERSION = "cricket_understanding_manifest_v1"


def _annotation_counts(vid: str) -> Dict[str, int]:
    return {
        "person_roles": len(schema.load_person_annotations(vid)),
        "bowling_phases": len(schema.load_phase_annotations(vid)),
        "deliveries": len(schema.load_deliveries(vid)),
        "objects": len(schema.load_object_annotations(vid)),
        "scene": 1 if schema.load_scene(vid) else 0,
    }


def _has_any_annotation(counts: Dict[str, int]) -> bool:
    return any(v > 0 for v in counts.values())


def _is_complete(counts: Dict[str, int], has_deliveries_required: bool) -> bool:
    """COMPLETE means: person roles AND phase labels AND a scene record; a
    delivery is additionally required when the clip declares deliveries."""
    if counts["person_roles"] == 0 or counts["bowling_phases"] == 0:
        return False
    if counts["scene"] == 0:
        return False
    if has_deliveries_required and counts["deliveries"] == 0:
        return False
    return True


def build(records: Optional[List[VideoRecord]] = None,
          qc_result: Optional[Dict[str, Any]] = None,
          splits: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    records = records if records is not None else []
    splits = splits if splits is not None else (splits_mod.load_splits() or {})

    if qc_result is None:
        n_frames = {r.video_id: r.frame_count for r in records if r.frame_count}
        qc_result = qc.run_all(
            video_ids=[r.video_id for r in records] or None,
            n_frames_by_id=n_frames, splits=splits or None, records=records)
    per_video_issues = {v["video_id"]: v.get("issues", []) for v in qc_result["per_video"]}

    videos: List[Dict[str, Any]] = []
    totals = defaultdict(int)

    for r in sorted(records, key=lambda r: r.video_id):
        counts = _annotation_counts(r.video_id)
        issues = per_video_issues.get(r.video_id, [])
        n_errors = sum(1 for i in issues if i.get("severity") == "ERROR")
        n_warnings = sum(1 for i in issues if i.get("severity") == "WARNING")

        # Which split owns this video (a video must be in exactly one).
        split_name = next((s for s, ids in splits.items() if r.video_id in ids), "UNASSIGNED")

        if n_errors:
            status = "QC_FAILED"
        elif not _has_any_annotation(counts):
            status = "NONE"
        elif qc_result.get("per_video") and r.video_id in per_video_issues:
            status = "QC_PASSED" if _is_complete(counts, bool(counts["deliveries"])) else "PARTIAL"
        else:
            status = "PARTIAL" if _has_any_annotation(counts) else "NONE"

        if r.number_of_deliveries is None and counts["deliveries"]:
            r.number_of_deliveries = counts["deliveries"]

        entry = {
            "video_id": r.video_id,
            "filename": r.filename,
            "path": r.path,
            "bytes": r.bytes,
            "fps": r.fps,
            "width": r.width,
            "height": r.height,
            "frame_count": r.frame_count,
            "duration_sec": r.duration_sec,
            "probe_status": r.probe_status,
            "camera_view": r.camera_view,
            "bowling_side_if_known": r.bowling_side_if_known,
            "bowler_id_if_known": r.bowler_id_if_known or None,
            "number_of_deliveries": r.number_of_deliveries,
            "annotation_status": status,
            "review_status": r.review_status,
            "split": split_name,
            "annotation_counts": counts,
            "qc_error_count": n_errors,
            "qc_warning_count": n_warnings,
            "notes": r.notes,
        }
        videos.append(entry)

        totals["videos"] += 1
        totals[f"status_{status}"] += 1
        totals[f"review_{r.review_status}"] += 1
        totals[f"split_{split_name}"] += 1
        totals["person_roles"] += counts["person_roles"]
        totals["bowling_phases"] += counts["bowling_phases"]
        totals["deliveries"] += counts["deliveries"]
        totals["objects"] += counts["objects"]
        totals["scene"] += counts["scene"]

    n_annotated = sum(1 for v in videos if v["annotation_status"] != "NONE")
    ready = [v for v in videos
             if v["annotation_status"] == "QC_PASSED"
             and v["split"] in ("train", "val", "test")]

    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "totals": dict(sorted(totals.items())),
        "n_videos_registered": len(videos),
        "n_videos_annotated": n_annotated,
        "n_videos_trainable": len(ready),
        "deliveries_annotated": totals["deliveries"],
        "person_role_annotations": totals["person_roles"],
        "bowling_phase_annotations": totals["bowling_phases"],
        "splits": {k: len(v) for k, v in splits.items()},
        "videos": videos,
    }


def save(manifest: Dict[str, Any], path: str = MANIFEST_PATH) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema.dumps(manifest))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load(path: str = MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
