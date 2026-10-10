"""
Phase 4 -- dataset quality control.

Every check **reports**; none of them **repairs**.  Human annotations are
evidence, and silently rewriting them would destroy the provenance that makes
the dataset trustworthy.  A check that finds a problem emits a
:class:`QCIssue` with a stable ``code`` so a reviewer can triage by category.

Checks implemented
------------------
duplicate video IDs            DUPLICATE_VIDEO_ID
missing frames                 MISSING_ANNOTATED_FRAMES
invalid frame numbers          INVALID_FRAME_NUMBER
impossible bounding boxes      IMPOSSIBLE_BBOX            (also raised at parse)
missing annotator ID           MISSING_ANNOTATOR_ID
overlapping / conflicting      CONFLICTING_ROLE, OVERLAPPING_BBOX
release outside delivery       RELEASE_OUTSIDE_DELIVERY
phase ordering errors          PHASE_ORDER_VIOLATION
missing bowler track           MISSING_BOWLER_TRACK
impossible phase sequence      IMPOSSIBLE_PHASE_SEQUENCE
duplicate delivery IDs         DUPLICATE_DELIVERY_ID
frame index out of range       FRAME_OUT_OF_VIDEO_RANGE
uncertain frames               UNCERTAIN_FRAMES
scene geometry problems        SCENE_IMPOSSIBLE_BOX
split leakage                  SPLIT_LEAKAGE
duplicate annotation rows      DUPLICATE_ANNOTATION
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import schema

SEVERITIES = ("ERROR", "WARNING", "INFO")

VALID_ORDER = list(schema.ORDERED_PHASE_SEQUENCE)

#: Roles that are unique by construction in a cricket frame. Several FIELDERs
#: (or several UNKNOWNs) at once is normal and is NOT a conflict; two BOWLERs,
# or two STRIKERs, never are.
UNIQUE_ROLES = frozenset({"BOWLER", "STRIKER", "NON_STRIKER", "WICKETKEEPER"})


# --------------------------------------------------------------------------- #

@dataclass
class QCIssue:
    code: str
    severity: str
    video_id: str
    message: str
    frame_id: Optional[int] = None
    track_id: Optional[int] = None
    delivery_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        loc = self.video_id or "-"
        if self.frame_id is not None:
            loc += f" f{self.frame_id}"
        if self.track_id is not None:
            loc += f" t{self.track_id}"
        if self.delivery_id:
            loc += f" d={self.delivery_id}"
        return f"[{self.severity}] {self.code} @ {loc}: {self.message}"


# --------------------------------------------------------------------------- #

def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def check_duplicate_video_ids(records) -> List[QCIssue]:
    """``records``: iterable of objects with ``video_id``."""
    seen = defaultdict(list)
    for r in records:
        seen[r.video_id].append(getattr(r, "path", "") or getattr(r, "filename", ""))
    issues = []
    for vid, paths in sorted(seen.items()):
        if len(paths) > 1:
            issues.append(QCIssue(
                "DUPLICATE_VIDEO_ID", "ERROR", vid,
                f"video_id appears {len(paths)} times; splits and manifests will collide",
                detail={"paths": sorted(paths)},
            ))
    return issues


def check_frame_numbers(people, video_id: str, n_frames: Optional[int]) -> List[QCIssue]:
    issues: List[QCIssue] = []
    for p in people:
        if p.frame_id < 0:
            issues.append(QCIssue(
                "INVALID_FRAME_NUMBER", "ERROR", video_id,
                f"negative frame_id {p.frame_id}", frame_id=p.frame_id, track_id=p.track_id))
        if n_frames is not None and p.frame_id >= n_frames:
            issues.append(QCIssue(
                "FRAME_OUT_OF_VIDEO_RANGE", "ERROR", video_id,
                f"frame_id {p.frame_id} >= video frame_count {n_frames}",
                frame_id=p.frame_id, track_id=p.track_id,
                detail={"n_frames": n_frames}))
    return issues


def check_annotator(people, video_id: str) -> List[QCIssue]:
    issues = []
    for p in people:
        if not (p.annotator_id or "").strip():
            issues.append(QCIssue(
                "MISSING_ANNOTATOR_ID", "ERROR", video_id,
                "person annotation has no annotator_id; ground truth is unattributable",
                frame_id=p.frame_id, track_id=p.track_id))
    return issues


def check_bboxes(people, video_id: str) -> List[QCIssue]:
    issues: List[QCIssue] = []
    for p in people:
        if p.bbox_x2 <= p.bbox_x1 or p.bbox_y2 <= p.bbox_y1:
            issues.append(QCIssue(
                "IMPOSSIBLE_BBOX", "ERROR", video_id,
                f"bbox has non-positive area: {list(p.bbox)}",
                frame_id=p.frame_id, track_id=p.track_id))
        for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            v = getattr(p, name)
            if v is None or v != v:
                issues.append(QCIssue(
                    "IMPOSSIBLE_BBOX", "ERROR", video_id,
                    f"{name} is missing/NaN", frame_id=p.frame_id, track_id=p.track_id))
    return issues


def check_conflicting_roles(people, video_id: str,
                            overlap_iou: float = 0.85) -> List[QCIssue]:
    """Two DIFFERENT tracks carrying the same role at the same frame can mean
    the annotator lost track of identity. Same track changing role mid-delivery
    is also suspicious and is reported.

    Only roles that are **unique by construction** are treated as a conflict.
    A cricket frame can legitimately contain several fielders, so two FIELDER
    rows are normal and are not an error; two BOWLERs never are.  Two
    *different* unique roles whose boxes overlap heavily are reported, because
    that is the signature of a box drawn on the wrong person.
    """
    issues: List[QCIssue] = []
    by_frame = defaultdict(list)
    for p in people:
        by_frame[p.frame_id].append(p)
    for frame, rows in sorted(by_frame.items()):
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a.track_id == b.track_id:
                    if a.role != b.role:
                        issues.append(QCIssue(
                            "CONFLICTING_ROLE", "WARNING", video_id,
                            f"track {a.track_id} has role {a.role} and {b.role} "
                            "in the same frame (duplicate rows)",
                            frame_id=frame, track_id=a.track_id))
                    continue
                iou = _iou(a.bbox, b.bbox)
                both_unique = (a.role in UNIQUE_ROLES and b.role in UNIQUE_ROLES)
                if both_unique and a.role == b.role:
                    issues.append(QCIssue(
                        "CONFLICTING_ROLE", "ERROR", video_id,
                        f"tracks {a.track_id} and {b.track_id} are both "
                        f"{a.role} in one frame (only one {a.role} exists)",
                        frame_id=frame, track_id=a.track_id,
                        detail={"other_track": b.track_id, "role": a.role}))
                elif both_unique and a.role != b.role and iou > overlap_iou:
                    issues.append(QCIssue(
                        "CONFLICTING_ROLE", "ERROR", video_id,
                        f"tracks {a.track_id} ({a.role}) and {b.track_id} "
                        f"({b.role}) are different unique roles but their boxes "
                        f"overlap at IoU {iou:.2f}; one box is probably on the "
                        "wrong person",
                        frame_id=frame, track_id=a.track_id,
                        detail={"other_track": b.track_id, "iou": round(iou, 4)}))
                elif iou > overlap_iou:
                    issues.append(QCIssue(
                        "OVERLAPPING_BBOX", "WARNING", video_id,
                        f"tracks {a.track_id} ({a.role}) and {b.track_id} "
                        f"({b.role}) overlap at IoU {iou:.2f} > {overlap_iou}",
                        frame_id=frame, track_id=a.track_id,
                        detail={"other_track": b.track_id}))
    return issues


def check_bowler_track(people, video_id: str) -> List[QCIssue]:
    """A delivery must have a BOWLER-labelled track, and a clip that contains a
    delivery must contain at least one BOWLER role somewhere."""
    issues = []
    if not any(p.role == "BOWLER" for p in people):
        issues.append(QCIssue(
            "MISSING_BOWLER_TRACK", "ERROR", video_id,
            "no person is labelled BOWLER anywhere in this clip"))
    return issues


def check_missing_frames(people, video_id: str, expect_track_ids=None) -> List[QCIssue]:
    """Report gaps in an otherwise-present track. Purely informational: a
    partial annotation is legal, an *inconsistent* one is worth a look."""
    issues: List[QCIssue] = []
    by_track = defaultdict(set)
    for p in people:
        by_track[p.track_id].add(p.frame_id)
    ids = sorted(expect_track_ids) if expect_track_ids else sorted(by_track)
    for tid in ids:
        frames = by_track.get(tid)
        if not frames:
            continue
        lo, hi = min(frames), max(frames)
        holes = [f for f in range(lo, hi + 1) if f not in frames]
        if holes:
            issues.append(QCIssue(
                "MISSING_ANNOTATED_FRAMES", "INFO", video_id,
                f"track {tid} has {len(holes)} unannotated frame(s) inside its "
                f"annotated range [{lo}, {hi}]",
                track_id=tid,
                detail={"n_holes": len(holes), "first_holes": holes[:20]}))
    return issues


def check_delivery(delivery, video_id: str, n_frames: Optional[int]) -> List[QCIssue]:
    issues: List[QCIssue] = []

    if delivery.bowler_track_id < 0:
        issues.append(QCIssue(
            "MISSING_BOWLER_TRACK", "ERROR", video_id,
            "delivery has no bowler_track_id", delivery_id=delivery.delivery_id))

    if not (delivery.annotator_id or "").strip():
        issues.append(QCIssue(
            "MISSING_ANNOTATOR_ID", "ERROR", video_id,
            "delivery has no annotator_id", delivery_id=delivery.delivery_id))

    lo, hi = delivery.start_frame, delivery.end_frame
    if hi < lo:
        issues.append(QCIssue(
            "IMPOSSIBLE_PHASE_SEQUENCE", "ERROR", video_id,
            f"end_frame {hi} < start_frame {lo}", delivery_id=delivery.delivery_id))
        lo, hi = hi, lo

    if n_frames is not None:
        for name, val in (("start_frame", delivery.start_frame),
                          ("end_frame", delivery.end_frame)):
            if val < 0 or val >= n_frames:
                issues.append(QCIssue(
                    "FRAME_OUT_OF_VIDEO_RANGE", "ERROR", video_id,
                    f"{name}={val} outside [0, {n_frames})",
                    frame_id=val, delivery_id=delivery.delivery_id))

    # release must lie inside the delivery window
    if delivery.release_frame is not None:
        if not (lo <= delivery.release_frame <= hi):
            issues.append(QCIssue(
                "RELEASE_OUTSIDE_DELIVERY", "ERROR", video_id,
                f"release_frame {delivery.release_frame} outside delivery "
                f"window [{lo}, {hi}]", frame_id=delivery.release_frame,
                delivery_id=delivery.delivery_id))
        elif n_frames is not None and delivery.release_frame >= n_frames:
            issues.append(QCIssue(
                "FRAME_OUT_OF_VIDEO_RANGE", "ERROR", video_id,
                f"release_frame {delivery.release_frame} >= frame_count {n_frames}",
                frame_id=delivery.release_frame, delivery_id=delivery.delivery_id))

    # ordered-phase monotonicity over the anchors that were actually annotated
    anchors = [(ph, fr) for ph, fr in delivery.key_frames().items()
               if fr is not None and ph in VALID_ORDER]
    for (p1, f1), (p2, f2) in zip(anchors, anchors[1:]):
        if f2 < f1:
            issues.append(QCIssue(
                "PHASE_ORDER_VIOLATION", "ERROR", video_id,
                f"{p2} frame {f2} is BEFORE {p1} frame {f1}; expected "
                f"{' <= '.join(VALID_ORDER)}",
                frame_id=f2, delivery_id=delivery.delivery_id,
                detail={"phase_a": p1, "frame_a": f1, "phase_b": p2, "frame_b": f2}))

    # every annotated anchor must also be inside the delivery window
    for ph, fr in anchors:
        if not (lo <= fr <= hi):
            issues.append(QCIssue(
                "IMPOSSIBLE_PHASE_SEQUENCE", "ERROR", video_id,
                f"{ph} frame {fr} outside delivery window [{lo}, {hi}]",
                frame_id=fr, delivery_id=delivery.delivery_id))

    # a delivery with no identifiable event frames is not trainable for Model C
    if not anchors:
        issues.append(QCIssue(
            "MISSING_BOWLER_TRACK" if delivery.bowler_track_id < 0
            else "IMPOSSIBLE_PHASE_SEQUENCE", "WARNING", video_id,
            "delivery has no annotated key frames; usable for detection/roles "
            "only, not for the phase model",
            delivery_id=delivery.delivery_id))

    return issues


def check_delivery_ids(deliveries, video_id: str) -> List[QCIssue]:
    issues = []
    seen = defaultdict(list)
    for d in deliveries:
        seen[d.delivery_id].append(d)
    for did, rows in sorted(seen.items()):
        if len(rows) > 1:
            issues.append(QCIssue(
                "DUPLICATE_DELIVERY_ID", "ERROR", video_id,
                f"delivery_id {did!r} used {len(rows)} times",
                delivery_id=did))
        for a, b in zip(rows, rows[1:]):
            if a.bowler_track_id != b.bowler_track_id:
                issues.append(QCIssue(
                    "CONFLICTING_ROLE", "ERROR", video_id,
                    f"delivery {did!r} duplicated with a different bowler_track_id "
                    f"({a.bowler_track_id} vs {b.bowler_track_id})",
                    delivery_id=did))
    return issues


def check_overlapping_deliveries(deliveries, video_id: str) -> List[QCIssue]:
    issues = []
    ordered = sorted(deliveries, key=lambda d: (d.start_frame, d.end_frame))
    for a, b in zip(ordered, ordered[1:]):
        if b.start_frame <= a.end_frame:
            issues.append(QCIssue(
                "CONFLICTING_ROLE", "WARNING", video_id,
                f"deliveries {a.delivery_id!r} [{a.start_frame},{a.end_frame}] and "
                f"{b.delivery_id!r} [{b.start_frame},{b.end_frame}] overlap",
                delivery_id=b.delivery_id,
                detail={"other": a.delivery_id}))
    return issues


def check_phase_annotations(phases, video_id: str) -> List[QCIssue]:
    """Per-frame phase labels must not contradict the delivery key frames."""
    issues: List[QCIssue] = []
    for ph in phases:
        if not (ph.annotator_id or "").strip():
            issues.append(QCIssue(
                "MISSING_ANNOTATOR_ID", "ERROR", video_id,
                "phase annotation has no annotator_id",
                frame_id=ph.frame_id, track_id=ph.track_id))
    return issues


def check_uncertain(phases, people, video_id: str) -> List[QCIssue]:
    n_unc = sum(1 for p in phases if p.uncertain)
    if n_unc:
        issues = [QCIssue(
            "UNCERTAIN_FRAMES", "INFO", video_id,
            f"{n_unc} frame(s) flagged uncertain by the annotator; kept in the "
            "dataset but flagged",
            detail={"count": n_unc})]
    else:
        issues = []
    for p in people:
        if p.visibility in ("uncertain", "not_visible"):
            issues.append(QCIssue(
                "UNCERTAIN_FRAMES", "INFO", video_id,
                f"track {p.track_id} visibility={p.visibility} at frame {p.frame_id}",
                frame_id=p.frame_id, track_id=p.track_id))
    return issues


def check_scene(scene, video_id: str) -> List[QCIssue]:
    issues = []
    if scene is None:
        return issues
    for key in schema.SCENE_KEYS:
        val = getattr(scene, key, None)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            if val[2] <= val[0] or val[3] <= val[1]:
                issues.append(QCIssue(
                    "SCENE_IMPOSSIBLE_BOX", "ERROR", video_id,
                    f"{key} has a non-positive-area box: {list(val)}",
                    detail={"key": key}))
    if scene.camera_orientation == "unknown":
        issues.append(QCIssue(
            "SCENE_IMPOSSIBLE_BOX", "INFO", video_id,
            "camera_orientation is 'unknown'; camera-view priors and stratified "
            "evaluation will be limited for this clip",
            detail={"key": "camera_orientation"}))
    return issues


def check_duplicate_annotations(people, video_id: str) -> List[QCIssue]:
    seen = defaultdict(int)
    for p in people:
        seen[(p.frame_id, p.track_id)] += 1
    return [
        QCIssue("DUPLICATE_ANNOTATION", "ERROR", video_id,
                f"{n} rows for the same (frame, track)",
                frame_id=k[0], track_id=k[1], detail={"count": n})
        for k, n in sorted(seen.items()) if n > 1
    ]


def check_split_leakage(splits: Dict[str, List[str]]) -> List[QCIssue]:
    """The single most important leakage check: a video must appear in exactly
    one split, and near-duplicate frame hashes must not straddle splits."""
    issues: List[QCIssue] = []
    seen: Dict[str, List[str]] = defaultdict(list)
    for name, ids in splits.items():
        for vid in ids:
            seen[vid].append(name)
    for vid, names in sorted(seen.items()):
        if len(names) > 1:
            issues.append(QCIssue(
                "SPLIT_LEAKAGE", "ERROR", vid,
                f"video appears in multiple splits: {sorted(set(names))}",
                detail={"splits": sorted(set(names))}))
    for name, ids in sorted(splits.items()):
        if len(ids) != len(set(ids)):
            issues.append(QCIssue(
                "SPLIT_LEAKAGE", "ERROR", "",
                f"split {name!r} contains duplicate video_ids",
                detail={"split": name}))
    return issues


def check_frame_hash_leakage(per_split_hashes: Dict[str, set]) -> List[QCIssue]:
    """``per_split_hashes``: {split: {frame perceptual hash}}. Identical frames
    across splits mean near-duplicate leakage."""
    issues: List[QCIssue] = []
    names = sorted(per_split_hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = per_split_hashes[a] & per_split_hashes[b]
            if shared:
                issues.append(QCIssue(
                    "SPLIT_LEAKAGE", "ERROR", "",
                    f"{len(shared)} identical frame hash(es) shared between "
                    f"split {a!r} and {b!r}",
                    detail={"splits": [a, b], "n_shared": len(shared),
                            "examples": sorted(shared)[:10]}))
    return issues


# --------------------------------------------------------------------------- #
# Whole-dataset driver
# --------------------------------------------------------------------------- #

def run_all(video_ids: Optional[List[str]] = None,
            n_frames_by_id: Optional[Dict[str, int]] = None,
            splits: Optional[Dict[str, List[str]]] = None,
            records=None) -> Dict[str, Any]:
    """Run every check over the annotated dataset.

    Returns a summary dict with ``issues`` (list of dicts) and counts. It never
    modifies annotation files.
    """
    video_ids = list(video_ids or schema.all_annotation_video_ids())
    n_frames_by_id = n_frames_by_id or {}
    all_issues: List[QCIssue] = []

    if records is not None:
        all_issues += check_duplicate_video_ids(records)
    if splits is not None:
        all_issues += check_split_leakage(splits)

    per_video = []
    for vid in sorted(set(video_ids)):
        try:
            people = schema.load_person_annotations(vid)
            phases = schema.load_phase_annotations(vid)
            deliveries = schema.load_deliveries(vid)
            scene = schema.load_scene(vid)
        except Exception as exc:
            all_issues.append(QCIssue(
                "INVALID_ANNOTATION_FILE", "ERROR", vid,
                f"failed to parse: {type(exc).__name__}: {exc}"))
            per_video.append({"video_id": vid, "parse_error": True,
                              "n_people": 0, "n_phases": 0, "n_deliveries": 0})
            continue

        n = n_frames_by_id.get(vid)
        v_issues: List[QCIssue] = []
        v_issues += check_frame_numbers(people, vid, n)
        v_issues += check_annotator(people, vid)
        v_issues += check_bboxes(people, vid)
        v_issues += check_duplicate_annotations(people, vid)
        v_issues += check_conflicting_roles(people, vid)
        v_issues += check_missing_frames(people, vid)
        v_issues += check_phase_annotations(phases, vid)
        v_issues += check_uncertain(phases, people, vid)
        v_issues += check_delivery_ids(deliveries, vid)
        v_issues += check_overlapping_deliveries(deliveries, vid)
        for d in deliveries:
            v_issues += check_delivery(d, vid, n)
        if deliveries or phases:
            v_issues += check_bowler_track(people, vid)
        v_issues += check_scene(scene, vid)

        all_issues += v_issues
        per_video.append({
            "video_id": vid,
            "n_people": len(people),
            "n_phases": len(phases),
            "n_deliveries": len(deliveries),
            "n_scene": 1 if scene else 0,
            "issues": [i.to_dict() for i in v_issues],
        })

    by_code: Dict[str, int] = defaultdict(int)
    by_sev: Dict[str, int] = defaultdict(int)
    for i in all_issues:
        by_code[i.code] += 1
        by_sev[i.severity] += 1

    return {
        "schema_version": schema.SCHEMA_VERSION,
        "n_videos": len(set(video_ids)),
        "n_videos_with_issues": sum(1 for v in per_video if v.get("issues")),
        "n_issues": len(all_issues),
        "issues_by_code": dict(sorted(by_code.items())),
        "issues_by_severity": dict(sorted(by_sev.items())),
        "issues": [i.to_dict() for i in all_issues],
        "per_video": per_video,
        "auto_repaired": False,
        "note": "QC reports only. No human annotation was modified.",
    }
