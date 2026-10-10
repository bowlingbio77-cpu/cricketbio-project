"""
Phase 3 -- HUMAN ANNOTATION session (OpenCV GUI, local, offline).

This module builds the annotation **state machine**.  It is deliberately kept
free of any OpenCV *window* code so it can be unit tested headlessly; the
window, key bindings and drawing live in
``tools/annotate_cricket.py``.

NON-NEGOTIABLE RULES (from the project brief)
--------------------------------------------
1. **Ground truth is never pre-filled from a PaceAI prediction.**  The session
   object has no reference to any model, tracker, or prediction cache.  If
   ``--show-suggestions`` is used, suggestions arrive in a *separate* object
   with a ``suggestion=True`` marker, are drawn in a distinct colour inside a
   panel explicitly captioned as non-ground-truth, and are only ever applied
   when the human presses a key to accept them.  They are never written to the
   annotation file without that explicit act, and ``source`` is recorded as
   ``"human"`` only after the human sets the value.
2. ``annotator_id`` is mandatory.  The constructor refuses to build without it.
3. **Incremental, crash-safe saving.**  Every mutating call is followed by an
   atomic write (temp file + ``os.replace``), and a rolling autosave timer keeps
   a crash from losing more than a few seconds of work.
4. A prominent ``HUMAN ANNOTATION / Measured independently -- NOT model output``
   banner is drawn on every frame by the renderer.

The session stores per-frame person boxes keyed by ``track_id``.  ``track_id``
here is an *annotator-assigned identity label* for a person, not a ByteTrack id
(see :meth:`AnnotationSession.note_suggested_track_id`).
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import schema

BANNER_LINE_1 = "HUMAN ANNOTATION"
BANNER_LINE_2 = "Measured independently - NOT model output"

AUTOSAVE_SECONDS = 20.0


class AnnotationError(ValueError):
    """Raised for an invalid annotation edit. The session state is unchanged."""


# --------------------------------------------------------------------------- #

@dataclass
class Suggestion:
    """A model-derived hint. NEVER ground truth.

    Held in a parallel structure so it cannot be confused with an annotation
    record, and so a bug that copies one into the other is visible in review.
    """
    video_id: str
    frame_id: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    suggested_role: Optional[str] = None
    suggested_phase: Optional[str] = None
    source: str = "paceai"
    score: Optional[float] = None
    suggestion: bool = True          # always True -- the marker that matters

    def __post_init__(self) -> None:
        if not self.suggestion:
            raise AnnotationError("Suggestion.suggestion must always be True.")
        if self.suggested_role is not None and self.suggested_role not in schema.PLAYER_ROLES.values():
            raise AnnotationError(f"bad suggested_role {self.suggested_role!r}")
        if self.suggested_phase is not None and self.suggested_phase not in schema.BOWLING_PHASES:
            raise AnnotationError(f"bad suggested_phase {self.suggested_phase!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id, "frame_id": self.frame_id,
            "track_id": self.track_id, "bbox": list(self.bbox),
            "suggested_role": self.suggested_role,
            "suggested_phase": self.suggested_phase,
            "source": self.source, "score": self.score, "suggestion": True,
        }


# --------------------------------------------------------------------------- #

class AnnotationSession:
    """Mutable, crash-safe annotation state for exactly one video + one human."""

    def __init__(self,
                 video_id: str,
                 annotator_id: str,
                 n_frames: int,
                 video_path: Optional[str] = None,
                 fps: Optional[float] = None,
                 autosave: bool = True):
        annotator_id = (annotator_id or "").strip()
        if not annotator_id:
            raise AnnotationError(
                "annotator_id is REQUIRED. Ground truth must be attributable to a "
                "named human. Pass --annotator_id <ID>."
            )
        if not video_id:
            raise AnnotationError("video_id is required.")
        if n_frames <= 0:
            raise AnnotationError(f"n_frames must be positive, got {n_frames}")

        self.video_id = video_id
        self.annotator_id = annotator_id
        self.n_frames = n_frames
        self.video_path = video_path
        self.fps = fps
        self.autosave = autosave

        # ---- ground truth, keyed by (frame_id, track_id) -------------------
        self.people: Dict[Tuple[int, int], schema.PersonAnnotation] = {}
        self.objects: Dict[Tuple[int, int, str], schema.ObjectAnnotation] = {}
        self.phases: Dict[Tuple[int, int], schema.PhaseAnnotation] = {}
        self.deliveries: Dict[str, schema.DeliveryAnnotation] = {}
        self.scene: Optional[schema.SceneAnnotation] = None
        self.notes: str = ""

        # ---- non-ground-truth ---------------------------------------------
        self.suggestions: Dict[Tuple[int, int], Suggestion] = {}
        self.suggestions_enabled = False

        # ---- ui state -------------------------------------------------------
        self.current_frame = 0
        self.selected_track: Optional[int] = None
        self.draft_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_save = 0.0
        self._dirty = False
        self.messages: List[str] = []

    # -- persistence -------------------------------------------------------- #

    def _dirty_flag(self) -> None:
        self._dirty = True
        if not self.autosave:
            return
        if (time.time() - self._last_save) >= AUTOSAVE_SECONDS:
            self.save()

    def load(self) -> bool:
        """Load any existing annotation for this video. Returns True if found."""
        people = schema.load_person_annotations(self.video_id)
        if people:
            self.people = {(p.frame_id, p.track_id): p for p in people}
        objs = schema.load_object_annotations(self.video_id)
        if objs:
            self.objects = {(o.frame_id, o.track_id, o.object_class): o for o in objs}
        phases = schema.load_phase_annotations(self.video_id)
        if phases:
            self.phases = {(p.frame_id, p.track_id): p for p in phases}
        dels = schema.load_deliveries(self.video_id)
        if dels:
            self.deliveries = {d.delivery_id: d for d in dels}
        scene = schema.load_scene(self.video_id)
        if scene:
            self.scene = scene
        self._dirty = False
        self._last_save = time.time()
        return bool(people or objs or phases or dels or scene)

    def save(self, force: bool = True) -> Dict[str, str]:
        """Atomically persist every annotation kind for this video."""
        if not self._dirty and not force:
            return {}
        paths = {
            "roles": schema.save_person_annotations(self.video_id, list(self.people.values())),
            "objects": schema.save_object_annotations(self.video_id, list(self.objects.values())),
            "bowling_phases": schema.save_phase_annotations(self.video_id, list(self.phases.values())),
            "deliveries": schema.save_deliveries(self.video_id, list(self.deliveries.values())),
        }
        if self.scene is not None:
            paths["scene"] = schema.save_scene(self.video_id, self.scene)
        self._dirty = False
        self._last_save = time.time()
        return paths

    # -- people / roles ----------------------------------------------------- #

    def _check_frame(self, frame_id: int) -> int:
        if not isinstance(frame_id, (int,)) or isinstance(frame_id, bool):
            raise AnnotationError(f"frame_id must be an int, got {frame_id!r}")
        if not 0 <= frame_id < self.n_frames:
            raise AnnotationError(
                f"frame_id {frame_id} outside video range [0, {self.n_frames})")
        return frame_id

    @staticmethod
    def _check_track(track_id: int) -> int:
        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
            raise AnnotationError(f"track_id must be a non-negative int, got {track_id!r}")
        return track_id

    def set_person(self, frame_id: int, track_id: int, bbox: Sequence[float],
                   role: str = "UNKNOWN",
                   visibility: str = "fully_visible",
                   annotation_confidence: float = 1.0,
                   notes: str = "") -> schema.PersonAnnotation:
        self._check_frame(frame_id)
        self._check_track(track_id)
        if len(bbox) != 4:
            raise AnnotationError(f"bbox must be [x1,y1,x2,y2], got {bbox!r}")
        rec = schema.PersonAnnotation(
            video_id=self.video_id, frame_id=frame_id, track_id=track_id,
            role=role, bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
            bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]),
            visibility=visibility, annotation_confidence=annotation_confidence,
            annotator_id=self.annotator_id, notes=notes,
        )
        self.people[(frame_id, track_id)] = rec
        self.selected_track = track_id
        self._dirty_flag()
        return rec

    def set_role(self, frame_id: int, track_id: int, role: str) -> schema.PersonAnnotation:
        """Change only the role of an existing person row (bbox must exist)."""
        key = (frame_id, track_id)
        cur = self.people.get(key)
        if cur is None:
            raise AnnotationError(
                f"no person box at frame {frame_id} track {track_id}; draw the box "
                "before assigning a role."
            )
        rec = self.set_person(frame_id, track_id, cur.bbox, role=role,
                              visibility=cur.visibility,
                              annotation_confidence=cur.annotation_confidence,
                              notes=cur.notes)
        return rec

    def set_bowl(self, frame_id: int, track_id: int) -> schema.DeliveryAnnotation:
        """Set the phase row of one track to the BOWLER-anchored anchor frame."""
        key = (frame_id, track_id)
        rec = self.phases.get(key)
        if rec is None:
            rec = schema.PhaseAnnotation(
                video_id=self.video_id, frame_id=frame_id, track_id=track_id,
                phase="GATHER", annotation_confidence=1.0,
                annotator_id=self.annotator_id)
        self.phases[key] = rec
        self._dirty_flag()
        return rec

    def set_visibility(self, frame_id: int, track_id: int, visibility: str) -> None:
        rec = self.people.get((frame_id, track_id))
        if rec is None:
            raise AnnotationError(
                f"no person box at frame {frame_id} track {track_id}")
        self.people[(frame_id, track_id)] = self.set_person(
            frame_id, track_id, rec.bbox, role=rec.role, visibility=visibility,
            annotation_confidence=rec.annotation_confidence, notes=rec.notes)

    def move_bbox(self, frame_id: int, track_id: int,
                  bbox: Optional[Sequence[float]]) -> None:
        """Edit / resize a box, or delete it with ``bbox=None``."""
        key = (frame_id, track_id)
        rec = self.people.get(key)
        if rec is None:
            raise AnnotationError(f"no person box at frame {frame_id} track {track_id}")
        if bbox is None:
            self.people.pop(key, None)
            self.phases.pop(key, None)
            self._dirty_flag()
            return
        self.set_person(frame_id, track_id, bbox, role=rec.role,
                        visibility=rec.visibility,
                        annotation_confidence=rec.annotation_confidence,
                        notes=rec.notes)

    def apply_role_range(self, start_frame: int, end_frame: int, track_id: int,
                         role: str) -> int:
        """Label a whole frame range for one track in one action.

        Only frames that already have a box for ``track_id`` are touched -- the
        tool never invents a box the human did not draw.
        """
        if end_frame < start_frame:
            raise AnnotationError("end_frame < start_frame")
        n = 0
        for f in range(max(0, start_frame), min(self.n_frames, end_frame + 1)):
            if (f, track_id) in self.people:
                self.set_role(f, track_id, role)
                n += 1
        if n == 0:
            raise AnnotationError(
                f"track {track_id} has no boxes in [{start_frame}, {end_frame}]")
        return n

    # -- objects ------------------------------------------------------------ #

    def set_object(self, frame_id: int, track_id: int, object_class: str,
                   bbox: Sequence[float], visibility: str = "fully_visible",
                   annotation_confidence: float = 1.0) -> schema.ObjectAnnotation:
        self._check_frame(frame_id)
        self._check_track(track_id)
        if len(bbox) != 4:
            raise AnnotationError(f"bbox must be [x1,y1,x2,y2], got {bbox!r}")
        rec = schema.ObjectAnnotation(
            video_id=self.video_id, frame_id=frame_id, track_id=track_id,
            object_class=object_class, bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
            bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]),
            visibility=visibility, annotation_confidence=annotation_confidence,
            annotator_id=self.annotator_id)
        self.objects[(frame_id, track_id, object_class)] = rec
        self._dirty_flag()
        return rec

    def clear_object(self, frame_id: int, track_id: int, object_class: str) -> bool:
        return self.objects.pop((frame_id, track_id, object_class), None) is not None

    # -- phases ------------------------------------------------------------- #

    def set_phase(self, frame_id: int, track_id: int, phase: str,
                  uncertain: bool = False,
                  pose_confidence: Optional[float] = None,
                  annotation_confidence: float = 1.0,
                  notes: str = "") -> schema.PhaseAnnotation:
        self._check_frame(frame_id)
        self._check_track(track_id)
        rec = schema.PhaseAnnotation(
            video_id=self.video_id, frame_id=frame_id, track_id=track_id, phase=phase,
            pose_confidence=pose_confidence, uncertain=uncertain,
            annotation_confidence=annotation_confidence,
            annotator_id=self.annotator_id, notes=notes)
        self.phases[(frame_id, track_id)] = rec
        self._dirty_flag()
        return rec

    def apply_phase_range(self, start_frame: int, end_frame: int, track_id: int,
                          phase: str, uncertain: bool = False) -> int:
        if end_frame < start_frame:
            raise AnnotationError("end_frame < start_frame")
        n = 0
        for f in range(max(0, start_frame), min(self.n_frames, end_frame + 1)):
            self.set_phase(f, track_id, phase, uncertain=uncertain)
            n += 1
        if n == 0:
            raise AnnotationError("empty phase range")
        return n

    # -- deliveries --------------------------------------------------------- #

    def start_delivery(self, delivery_id: str, bowler_track_id: int,
                       start_frame: Optional[int] = None) -> schema.DeliveryAnnotation:
        if not delivery_id or not str(delivery_id).strip():
            raise AnnotationError("delivery_id must be a non-empty string")
        self._check_track(bowler_track_id)
        d = schema.DeliveryAnnotation(
            delivery_id=str(delivery_id).strip(), video_id=self.video_id,
            bowler_track_id=bowler_track_id,
            start_frame=self.current_frame if start_frame is None else int(start_frame),
            end_frame=self.current_frame if start_frame is None else int(start_frame),
            annotator_id=self.annotator_id, notes=self.notes)
        self.deliveries[d.delivery_id] = d
        self._dirty_flag()
        return d

    def set_delivery_frame(self, delivery_id: str, field_name: str,
                           frame: Optional[int]) -> schema.DeliveryAnnotation:
        """Set one key frame. ``None`` is legal and means 'not annotated'."""
        allowed = {
            "runup_start_frame", "gather_frame", "delivery_stride_frame",
            "front_foot_contact_frame", "release_frame", "followthrough_end_frame",
            "start_frame", "end_frame",
        }
        if field_name not in allowed:
            raise AnnotationError(
                f"unknown delivery field {field_name!r}; expected one of {sorted(allowed)}")
        d = self.deliveries.get(str(delivery_id))
        if d is None:
            raise AnnotationError(f"no delivery {delivery_id!r}")
        if frame is not None:
            frame = int(frame)
            if not 0 <= frame < self.n_frames:
                raise AnnotationError(
                    f"frame {frame} outside video range [0, {self.n_frames})")
        # A rebuild enforces the schema's own validation (end >= start) without
        # ever inventing a value: unset key frames stay None.
        d = schema.DeliveryAnnotation(**{**d.to_dict(), field_name: frame,
                                         "notes": self.notes or d.notes})
        self.deliveries[d.delivery_id] = d
        self._dirty_flag()
        return d

    def set_delivery_window(self, delivery_id: str, start_frame: int,
                            end_frame: int) -> schema.DeliveryAnnotation:
        d = self.deliveries.get(str(delivery_id))
        if d is None:
            raise AnnotationError(f"no delivery {delivery_id!r}")
        d = schema.DeliveryAnnotation(**{
            **d.to_dict(), "start_frame": int(start_frame),
            "end_frame": int(end_frame), "notes": self.notes or d.notes})
        self.deliveries[d.delivery_id] = d
        self._dirty_flag()
        return d

    def set_delivery_uncertain(self, delivery_id: str, uncertain: bool) -> None:
        d = self.deliveries.get(str(delivery_id))
        if d is None:
            raise AnnotationError(f"no delivery {delivery_id!r}")
        d = schema.DeliveryAnnotation(**{**d.to_dict(), "uncertain": bool(uncertain)})
        self.deliveries[d.delivery_id] = d
        self._dirty_flag()

    def delete_delivery(self, delivery_id: str) -> bool:
        return self.deliveries.pop(str(delivery_id), None) is not None

    def current_delivery(self) -> Optional[schema.DeliveryAnnotation]:
        """The delivery whose window contains the current frame, else the last
        one that starts at or before it."""
        f = self.current_frame
        inside = [d for d in self.deliveries.values()
                  if d.start_frame <= f <= d.end_frame]
        if inside:
            return sorted(inside, key=lambda d: d.delivery_id)[0]
        before = [d for d in self.deliveries.values() if d.start_frame <= f]
        if before:
            return max(before, key=lambda d: (d.start_frame, d.delivery_id))
        return None

    def mark_key_frame(self, field_name: str,
                       frame: Optional[int] = None) -> schema.DeliveryAnnotation:
        """One-key workflow: stamp the current frame onto the current delivery."""
        d = self.current_delivery()
        if d is None:
            raise AnnotationError(
                "no delivery covers this frame. Press 'n' to start one first.")
        return self.set_delivery_frame(d.delivery_id, field_name,
                                      self.current_frame if frame is None else frame)

    # -- scene -------------------------------------------------------------- #

    def set_scene_field(self, key: str, value) -> Optional[schema.SceneAnnotation]:
        if key not in schema.SCENE_KEYS:
            raise AnnotationError(
                f"unknown scene key {key!r}; expected one of {schema.SCENE_KEYS}")
        if self.scene is None:
            self.scene = schema.SceneAnnotation(
                video_id=self.video_id, camera_orientation="unknown",
                annotator_id=self.annotator_id)
        data = self.scene.to_dict()
        data[key] = value
        self.scene = schema.SceneAnnotation(**data)
        self._dirty_flag()
        return self.scene

    def set_scene_from_box(self, key: str, bbox: Sequence[float]) -> None:
        if len(bbox) != 4:
            raise AnnotationError(f"scene {key} box must be [x1,y1,x2,y2]")
        self.set_scene_field(key, [float(v) for v in bbox])

    def set_camera_orientation(self, value: str) -> None:
        self.set_scene_field("camera_orientation", value)

    # -- suggestions (NEVER ground truth) ----------------------------------- #

    def load_suggestions(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Load model hints into a *separate* store. Nothing is written to the
        ground-truth dicts by this method."""
        self.suggestions = {}
        for r in rows:
            try:
                s = Suggestion(
                    video_id=r.get("video_id", self.video_id),
                    frame_id=int(r["frame_id"]), track_id=int(r["track_id"]),
                    bbox=tuple(float(v) for v in r["bbox"]),
                    suggested_role=r.get("suggested_role"),
                    suggested_phase=r.get("suggested_phase"),
                    source=r.get("source", "paceai"), score=r.get("score"))
            except (KeyError, TypeError, ValueError, AnnotationError):
                continue
            self.suggestions[(s.frame_id, s.track_id)] = s
        self.suggestions_enabled = bool(self.suggestions)
        return len(self.suggestions)

    def suggestions_at(self, frame_id: int) -> List[Suggestion]:
        return [s for (f, _t), s in self.suggestions.items() if f == frame_id]

    def accept_suggestion(self, track_id: int, frame_id: Optional[int] = None) -> schema.PersonAnnotation:
        """The ONLY path by which a suggestion becomes an annotation -- and it
        is an explicit human action, so the resulting record is human-verified.

        The accepted box still gets ``role="UNKNOWN"`` and
        ``annotation_confidence=0.5``: the human accepted the *box*, not
        silently endorsing a model label.
        """
        f = self.current_frame if frame_id is None else frame_id
        s = self.suggestions.get((f, track_id))
        if s is None:
            raise AnnotationError(
                f"no suggestion for track {track_id} at frame {f}")
        rec = self.set_person(f, track_id, s.bbox, role="UNKNOWN",
                              visibility="uncertain",
                              annotation_confidence=0.5,
                              notes=f"box accepted from {s.source} suggestion; "
                                    "role/visibility must be set by the human")
        if s.suggested_role:
            self.set_role(f, track_id, s.suggested_role)
        return self.people[(f, track_id)]

    # -- introspection / QC preview ----------------------------------------- #

    def people_at(self, frame_id: int) -> List[schema.PersonAnnotation]:
        return sorted((p for (f, _t), p in self.people.items() if f == frame_id),
                      key=lambda p: p.track_id)

    def track_ids(self) -> List[int]:
        return sorted({t for (_f, t) in self.people})

    def frames_for_track(self, track_id: int) -> List[int]:
        return sorted(f for (f, t) in self.people if t == track_id)

    def annotated_frames(self) -> List[int]:
        return sorted({f for (f, _t) in self.people})

    def phase_at(self, frame_id: int, track_id: int) -> Optional[schema.PhaseAnnotation]:
        return self.phases.get((frame_id, track_id))

    def role_at(self, frame_id: int, track_id: int) -> Optional[str]:
        rec = self.people.get((frame_id, track_id))
        return rec.role if rec else None

    def counts(self) -> Dict[str, int]:
        return {
            "people": len(self.people),
            "objects": len(self.objects),
            "phases": len(self.phases),
            "deliveries": len(self.deliveries),
            "scene": 1 if self.scene else 0,
            "annotated_frames": len(self.annotated_frames()),
            "tracks": len(self.track_ids()),
            "suggestions": len(self.suggestions),
        }

    def next_delivery_id(self) -> str:
        n = 1
        used = set(self.deliveries)
        while f"d{n}" in used:
            n += 1
        return f"d{n}"

    def status_line(self) -> str:
        c = self.counts()
        return (f"f{self.current_frame}/{self.n_frames - 1} | "
                f"tracks={c['tracks']} boxes={c['people']} phases={c['phases']} "
                f"deliveries={c['deliveries']} | sel=T{self.selected_track}")
