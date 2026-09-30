"""
Phase 2 -- Versioned annotation schema for ``cricket_understanding_v1``.

This module is the single source of truth for label vocabularies, the on-disk
annotation layout, and (de)serialisation.  It is deliberately dependency-light
(standard library only) so that annotation, QC, splitting and dataset
generation can all run without torch, ultralytics or a GPU.

DESIGN RULES (from the project brief)
-------------------------------------
* Labels are human-authored.  Nothing here reads model predictions, and no
  default value is ever derived from a PaceAI output.
* Missing data is represented *explicitly* (``None`` / ``"UNKNOWN"`` /
  ``visibility="not_visible"``).  It is never silently imputed.
* Every annotation carries provenance: ``annotator_id``, ``annotation_date``,
  ``schema_version``.
* Serialisation is deterministic: key order is fixed, so re-saving an
  unchanged annotation produces a byte-identical file.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "cricket_understanding_v1"

# --------------------------------------------------------------------------- #
# Class vocabularies
# --------------------------------------------------------------------------- #

#: Player roles. Index -> label, exactly as specified in the brief.
PLAYER_ROLES: Dict[int, str] = {
    0: "BOWLER",
    1: "STRIKER",
    2: "NON_STRIKER",
    3: "WICKETKEEPER",
    4: "FIELDER",
    5: "UNKNOWN",
}
ROLE_INDEX: Dict[str, int] = {v: k for k, v in PLAYER_ROLES.items()}

#: Object classes live in a disjoint index range so one YOLO model can carry
#: persons (0-5) and cricket objects (10, 11) in a single ``names`` map.
OBJECT_CLASSES: Dict[int, str] = {
    10: "BALL",
    11: "STUMPS",
}
OBJECT_INDEX: Dict[str, int] = {v: k for k, v in OBJECT_CLASSES.items()}

#: Unified YOLO class map for Model A (Phase 9). Gaps are intentional and
#: represent classes with no labels yet; YOLO tolerates unused indices.
YOLO_CLASS_MAP: Dict[int, str] = {**PLAYER_ROLES, **OBJECT_CLASSES}
YOLO_NAMES: List[str] = [""] * (max(YOLO_CLASS_MAP) + 1)
for _i, _n in YOLO_CLASS_MAP.items():
    YOLO_NAMES[_i] = _n

#: Bowling action phases.
BOWLING_PHASES: List[str] = [
    "NON_BOWLING",
    "RUN_UP",
    "APPROACH",
    "GATHER",
    "DELIVERY_STRIDE",
    "FRONT_FOOT_CONTACT",
    "RELEASE",
    "FOLLOW_THROUGH",
    "UNKNOWN",
]
PHASE_INDEX: Dict[str, int] = {p: i for i, p in enumerate(BOWLING_PHASES)}

#: The canonical valid ordering of the ordered phases.  QC enforces that a
#: delivery's key frames are non-decreasing in this sequence.  ``NON_BOWLING``
#: and ``UNKNOWN`` are *excluded* from the ordering: they are not positions in
#: the delivery cycle.
ORDERED_PHASE_SEQUENCE: List[str] = [
    "RUN_UP",
    "APPROACH",
    "GATHER",
    "DELIVERY_STRIDE",
    "FRONT_FOOT_CONTACT",
    "RELEASE",
    "FOLLOW_THROUGH",
]
ORDERED_PHASE_RANK: Dict[str, int] = {p: i for i, p in enumerate(ORDERED_PHASE_SEQUENCE)}

#: Scene annotation keys.
SCENE_KEYS: List[str] = [
    "pitch_region",
    "bowling_crease",
    "popping_crease",
    "batting_crease",
    "stumps",
    "bowler_runup_region",
    "striker_region",
    "camera_orientation",
]

#: Explicit per-object visibility vocabulary (never a float, never inferred).
VISIBILITY_VALUES: List[str] = [
    "fully_visible",
    "partially_occluded",
    "heavily_occluded",
    "not_visible",
    "uncertain",
]

# --------------------------------------------------------------------------- #
# Review / status vocabularies (Phase 6)
# --------------------------------------------------------------------------- #

REVIEW_STATUS_VALUES: List[str] = ["UNREVIEWED", "IN_PROGRESS", "REVIEWED", "REJECTED"]
ANNOTATION_STATUS_VALUES: List[str] = ["NONE", "PARTIAL", "COMPLETE", "QC_FAILED", "QC_PASSED"]

CAMERA_VIEWS: List[str] = [
    "behind_bowler",
    "side_on",
    "square_leg",
    "fine_leg",
    "third_man",
    "short_boundary",
    "broadcast_mixed",
    "other",
    "unknown",
]
BOWLING_SIDES: List[str] = ["right", "left", "unknown"]


# --------------------------------------------------------------------------- #
# Enumerated string types
# --------------------------------------------------------------------------- #

def _enum(values: List[str], name: str):
    """Return a tiny validating enum-like str factory."""

    class _Enum(str):
        __slots__ = ()

        def __new__(cls, value: Any = None):
            if value is None:
                return None
            v = str(value)
            if v not in values:
                raise ValueError(
                    f"{name}: {v!r} is not valid; expected one of {values}"
                )
            return super().__new__(cls, v)

        @staticmethod
        def all() -> List[str]:
            return list(values)

    return _Enum


Role = _enum(list(PLAYER_ROLES.values()), "Role")
Phase = _enum(BOWLING_PHASES, "Phase")
Visibility = _enum(VISIBILITY_VALUES, "Visibility")
ReviewStatus = _enum(REVIEW_STATUS_VALUES, "ReviewStatus")
AnnotationStatus = _enum(ANNOTATION_STATUS_VALUES, "AnnotationStatus")
CameraView = _enum(CAMERA_VIEWS, "CameraView")
BowlingSide = _enum(BOWLING_SIDES, "BowlingSide")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cricket_understanding",
)


def dataset_root() -> str:
    return DATASET_ROOT


def p(*parts: str) -> str:
    """Join a path under the dataset root."""
    return os.path.join(DATASET_ROOT, *parts)


RAW_VIDEOS_DIR = p("raw", "videos")
RAW_METADATA_DIR = p("raw", "metadata")
EXTRACTED_FRAMES_DIR = p("extracted", "frames")
EXTRACTED_TRACKS_DIR = p("extracted", "tracks")
ANNOTATIONS_DIR = p("annotations")
ROLES_DIR = p("annotations", "roles")
BOWLING_PHASES_DIR = p("annotations", "bowling_phases")
DELIVERIES_DIR = p("annotations", "deliveries")
SCENE_DIR = p("annotations", "scene")
SPLITS_DIR = p("splits")
DATASET_YAML = p("dataset.yaml")
MANIFEST_JSON = p("dataset_manifest.json")

ALL_DIRS = [
    RAW_VIDEOS_DIR, RAW_METADATA_DIR, EXTRACTED_FRAMES_DIR, EXTRACTED_TRACKS_DIR,
    ANNOTATIONS_DIR, ROLES_DIR, BOWLING_PHASES_DIR, DELIVERIES_DIR, SCENE_DIR,
    SPLITS_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Stable video id
# --------------------------------------------------------------------------- #

def video_id_for_path(path: str) -> str:
    """Content-addressed video id.

    The repository contains 410 filename collisions caused by mixed 7-/8-digit
    zero padding (see ``evaluation/cricket_training_current_state.md`` W6), so a
    stem-derived id is unsafe.  More importantly a **stem-derived id also makes
    the same file look like two videos** when it is copied or renamed, which is
    exactly the duplication the registry is supposed to detect.

    So the id is derived from the bytes and the length only -- never the
    filename.  The full content is hashed (not a head/tail sample) so that two
    different files cannot collide; a sampled hash would be fast but would make
    "two different files can never collide" an unverified claim.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return f"v_{h.hexdigest()[:12]}"


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class PersonAnnotation:
    """One tracked person in one frame.

    ``visibility`` is an explicit enum string.  ``annotation_confidence`` is the
    annotator's own certainty (1.0 = certain).  Both are human-authored; neither
    is ever filled from a model.
    """
    video_id: str
    frame_id: int
    track_id: int
    role: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    visibility: str = "fully_visible"
    annotation_confidence: float = 1.0
    schema_version: str = SCHEMA_VERSION
    annotator_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.role = Role(self.role)
        self.visibility = Visibility(self.visibility)
        self._validate()

    def _validate(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            v = getattr(self, name)
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                raise ValueError(f"{name} is missing for {self.video_id} f{self.frame_id}")
        if self.bbox_x2 <= self.bbox_x1 or self.bbox_y2 <= self.bbox_y1:
            raise ValueError(
                f"impossible bbox for {self.video_id} f{self.frame_id} "
                f"track {self.track_id}: "
                f"({self.bbox_x1}, {self.bbox_y1}, {self.bbox_x2}, {self.bbox_y2})"
            )
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            raise ValueError(
                f"annotation_confidence must be in [0,1], got {self.annotation_confidence}"
            )

    @property
    def bbox(self):
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @property
    def role_index(self) -> int:
        return ROLE_INDEX[self.role]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PersonAnnotation":
        allowed = {f for f in PersonAnnotation.__dataclass_fields__}
        return PersonAnnotation(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class ObjectAnnotation:
    """A cricket object (BALL / STUMPS) in one frame."""
    video_id: str
    frame_id: int
    track_id: int
    object_class: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    visibility: str = "fully_visible"
    annotation_confidence: float = 1.0
    schema_version: str = SCHEMA_VERSION
    annotator_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.object_class not in OBJECT_INDEX:
            raise ValueError(
                f"object_class must be one of {list(OBJECT_CLASSES.values())}, "
                f"got {self.object_class!r}"
            )
        self.visibility = Visibility(self.visibility)
        if self.frame_id < 0 or self.track_id < 0:
            raise ValueError("frame_id and track_id must be >= 0")
        if self.bbox_x2 <= self.bbox_x1 or self.bbox_y2 <= self.bbox_y1:
            raise ValueError(
                f"impossible bbox for {self.video_id} f{self.frame_id} "
                f"object {self.object_class}"
            )
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            raise ValueError("annotation_confidence must be in [0,1]")

    @property
    def class_index(self) -> int:
        return OBJECT_INDEX[self.object_class]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectAnnotation":
        allowed = set(ObjectAnnotation.__dataclass_fields__)
        return ObjectAnnotation(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class PhaseAnnotation:
    """Bowling phase for one tracked person in one frame."""
    video_id: str
    frame_id: int
    track_id: int
    phase: str
    pose_confidence: Optional[float] = None
    annotation_confidence: float = 1.0
    uncertain: bool = False
    schema_version: str = SCHEMA_VERSION
    annotator_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.phase = Phase(self.phase)
        if self.frame_id < 0 or self.track_id < 0:
            raise ValueError("frame_id and track_id must be >= 0")
        if self.pose_confidence is not None and not 0.0 <= float(self.pose_confidence) <= 1.0:
            raise ValueError("pose_confidence must be in [0,1] or None")
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            raise ValueError("annotation_confidence must be in [0,1]")

    @property
    def phase_index(self) -> int:
        return PHASE_INDEX[self.phase]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhaseAnnotation":
        allowed = set(PhaseAnnotation.__dataclass_fields__)
        return PhaseAnnotation(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class DeliveryAnnotation:
    """A single delivery: the delivery window plus its key-phase frames.

    Every key frame is optional (``None``) -- an annotator may legitimately not
    be able to identify, say, the gather frame in a cropped clip.  QC reports
    gaps; it never invents them.
    """
    delivery_id: str
    video_id: str
    bowler_track_id: int
    start_frame: int
    runup_start_frame: Optional[int] = None
    gather_frame: Optional[int] = None
    delivery_stride_frame: Optional[int] = None
    front_foot_contact_frame: Optional[int] = None
    release_frame: Optional[int] = None
    followthrough_end_frame: Optional[int] = None
    end_frame: int = 0
    annotation_confidence: float = 1.0
    annotator_id: str = ""
    notes: str = ""
    uncertain: bool = False
    schema_version: str = SCHEMA_VERSION
    annotation_date: str = ""

    def __post_init__(self) -> None:
        if not self.annotator_id:
            raise ValueError(
                f"annotator_id is required on delivery {self.delivery_id!r}; "
                "ground truth must be attributable to a human."
            )
        if self.bowler_track_id < 0:
            raise ValueError("bowler_track_id must be >= 0")
        if self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            raise ValueError("annotation_confidence must be in [0,1]")
        if not self.annotation_date:
            self.annotation_date = _utc_now()
        if self.end_frame < self.start_frame:
            self.end_frame = self.start_frame

    def key_frames(self) -> Dict[str, Optional[int]]:
        """Ordered-phase anchors as ``{phase: frame or None}``.

        The brief defines seven ordered phases but supplies six key-frame
        fields.  ``APPROACH`` therefore has **no** dedicated field: it is
        implied to fall between ``runup_start_frame`` and ``gather_frame`` and
        is reported as ``None`` rather than being back-filled from a neighbour.
        QC checks monotonicity over the non-``None`` anchors only.
        """
        return {
            "RUN_UP": self.runup_start_frame,
            "APPROACH": None,
            "GATHER": self.gather_frame,
            "DELIVERY_STRIDE": self.delivery_stride_frame,
            "FRONT_FOOT_CONTACT": self.front_foot_contact_frame,
            "RELEASE": self.release_frame,
            "FOLLOW_THROUGH": self.followthrough_end_frame,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeliveryAnnotation":
        allowed = set(DeliveryAnnotation.__dataclass_fields__)
        return DeliveryAnnotation(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class SceneAnnotation:
    """Per-video scene context. Each field is either an explicit value or None
    (not recorded). Never a guess."""
    video_id: str
    pitch_region: Optional[List[float]] = None
    bowling_crease: Optional[List[float]] = None
    popping_crease: Optional[List[float]] = None
    batting_crease: Optional[List[float]] = None
    stumps: Optional[List[float]] = None
    bowler_runup_region: Optional[List[float]] = None
    striker_region: Optional[List[float]] = None
    camera_orientation: str = "unknown"
    schema_version: str = SCHEMA_VERSION
    annotator_id: str = ""
    annotation_confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.camera_orientation = CameraView(self.camera_orientation)
        for name in (
            "pitch_region", "bowling_crease", "popping_crease", "batting_crease",
            "stumps", "bowler_runup_region", "striker_region",
        ):
            val = getattr(self, name)
            if val is None:
                continue
            if not isinstance(val, (list, tuple)) or len(val) != 4:
                raise ValueError(f"{name} must be [x1, y1, x2, y2] or None, got {val!r}")
            if val[2] <= val[0] or val[3] <= val[1]:
                raise ValueError(f"{name} has an impossible box: {val!r}")
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            raise ValueError("annotation_confidence must be in [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SceneAnnotation":
        allowed = set(SceneAnnotation.__dataclass_fields__)
        return SceneAnnotation(**{k: v for k, v in d.items() if k in allowed})


# --------------------------------------------------------------------------- #
# Deterministic JSON I/O
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _default(o: Any):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps(obj: Any) -> str:
    """Stable JSON: sorted keys are NOT used (key order is semantic here) but
    the payload is fully ordered and re-serialisation is byte-stable."""
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_default)


def _atomic_write(path: str, text: str) -> None:
    """Write via a temp file + replace so a crash cannot corrupt a save."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _annot_path(subdir: str, video_id: str) -> str:
    return os.path.join(ANNOTATIONS_DIR, subdir, f"{video_id}.json")


def save_person_annotations(video_id: str, rows: List[PersonAnnotation]) -> str:
    path = _annot_path("roles", video_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "kind": "person_roles",
        "records": [r.to_dict() for r in sorted(rows, key=_person_key)],
    }
    _atomic_write(path, dumps(payload))
    return path


def load_person_annotations(video_id: str) -> List[PersonAnnotation]:
    path = _annot_path("roles", video_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [PersonAnnotation.from_dict(d) for d in payload.get("records", [])]


def save_object_annotations(video_id: str, rows: List[ObjectAnnotation]) -> str:
    path = _annot_path("objects", video_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "kind": "objects",
        "records": [r.to_dict() for r in sorted(
            rows, key=lambda r: (r.frame_id, r.track_id, r.object_class))],
    }
    _atomic_write(path, dumps(payload))
    return path


def load_object_annotations(video_id: str) -> List[ObjectAnnotation]:
    path = _annot_path("objects", video_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [ObjectAnnotation.from_dict(d) for d in payload.get("records", [])]


def save_phase_annotations(video_id: str, rows: List[PhaseAnnotation]) -> str:
    path = _annot_path("bowling_phases", video_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "kind": "bowling_phases",
        "records": [r.to_dict() for r in sorted(
            rows, key=lambda r: (r.frame_id, r.track_id))],
    }
    _atomic_write(path, dumps(payload))
    return path


def load_phase_annotations(video_id: str) -> List[PhaseAnnotation]:
    path = _annot_path("bowling_phases", video_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [PhaseAnnotation.from_dict(d) for d in payload.get("records", [])]


def save_deliveries(video_id: str, rows: List[DeliveryAnnotation]) -> str:
    path = _annot_path("deliveries", video_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "kind": "deliveries",
        "records": [r.to_dict() for r in sorted(rows, key=lambda d: d.delivery_id)],
    }
    _atomic_write(path, dumps(payload))
    return path


def load_deliveries(video_id: str) -> List[DeliveryAnnotation]:
    path = _annot_path("deliveries", video_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [DeliveryAnnotation.from_dict(d) for d in payload.get("records", [])]


def save_scene(video_id: str, scene: SceneAnnotation) -> str:
    path = _annot_path("scene", video_id)
    payload = {"schema_version": SCHEMA_VERSION, "kind": "scene", **scene.to_dict()}
    _atomic_write(path, dumps(payload))
    return path


def load_scene(video_id: str) -> Optional[SceneAnnotation]:
    path = _annot_path("scene", video_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return SceneAnnotation.from_dict(payload)


# --------------------------------------------------------------------------- #
# Inventory helpers
# --------------------------------------------------------------------------- #

def _person_key(r: PersonAnnotation):
    return (r.frame_id, r.track_id)


def all_annotation_video_ids() -> List[str]:
    """Every video_id that has at least one annotation file of any kind."""
    ids = set()
    for sub in ("roles", "bowling_phases", "deliveries", "scene", "objects"):
        d = os.path.join(ANNOTATIONS_DIR, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                ids.add(fn[:-5])
    return sorted(ids)


def all_delivery_annotations() -> List[DeliveryAnnotation]:
    out: List[DeliveryAnnotation] = []
    for vid in all_annotation_video_ids():
        out.extend(load_deliveries(vid))
    return out


def all_person_annotations() -> List[PersonAnnotation]:
    out: List[PersonAnnotation] = []
    for vid in all_annotation_video_ids():
        out.extend(load_person_annotations(vid))
    return out


def all_phase_annotations() -> List[PhaseAnnotation]:
    out: List[PhaseAnnotation] = []
    for vid in all_annotation_video_ids():
        out.extend(load_phase_annotations(vid))
    return out


# --------------------------------------------------------------------------- #
# Schema description (emitted by scripts/describe_annotation_schema.py)
# --------------------------------------------------------------------------- #

def describe() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "player_roles": PLAYER_ROLES,
        "object_classes": OBJECT_CLASSES,
        "bowling_phases": BOWLING_PHASES,
        "ordered_phase_sequence": ORDERED_PHASE_SEQUENCE,
        "visibility_values": VISIBILITY_VALUES,
        "scene_keys": SCENE_KEYS,
        "review_status_values": REVIEW_STATUS_VALUES,
        "annotation_status_values": ANNOTATION_STATUS_VALUES,
        "camera_views": CAMERA_VIEWS,
        "bowling_sides": BOWLING_SIDES,
        "yolo_class_map": YOLO_CLASS_MAP,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
