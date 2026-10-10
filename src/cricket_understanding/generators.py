"""
Phase 7 -- deterministic training-data generators.

Five datasets, one shared provenance rule.

Every emitted sample writes a **manifest row** naming:
``source video, source frame, source track, label, transformation, split``.
That row is what makes a generated dataset reproducible and auditable: given the
row you can get back to the exact human annotation that produced it.

Datasets
--------
======================  ==========================================  ==============
name                    task                                        human label
======================  ==========================================  ==============
``det``                 Model A: object detection (YOLO)             roles+objects
``role``                Model B: cricket role classification          roles
``track_seq``           track-sequence / motion context              roles
``action``              Model C: temporal bowling action             phases
``delivery``            delivery-phase (per delivery)                deliveries
======================  ==========================================  ==============

Integrity rules
---------------
* The ONLY accepted label source is ``data/cricket_understanding/annotations/``.
  ``data/bowler_batsman_dataset/auto_labeled`` and
  ``data/cricket_ball_dataset/auto_labeled`` are model-derived and are never
  read (see ``evaluation/cricket_training_current_state.md`` W4).
* A generator with zero human annotations produces an **empty** dataset and
  says so. It never falls back to a heuristic label.
* Every generator is deterministic: fixed ordering, fixed seed, no sampling.
* ``--dry-run`` performs the full bookkeeping and writes the manifests while
  skipping pixel extraction, so the dataset *contract* can be validated on a
  machine with no GPU and no torch.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import schema, splits as splits_mod

GENERATOR_VERSION = "cricket_understanding_gen_v1"
EXTRACTED = schema.p("extracted")
RANDOM_STATE = 42

#: Human annotation root. Nothing else is ever a label source.
LABEL_ROOT = schema.ANNOTATIONS_DIR

#: Model-derived trees that must never be used as labels. Present so the
#: refusal is explicit and testable rather than an accident of file layout.
_REPO_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
FORBIDDEN_LABEL_ROOTS = (
    os.path.join(_REPO_DATA_DIR, "bowler_batsman_dataset", "auto_labeled"),
    os.path.join(_REPO_DATA_DIR, "cricket_ball_dataset", "auto_labeled"),
)


def assert_human_label_source(root: str) -> None:
    """Refuse to generate from a model-derived label tree.

    This is a hard guard, not a convention: if a caller points the generators at
    an auto-labelled tree, generation aborts instead of producing training data
    that would only re-learn the current model's mistakes.
    """
    norm = os.path.abspath(root)
    for bad in FORBIDDEN_LABEL_ROOTS:
        bad_n = os.path.abspath(bad)
        if norm == bad_n or norm.startswith(bad_n + os.sep):
            raise GeneratorError(
                f"REFUSING to read labels from {norm}: this is a MODEL-DERIVED "
                "tree (auto_labeled). Training on it would re-learn PaceAI's own "
                "predictions. Ground truth must come from "
                f"{LABEL_ROOT} and be annotated by a human.")


class GeneratorError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

@dataclass
class ProvenanceRow:
    """One generated sample's lineage. Required by the brief, non-optional."""
    dataset: str
    sample_id: str
    source_video_id: str
    source_video_path: str
    source_frame: Optional[int]
    source_track_id: Optional[int]
    label: str
    label_index: Optional[int]
    transformation: str
    split: str
    annotation_confidence: float = 1.0
    visibility: str = "fully_visible"
    uncertain: bool = False
    annotator_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ManifestWriter:
    """Accumulates provenance rows and writes JSONL + a JSON summary."""

    def __init__(self, dataset: str, out_dir: str, dry_run: bool = False):
        self.dataset = dataset
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.rows: List[ProvenanceRow] = []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, row: ProvenanceRow) -> None:
        if row.dataset != self.dataset:
            raise GeneratorError(
                f"row dataset {row.dataset!r} != writer {self.dataset!r}")
        if row.split not in ("train", "val", "test", "unassigned"):
            raise GeneratorError(f"bad split {row.split!r}")
        self.rows.append(row)

    def write(self, notes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        by_split: Dict[str, int] = {}
        by_label: Dict[str, int] = {}
        for r in self.rows:
            by_split[r.split] = by_split.get(r.split, 0) + 1
            by_label[r.label] = by_label.get(r.label, 0) + 1
        path = os.path.join(self.out_dir, "manifest.jsonl")
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for r in sorted(self.rows, key=lambda r: (r.source_video_id,
                                                      r.source_frame or -1,
                                                      r.source_track_id or -1,
                                                      r.sample_id)):
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

        summary = {
            "generator_version": GENERATOR_VERSION,
            "schema_version": schema.SCHEMA_VERSION,
            "dataset": self.dataset,
            "n_samples": len(self.rows),
            "n_by_split": dict(sorted(by_split.items())),
            "n_by_label": dict(sorted(by_label.items())),
            "label_source": LABEL_ROOT,
            "label_source_is_human": True,
            "forbidden_label_roots": list(FORBIDDEN_LABEL_ROOTS),
            "seed": RANDOM_STATE,
            "dry_run": self.dry_run,
            "manifest_path": path,
            "notes": notes or {},
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(schema.dumps(summary))
        return summary


# --------------------------------------------------------------------------- #
# Shared context
# --------------------------------------------------------------------------- #

@dataclass
class VideoContext:
    video_id: str
    path: str
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    n_frames: Optional[int]
    split: str
    people: List[schema.PersonAnnotation] = field(default_factory=list)
    objects: List[schema.ObjectAnnotation] = field(default_factory=list)
    phases: List[schema.PhaseAnnotation] = field(default_factory=list)
    deliveries: List[schema.DeliveryAnnotation] = field(default_factory=list)
    scene: Optional[schema.SceneAnnotation] = None

    def people_at(self, frame: int) -> List[schema.PersonAnnotation]:
        return [p for p in self.people if p.frame_id == frame]

    def objects_at(self, frame: int) -> List[schema.ObjectAnnotation]:
        return [o for o in self.objects if o.frame_id == frame]

    def phase_of(self, frame: int, track_id: int) -> Optional[schema.PhaseAnnotation]:
        for p in self.phases:
            if p.frame_id == frame and p.track_id == track_id:
                return p
        return None

    def has_labels(self) -> bool:
        return bool(self.people or self.objects or self.phases or self.deliveries)


def load_contexts(records: Sequence, split_map: Optional[Dict[str, List[str]]] = None,
                  require_labels: bool = True,
                  annotation_root: Optional[str] = None) -> List[VideoContext]:
    """Load annotation + registry metadata for every video that has labels.

    ``annotation_root`` defaults to the real human annotation tree.  It exists
    so tests can point at an isolated directory; production callers leave it
    ``None`` and therefore can only ever read human annotations from
    :data:`LABEL_ROOT`.
    """
    split_map = split_map if split_map is not None else (splits_mod.load_splits() or {})
    out: List[VideoContext] = []
    for rec in records:
        vid = rec.video_id
        ctx = VideoContext(
            video_id=vid, path=rec.path, fps=rec.fps, width=rec.width,
            height=rec.height, n_frames=rec.frame_count,
            split=splits_mod.split_of(vid, split_map) or "unassigned")
        if annotation_root is None:
            ctx.people = schema.load_person_annotations(vid)
            ctx.objects = schema.load_object_annotations(vid)
            ctx.phases = schema.load_phase_annotations(vid)
            ctx.deliveries = schema.load_deliveries(vid)
            ctx.scene = schema.load_scene(vid)
        else:
            ctx.people = _load_kind(annotation_root, "roles", vid,
                                    schema.PersonAnnotation.from_dict)
            ctx.objects = _load_kind(annotation_root, "objects", vid,
                                     schema.ObjectAnnotation.from_dict)
            ctx.phases = _load_kind(annotation_root, "bowling_phases", vid,
                                    schema.PhaseAnnotation.from_dict)
            ctx.deliveries = _load_kind(annotation_root, "deliveries", vid,
                                        schema.DeliveryAnnotation.from_dict)
            scene_path = os.path.join(annotation_root, "scene", f"{vid}.json")
            if os.path.exists(scene_path):
                with open(scene_path, encoding="utf-8") as fh:
                    ctx.scene = schema.SceneAnnotation.from_dict(json.load(fh))
        if require_labels and not ctx.has_labels():
            continue
        out.append(ctx)
    return out


def _load_kind(root: str, kind: str, video_id: str, ctor):
    path = os.path.join(root, kind, f"{video_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    out = []
    for row in payload.get("records", []):
        try:
            out.append(ctor(row))
        except (TypeError, ValueError):
            # An invalid row is skipped, never repaired; QC reports it later.
            continue
    return out


# --------------------------------------------------------------------------- #
# Deterministic frame extraction
# --------------------------------------------------------------------------- #

def frame_filename(video_id: str, frame_id: int) -> str:
    return f"{video_id}_f{frame_id:08d}.jpg"


def _jpeg_params() -> List[int]:
    import cv2
    return [int(cv2.IMWRITE_JPEG_QUALITY), 95]


def extract_frame(path: str, frame_id: int, dest_jpg: str,
                  src_size: Optional[Tuple[int, int]] = None,
                  imsize: Optional[int] = None) -> Tuple[bool, str]:
    """Write one frame as JPEG. Returns ``(ok, reason)``.

    ``src_size`` records the annotation coordinate space.  If the decoded frame
    is a different size the annotation cannot be trusted, so the frame is
    REJECTED rather than rescaled -- silently rescaling would move every box.
    """
    import cv2
    if imsize:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return False, "frame_unreadable"
        h, w = frame.shape[:2]
        if src_size and (w, h) != tuple(src_size):
            return False, f"size_mismatch:frame={w}x{h},annotation={src_size[0]}x{src_size[1]}"
        if imsize and max(w, h) != imsize:
            s = imsize / max(w, h)
            frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))))
        os.makedirs(os.path.dirname(os.path.abspath(dest_jpg)), exist_ok=True)
        if not cv2.imwrite(dest_jpg, frame, _jpeg_params()):
            return False, "jpeg_write_failed"
        return True, "ok"
    # full-resolution path
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return False, "frame_unreadable"
    if src_size:
        h, w = frame.shape[:2]
        if (w, h) != tuple(src_size):
            return False, f"size_mismatch:frame={w}x{h},annotation={src_size[0]}x{src_size[1]}"
    os.makedirs(os.path.dirname(os.path.abspath(dest_jpg)), exist_ok=True)
    if not cv2.imwrite(dest_jpg, frame, _jpeg_params()):
        return False, "jpeg_write_failed"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Generator 1 -- YOLO object detection (Model A)
# --------------------------------------------------------------------------- #

def yolo_label_line(class_index: int, bbox: Sequence[float],
                    width: float, height: float) -> Optional[str]:
    """Normalised YOLO row, or ``None`` for a degenerate box.

    Boxes outside the frame are CLIPPED (a human box that overshoots the edge
    is a drawing error, not a different object).  A box that becomes empty
    after clipping is dropped and reported, not written as a zero-size row.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    return f"{int(class_index)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def build_yolo_dataset(contexts: Sequence[VideoContext], out_dir: str,
                       imsize: Optional[int] = None, dry_run: bool = False,
                       include_objects: bool = True) -> Dict[str, Any]:
    """Model A.  Layout: ``out/images/{split}/``, ``out/labels/{split}/``."""
    out_dir = os.path.join(EXTRACTED, "det") if out_dir is None else out_dir
    writer = ManifestWriter("det", out_dir, dry_run=dry_run)
    stats = {"frames_written": 0, "frames_skipped": 0, "objects": 0,
             "skipped_reasons": {}}
    rejected: List[ProvenanceRow] = []

    for ctx in contexts:
        frames = sorted({p.frame_id for p in ctx.people})
        for f in frames:
            rows: List[str] = []
            for p in ctx.people_at(f):
                line = yolo_label_line(p.role_index, p.bbox, ctx.width, ctx.height)
                if line is None:
                    stats["skipped_reasons"]["degenerate_person_box"] = \
                        stats["skipped_reasons"].get("degenerate_person_box", 0) + 1
                    rejected.append(ProvenanceRow(
                        "det", f"{ctx.video_id}:{f}:{p.track_id}", ctx.video_id, ctx.path,
                        f, p.track_id, p.role, p.role_index,
                        "rejected:degenerate_or_out_of_frame_bbox", ctx.split,
                        p.annotation_confidence, p.visibility, False, p.annotator_id))
                    continue
                rows.append(line)
                stats["objects"] += 1
                writer.add(ProvenanceRow(
                    "det", f"{ctx.video_id}:{f}:{p.track_id}:{p.role_index}",
                    ctx.video_id, ctx.path, f, p.track_id, p.role, p.role_index,
                    "identity:human_bbox", ctx.split, p.annotation_confidence,
                    p.visibility, False, p.annotator_id))
            if include_objects:
                for o in ctx.objects_at(f):
                    line = yolo_label_line(o.class_index,
                                           (o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2),
                                           ctx.width, ctx.height)
                    if line is None:
                        continue
                    rows.append(line)
                    stats["objects"] += 1
                    writer.add(ProvenanceRow(
                        "det", f"{ctx.video_id}:{f}:{o.track_id}:{o.class_index}",
                        ctx.video_id, ctx.path, f, o.track_id, o.object_class,
                        o.class_index, "identity:human_bbox", ctx.split,
                        o.annotation_confidence, o.visibility, False, o.annotator_id))
            if not rows:
                stats["frames_skipped"] += 1
                continue

            split = ctx.split if ctx.split in ("train", "val", "test") else "train"
            rel_img = os.path.join("images", split, frame_filename(ctx.video_id, f))
            rel_lbl = os.path.join("labels", split,
                                   frame_filename(ctx.video_id, f).replace(".jpg", ".txt"))
            if not dry_run:
                ok, reason = extract_frame(ctx.path, f, os.path.join(out_dir, rel_img),
                                           (ctx.width, ctx.height), imsize=imsize)
                if not ok:
                    stats["frames_skipped"] += 1
                    stats["skipped_reasons"][reason] = \
                        stats["skipped_reasons"].get(reason, 0) + 1
                    continue
                os.makedirs(os.path.dirname(os.path.join(out_dir, rel_lbl)), exist_ok=True)
                with open(os.path.join(out_dir, rel_lbl), "w",
                          encoding="utf-8", newline="\n") as fh:
                    fh.write("\n".join(rows) + "\n")
            stats["frames_written"] += 1

    for r in rejected:
        writer.add(r)
    summary = writer.write(notes={
        "class_map": schema.YOLO_CLASS_MAP,
        "imsize": imsize,
        "include_objects": include_objects,
        "frames_skipped": stats["frames_skipped"],
        "skipped_reasons": stats["skipped_reasons"],
        "degenerate_boxes_reported_not_fixed": len(rejected),
    })
    summary.update(stats)
    return summary


# --------------------------------------------------------------------------- #
# Generator 2 -- role classification (Model B, static features)
# --------------------------------------------------------------------------- #

#: Feature order is frozen. Appending is safe; reordering is a breaking change
#: and must bump GENERATOR_VERSION.
ROLE_FEATURE_NAMES: List[str] = [
    "bbox_cx_norm", "bbox_cy_norm", "bbox_w_norm", "bbox_h_norm", "bbox_area_norm",
    "aspect_ratio", "cx_rel_track_mean", "cy_rel_track_mean",
    "track_len_norm", "track_span_frames", "start_frame_norm", "end_frame_norm",
    "mean_speed_body", "max_speed_body", "speed_body_std",
    "net_displacement_body", "path_length_body", "straightness",
    "area_growth", "area_growth_monotonic", "mean_y_fraction", "y_range_norm",
    "x_range_norm", "motion_fraction", "in_central_band_frac",
    "late_decel", "direction_persistence", "late_stride_peak",
    "n_visible_frac", "mean_visibility_score", "uncertain_frac",
    "is_delivery_bowler", "is_delivery_participant",
    "median_height_over_frame", "nearest_neighbour_dist_norm",
    "n_co_persons", "second_order_speed_mean",
]


def _track_features(ctx: VideoContext, p: schema.PersonAnnotation) -> Dict[str, float]:
    """Pure geometry/temporal features from HUMAN boxes only.

    Deliberately excludes anything derived from a PaceAI score, so Model B
    cannot learn to imitate the current heuristic.  (The heuristic's own
    components are used in Phase 11 experiment B, which is a separate,
    explicitly-labelled comparison -- not as a training label here.)
    """
    import numpy as np

    W, H = float(ctx.width or 1), float(ctx.height or 1)
    boxes = sorted((q for q in ctx.people if q.track_id == p.track_id),
                   key=lambda q: q.frame_id)
    n = max(1, ctx.n_frames or 1)
    bx = [(float(q.bbox_x1), float(q.bbox_y1), float(q.bbox_x2), float(q.bbox_y2))
          for q in boxes]
    cx = np.array([(b[0] + b[2]) / 2 for b in bx]) if bx else np.zeros(1)
    cy = np.array([(b[1] + b[3]) / 2 for b in bx]) if bx else np.zeros(1)
    bw = np.array([b[2] - b[0] for b in bx]) if bx else np.ones(1)
    bh = np.array([b[3] - b[1] for b in bx]) if bx else np.ones(1)
    diag = np.sqrt((bw ** 2 + bh ** 2).clip(1e-6))

    steps = np.hypot(np.diff(cx), np.diff(cy)) if len(cx) > 1 else np.zeros(1)
    scales = diag[:-1] if len(diag) > 1 else diag
    body = steps / np.maximum(scales, 1e-6)
    path = float(steps.sum())
    net = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])) if len(cx) > 1 else 0.0
    areas = bw * bh
    if len(areas) >= 2 and abs(areas[-1] - areas[0]) > 1e-6:
        d = np.diff(areas)
        nz = d[d != 0]
        mono = float(np.mean(np.sign(nz) == np.sign(areas[-1] - areas[0]))) if len(nz) else 0.0
        growth = float(np.clip(abs(areas[-1] - areas[0]) / max(float(np.median(areas)), 1e-6),
                               0.0, 2.0))
    else:
        mono, growth = 0.0, 0.0
    k = max(2, int(0.2 * len(body))) if len(body) >= 8 else 0
    decel = 0.0
    if k:
        late, overall = float(np.median(body[-k:])), float(np.median(body))
        if overall > 1e-6 and late < overall * 0.85:
            decel = float(np.clip(1.0 - late / (overall * 0.85), 0.0, 1.0))
    directed = 0.0
    if len(cx) > 1 and steps.sum() > 1e-6:
        dxs, dys = np.diff(cx), np.diff(cy)
        hx, hy = float(dxs.sum()), float(dys.sum())
        hn = float(np.hypot(hx, hy))
        if hn > 1e-6:
            directed = float(np.mean((dxs * hx / hn + dys * hy / hn) /
                                     np.maximum(steps, 1e-6) > 0.5))
    late_peak = 0.0
    if len(steps) >= 10:
        s = steps / np.maximum(scales, 1e-6)
        ov = float(np.median(s)) or 1e-6
        pk = float(s[int(0.6 * len(s)):].max())
        late_peak = float(np.clip((pk / ov - 1.0) / 1.0, 0.0, 1.0))
    vis_score = {"fully_visible": 1.0, "partially_occluded": 0.75,
                 "heavily_occluded": 0.4, "uncertain": 0.3, "not_visible": 0.0}
    vis = [vis_score.get(q.visibility, 0.5) for q in boxes] or [0.5]
    others = [q for q in ctx.people_at(p.frame_id) if q.track_id != p.track_id]
    nn = min((float(np.hypot((o.bbox_x1 + o.bbox_x2) / 2 - (p.bbox_x1 + p.bbox_x2) / 2,
                             (o.bbox_y1 + o.bbox_y2) / 2 - (p.bbox_y1 + p.bbox_y2) / 2))
              for o in others), default=1.0)
    deliv = [d for d in ctx.deliveries if d.bowler_track_id == p.track_id]
    return {
        "bbox_cx_norm": float((p.bbox_x1 + p.bbox_x2) / 2 / W),
        "bbox_cy_norm": float((p.bbox_y1 + p.bbox_y2) / 2 / H),
        "bbox_w_norm": float((p.bbox_x2 - p.bbox_x1) / W),
        "bbox_h_norm": float((p.bbox_y2 - p.bbox_y1) / H),
        "bbox_area_norm": float((p.bbox_x2 - p.bbox_x1) * (p.bbox_y2 - p.bbox_y1) / (W * H)),
        "aspect_ratio": float((p.bbox_y2 - p.bbox_y1) / max(1e-6, p.bbox_x2 - p.bbox_x1)),
        "cx_rel_track_mean": float(np.mean(np.abs(cx - cx.mean()))) if len(cx) > 1 else 0.0,
        "cy_rel_track_mean": float(np.mean(np.abs(cy - cy.mean()))) if len(cy) > 1 else 0.0,
        "track_len_norm": float(len(boxes) / n),
        "track_span_frames": float(boxes[-1].frame_id - boxes[0].frame_id) if boxes else 0.0,
        "start_frame_norm": float(boxes[0].frame_id / n) if boxes else 0.0,
        "end_frame_norm": float(boxes[-1].frame_id / n) if boxes else 0.0,
        "mean_speed_body": float(np.mean(body)),
        "max_speed_body": float(np.max(body)),
        "speed_body_std": float(np.std(body)),
        "net_displacement_body": net / max(float(np.mean(diag)), 1e-6),
        "path_length_body": path / max(float(np.mean(diag)), 1e-6),
        "straightness": float(np.clip(net / path, 0.0, 1.0)) if path > 1e-6 else 0.0,
        "area_growth": growth,
        "area_growth_monotonic": mono,
        "mean_y_fraction": float(np.mean(cy) / H),
        "y_range_norm": float(np.ptp(cy) / H) if len(cy) > 1 else 0.0,
        "x_range_norm": float(np.ptp(cx) / W) if len(cx) > 1 else 0.0,
        "motion_fraction": float(np.tanh(path / max(float(np.hypot(W, H)), 1e-6))),
        "in_central_band_frac": float(np.mean((cx >= 0.15 * W) & (cx <= 0.85 * W))),
        "late_decel": decel,
        "direction_persistence": directed,
        "late_stride_peak": late_peak,
        "n_visible_frac": float(np.mean([q.visibility != "not_visible" for q in boxes])),
        "mean_visibility_score": float(np.mean(vis)),
        "uncertain_frac": float(np.mean([q.visibility == "uncertain" for q in boxes])),
        "is_delivery_bowler": 1.0 if deliv else 0.0,
        "is_delivery_participant": 1.0 if any(
            d.start_frame <= p.frame_id <= d.end_frame for d in ctx.deliveries) else 0.0,
        "median_height_over_frame": float(np.median(bh) / H),
        "nearest_neighbour_dist_norm": float(nn / max(float(np.hypot(W, H)), 1e-6)),
        "n_co_persons": float(len(others)),
        "second_order_speed_mean": float(np.mean(np.abs(np.diff(body)))) if len(body) > 1 else 0.0,
    }


def build_role_dataset(contexts: Sequence[VideoContext], out_dir: str,
                       dry_run: bool = False) -> Dict[str, Any]:
    """Model B, static/temporal per-track features -> role label.

    One row per (video, track): the features are aggregated over the WHOLE
    track, and the label is the **modal** human role on that track.  A track
    whose human labels disagree is skipped and reported, not majority-voted
    into a clean label.
    """
    out_dir = os.path.join(EXTRACTED, "role") if out_dir is None else out_dir
    writer = ManifestWriter("role", out_dir, dry_run=dry_run)
    rows: List[Dict[str, Any]] = []
    skipped_conflicting: List[str] = []
    skipped_low_conf: List[str] = []

    for ctx in contexts:
        by_track: Dict[int, List[schema.PersonAnnotation]] = {}
        for p in ctx.people:
            by_track.setdefault(p.track_id, []).append(p)
        for tid in sorted(by_track):
            people = sorted(by_track[tid], key=lambda q: q.frame_id)
            roles = {}
            for q in people:
                roles[q.role] = roles.get(q.role, 0) + 1
            top = sorted(roles.items(), key=lambda kv: (-kv[1], kv[0]))
            if len(top) > 1 and top[1][1] / sum(roles.values()) > 0.15:
                skipped_conflicting.append(f"{ctx.video_id}:T{tid}")
                writer.add(ProvenanceRow(
                    "role", f"{ctx.video_id}:T{tid}", ctx.video_id, ctx.path, None, tid,
                    "AMBIGUOUS", None, "rejected:conflicting_roles_on_track", ctx.split,
                    annotator_id=people[0].annotator_id,
                    extra={"role_counts": roles}))
                continue
            label = top[0][0]
            conf = float(np_mean_conf(people))
            if conf < 0.5:
                skipped_low_conf.append(f"{ctx.video_id}:T{tid}")
            feats = _track_features(ctx, people[len(people) // 2])
            rows.append({
                "sample_id": f"{ctx.video_id}:T{tid}",
                "video_id": ctx.video_id, "track_id": tid, "split": ctx.split,
                "label": label, "label_index": schema.ROLE_INDEX[label],
                "n_frames_labeled": len(people),
                "annotation_confidence": conf,
                "annotator_id": people[0].annotator_id,
                "features": feats,
            })
            writer.add(ProvenanceRow(
                "role", f"{ctx.video_id}:T{tid}", ctx.video_id, ctx.path, None, tid,
                label, schema.ROLE_INDEX[label], "aggregate:whole_track", ctx.split,
                conf, people[len(people) // 2].visibility, False, people[0].annotator_id,
                extra={"n_frames_labeled": len(people)}))

    summary = writer.write(notes={
        "feature_names": ROLE_FEATURE_NAMES,
        "label_policy": "modal human role per track; ambiguous tracks REJECTED",
        "skipped_conflicting_tracks": skipped_conflicting,
        "skipped_low_confidence_tracks": skipped_low_conf,
    })
    if not dry_run:
        with open(os.path.join(out_dir, "role_dataset.jsonl"), "w",
                  encoding="utf-8", newline="\n") as fh:
            for r in sorted(rows, key=lambda r: r["sample_id"]):
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        with open(os.path.join(out_dir, "features.json"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(schema.dumps(ROLE_FEATURE_NAMES))
    summary["n_rows"] = len(rows)
    summary["n_skipped_conflicting"] = len(skipped_conflicting)
    return summary


def np_mean_conf(people: Sequence[schema.PersonAnnotation]) -> float:
    return sum(float(q.annotation_confidence) for q in people) / max(1, len(people))


# --------------------------------------------------------------------------- #
# Generator 3 -- track-sequence dataset
# --------------------------------------------------------------------------- #

TRACK_SEQ_FEATURE_NAMES = [
    "cx_norm", "cy_norm", "w_norm", "h_norm", "aspect", "area_norm",
    "dx_norm", "dy_norm", "speed_norm", "accel_norm", "darea_norm",
    "rel_cx", "rel_cy", "frame_idx_norm", "missing",
]


def build_track_sequence_dataset(contexts: Sequence[VideoContext], out_dir: str,
                                 dry_run: bool = False) -> Dict[str, Any]:
    """Per-frame, per-track normalised trajectories.

    A frame with NO human box for a track that is otherwise present is emitted
    with ``missing=1`` and zeroed geometry -- the gap is data, not noise to be
    interpolated away.
    """
    out_dir = os.path.join(EXTRACTED, "track_seq") if out_dir is None else out_dir
    writer = ManifestWriter("track_seq", out_dir, dry_run=dry_run)
    n_seq = n_rows = n_missing = 0

    for ctx in contexts:
        by_track: Dict[int, Dict[int, schema.PersonAnnotation]] = {}
        for p in ctx.people:
            by_track.setdefault(p.track_id, {})[p.frame_id] = p
        for tid in sorted(by_track):
            frames = by_track[tid]
            lo, hi = min(frames), max(frames)
            W, H = float(ctx.width or 1), float(ctx.height or 1)
            norm = float(ctx.n_frames or 1)
            seq: List[Dict[str, Any]] = []
            prev = None
            prev_v = None
            for f in range(lo, hi + 1):
                q = frames.get(f)
                if q is None:
                    n_missing += 1
                    row = {k: 0.0 for k in TRACK_SEQ_FEATURE_NAMES}
                    row["frame_idx_norm"] = f / norm
                    row["missing"] = 1.0
                    seq.append(row)
                    continue
                cx = (q.bbox_x1 + q.bbox_x2) / 2 / W
                cy = (q.bbox_y1 + q.bbox_y2) / 2 / H
                w = (q.bbox_x2 - q.bbox_x1) / W
                h = (q.bbox_y2 - q.bbox_y1) / H
                area = w * h
                v = (cx, cy)
                if prev is None:
                    dx = dy = sp = ac = da = 0.0
                else:
                    dx, dy = cx - prev[0], cy - prev[1]
                    sp = float((dx * dx + dy * dy) ** 0.5)
                    psp = 0.0 if prev_v is None else float(
                        ((prev_v[0] - prev[0]) ** 2 + (prev_v[1] - prev[1]) ** 2) ** 0.5)
                    ac = sp - psp
                    da = area - (prev_v[2] if prev_v else area)
                row = {
                    "cx_norm": cx, "cy_norm": cy, "w_norm": w, "h_norm": h,
                    "aspect": float(h / max(1e-6, w)), "area_norm": area,
                    "dx_norm": dx, "dy_norm": dy, "speed_norm": sp, "accel_norm": ac,
                    "darea_norm": da,
                    "rel_cx": cx - 0.5, "rel_cy": cy - 0.5,
                    "frame_idx_norm": f / norm, "missing": 0.0,
                }
                seq.append(row)
                prev, prev_v = v, (cx, cy, area)
                n_rows += 1
            rec = {
                "video_id": ctx.video_id, "track_id": tid, "split": ctx.split,
                "start_frame": lo, "end_frame": hi, "length": len(seq),
                "roles": {f"{fr}": p.role for fr, p in sorted(frames.items())},
                "sequence": seq,
            }
            if not dry_run:
                d = os.path.join(out_dir, ctx.split if ctx.split != "unassigned" else "train")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{ctx.video_id}_T{tid}.json"), "w",
                          encoding="utf-8", newline="\n") as fh:
                    fh.write(schema.dumps(rec))
            n_seq += 1
            writer.add(ProvenanceRow(
                "track_seq", f"{ctx.video_id}:T{tid}", ctx.video_id, ctx.path, lo, tid,
                "MULTI", None, "normalise_to_frame_size", ctx.split,
                annotator_id=next(iter(frames.values())).annotator_id,
                extra={"start_frame": lo, "end_frame": hi, "length": len(seq),
                       "n_missing_frames": sum(1 for r in seq if r["missing"])}))

    summary = writer.write(notes={
        "feature_names": TRACK_SEQ_FEATURE_NAMES,
        "missing_policy": "gaps emitted with missing=1 and zeroed geometry; "
                          "never interpolated",
        "n_missing_frames": n_missing,
    })
    summary.update({"n_sequences": n_seq, "n_rows": n_rows, "n_missing": n_missing})
    return summary


# --------------------------------------------------------------------------- #
# Generator 4 -- bowling action temporal dataset (Model C)
# --------------------------------------------------------------------------- #

ACTION_FEATURE_NAMES = [
    # normalised, hip-centred pose
    "lsx", "lsy", "rsx", "rsy", "lex", "ley", "rex", "rey",
    "lwrist_x", "lwrist_y", "rwrist_x", "rwrist_y",
    "lank_x", "lank_y", "rank_x", "rank_y", "lknee_x", "lknee_y", "rknee_x", "rknee_y",
    "lhip_x", "lhip_y", "rhip_x", "rhip_y",
    # normalised velocity
    "v_lwrist_x", "v_lwrist_y", "v_rwrist_x", "v_rwrist_y",
    "v_lank_x", "v_lank_y", "v_rank_x", "v_rank_y",
    "v_trunk_x", "v_trunk_y",
    # normalised acceleration
    "a_rwrist_x", "a_rwrist_y", "a_rank_x", "a_rank_y",
    # scalars
    "elbow_extension", "knee_flexion", "trunk_lean", "wrist_height_rel_shoulder",
    "pose_confidence", "pose_missing", "frame_idx_norm",
]

MEDIAPIPE = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}


def _angle(a, b, c) -> float:
    import math
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1) or 1e-9
    n2 = math.hypot(*v2) or 1e-9
    cs = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cs))


def pose_features(landmarks, frame_idx: float, pose_conf: float,
                  missing: bool) -> Dict[str, float]:
    """Hip-centred, frame-normalised pose features.

    A missing observation returns an all-zero vector with ``pose_missing=1`` and
    ``pose_confidence=0.0``.  It is NEVER back-filled from the neighbouring
    frame, and never silently dropped from the window.
    """
    import numpy as np
    if missing or landmarks is None:
        row = {k: 0.0 for k in ACTION_FEATURE_NAMES}
        row["pose_missing"] = 1.0
        row["pose_confidence"] = 0.0
        row["frame_idx_norm"] = frame_idx
        return row
    lm = np.asarray(landmarks, dtype=float)
    lh, rh = lm[MEDIAPIPE["l_hip"]], lm[MEDIAPIPE["r_hip"]]
    mid = (lh + rh) / 2.0
    scale = float(np.hypot(lh[0] - rh[0], lh[1] - rh[1])) or 0.25
    def n(key):
        i = MEDIAPIPE[key]
        return ((lm[i, 0] - mid[0]) / scale, (lm[i, 1] - mid[1]) / scale)
    ls, rs, le, re_ = n("l_shoulder"), n("r_shoulder"), n("l_elbow"), n("r_elbow")
    lw, rw, la, ra = n("l_wrist"), n("r_wrist"), n("l_ankle"), n("r_ankle")
    lk, rk, lh_, rh_ = n("l_knee"), n("r_knee"), n("l_hip"), n("r_hip")
    row = {
        "lsx": ls[0], "lsy": ls[1], "rsx": rs[0], "rsy": rs[1],
        "lex": le[0], "ley": le[1], "rex": re_[0], "rey": re_[1],
        "lwrist_x": lw[0], "lwrist_y": lw[1], "rwrist_x": rw[0], "rwrist_y": rw[1],
        "lank_x": la[0], "lank_y": la[1], "rank_x": ra[0], "rank_y": ra[1],
        "lknee_x": lk[0], "lknee_y": lk[1], "rknee_x": rk[0], "rknee_y": rk[1],
        "lhip_x": lh_[0], "lhip_y": lh_[1], "rhip_x": rh_[0], "rhip_y": rh_[1],
        "elbow_extension": float(np.clip((180.0 - _angle(ls, le, lw)) / 180.0, 0.0, 1.0)),
        "knee_flexion": float(np.clip((180.0 - _angle(lh_, lk, la)) / 180.0, 0.0, 1.0)),
        "trunk_lean": float(np.clip(1.0 - (float(rs[1]) + float(ls[1])) / 2.0, -2.0, 2.0)),
        "wrist_height_rel_shoulder": float(rw[1] - rs[1]),
        "pose_confidence": float(pose_conf),
        "pose_missing": 0.0,
        "frame_idx_norm": frame_idx,
    }
    for k in ACTION_FEATURE_NAMES:
        row.setdefault(k, 0.0)
    return row


def _differentiate(seq: List[Dict[str, float]], keys: Sequence[str],
                   prefix: str) -> List[Dict[str, float]]:
    """Central-difference velocity/acceleration over a (possibly gappy) window.

    A step that crosses a ``pose_missing`` frame is emitted as 0.0 rather than
    computed across the gap -- we will not invent motion we did not observe.
    """
    out: List[Dict[str, float]] = []
    n = len(seq)
    for i in range(n):
        cur, prev, nxt = seq[i], seq[i - 1] if i > 0 else None, seq[i + 1] if i < n - 1 else None
        row: Dict[str, float] = {}
        for k in keys:
            if cur.get("pose_missing"):
                row[f"{prefix}_{k}"] = 0.0
                continue
            if prev is None or nxt is None or prev.get("pose_missing") or nxt.get("pose_missing"):
                row[f"{prefix}_{k}"] = 0.0
                continue
            row[f"{prefix}_{k}"] = (nxt[k] - prev[k]) / 2.0
        out.append(row)
    return out


def build_action_dataset(contexts: Sequence[VideoContext], out_dir: str,
                         dry_run: bool = False) -> Dict[str, Any]:
    """Model C. Requires Phase 8 pose sequences; without them this emits an
    EMPTY dataset and reports why, rather than substituting tracking boxes."""
    out_dir = os.path.join(EXTRACTED, "action") if out_dir is None else out_dir
    writer = ManifestWriter("action", out_dir, dry_run=dry_run)
    n_win = n_frames = 0
    reasons: Dict[str, int] = {}

    for ctx in contexts:
        from . import pose_seq as pose_seq_mod
        for d in ctx.deliveries:
            seq = pose_seq_mod.load_sequence(ctx.video_id, d.delivery_id)
            if not seq:
                reasons["no_pose_sequence"] = reasons.get("no_pose_sequence", 0) + 1
                continue
            by_frame = {s["frame_id"]: s for s in seq}
            lo, hi = d.start_frame, d.end_frame
            if hi <= lo:
                reasons["empty_window"] = reasons.get("empty_window", 0) + 1
                continue
            win: List[Dict[str, Any]] = []
            for f in range(lo, hi + 1):
                rec = by_frame.get(f)
                feats = pose_features(
                    None if rec is None else rec["keypoints"],
                    f / max(1.0, float(hi - lo + 1)),
                    0.0 if rec is None else float(rec.get("pose_confidence") or 0.0),
                    missing=rec is None)
                label = None
                for ph in ctx.phases:
                    if ph.frame_id == f and ph.track_id == d.bowler_track_id:
                        label = ph
                        break
                win.append({"frame_id": f, "features": feats,
                            "phase": label.phase if label else None,
                            "phase_uncertain": bool(label.uncertain) if label else False,
                            "pose_missing": rec is None})
                n_frames += 1
            filled = _differentiate([w["features"] for w in win],
                                    ["lwrist_x", "lwrist_y", "rwrist_x", "rwrist_y",
                                     "lank_x", "lank_y", "rank_x", "rank_y"], "v")
            filled2 = _differentiate([w["features"] for w in win],
                                     ["rwrist_x", "rwrist_y", "rank_x", "rank_y"], "a")
            for i in range(len(win)):
                win[i]["features"].update(filled[i])
                win[i]["features"].update(filled2[i])
            trunk = [{"trunk_x": w["features"]["lhip_x"], "trunk_y": w["features"]["lhip_y"]}
                     for w in win]
            for i, d2 in enumerate(_differentiate(trunk, ["trunk_x", "trunk_y"], "v")):
                win[i]["features"].update(d2)

            n_lab = sum(1 for w in win if w["phase"])
            if n_lab == 0:
                reasons["window_without_phase_labels"] = \
                    reasons.get("window_without_phase_labels", 0) + 1
                continue
            sid = f"{ctx.video_id}_{d.delivery_id}_T{d.bowler_track_id}"
            if not dry_run:
                os.makedirs(os.path.join(
                    out_dir, ctx.split if ctx.split != "unassigned" else "train"), exist_ok=True)
                with open(os.path.join(
                        out_dir, ctx.split if ctx.split != "unassigned" else "train",
                        f"{sid}.json"), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(schema.dumps({
                        "video_id": ctx.video_id, "delivery_id": d.delivery_id,
                        "track_id": d.bowler_track_id, "split": ctx.split,
                        "feature_names": ACTION_FEATURE_NAMES,
                        "start_frame": lo, "end_frame": hi, "length": len(win),
                        "n_frames_with_phase_label": n_lab,
                        "n_frames_missing_pose": sum(1 for w in win if w["pose_missing"]),
                        "windows": win,
                    }))
            n_win += 1
            writer.add(ProvenanceRow(
                "action", sid, ctx.video_id, ctx.path, lo, d.bowler_track_id,
                d.delivery_id, None, "pose_window:hip_centred+diff", ctx.split,
                d.annotation_confidence, uncertain=d.uncertain, annotator_id=d.annotator_id,
                extra={"length": len(win), "n_frames_with_phase_label": n_lab,
                       "n_frames_missing_pose": sum(1 for w in win if w["pose_missing"])}))

    summary = writer.write(notes={
        "feature_names": ACTION_FEATURE_NAMES,
        "requires": "Phase 8 pose sequences (src/cricket_understanding/pose_seq.py)",
        "skipped_reasons": reasons,
        "missing_pose_policy": "pose_missing=1 with zeroed features; never "
                               "interpolated, never imputed",
    })
    summary.update({"n_windows": n_win, "n_frames": n_frames, "skipped": reasons})
    return summary


# --------------------------------------------------------------------------- #
# Generator 5 -- delivery-phase dataset
# --------------------------------------------------------------------------- #

def build_delivery_dataset(contexts: Sequence[VideoContext], out_dir: str,
                           dry_run: bool = False) -> Dict[str, Any]:
    """One row per human-annotated delivery: the key-frame table + duration
    features. Rows with a phase-order violation are REJECTED, not repaired."""
    out_dir = os.path.join(EXTRACTED, "delivery") if out_dir is None else out_dir
    writer = ManifestWriter("delivery", out_dir, dry_run=dry_run)
    rows: List[Dict[str, Any]] = []
    rejected: List[str] = []

    for ctx in contexts:
        for d in ctx.deliveries:
            kf = d.key_frames()
            present = [(p, f) for p, f in kf.items()
                       if f is not None and p in schema.ORDERED_PHASE_SEQUENCE]
            bad = [(pa, a, pb, b) for (pa, a), (pb, b) in zip(present, present[1:])
                   if b < a]
            if bad:
                rejected.append(f"{ctx.video_id}/{d.delivery_id}")
                writer.add(ProvenanceRow(
                    "delivery", f"{ctx.video_id}/{d.delivery_id}", ctx.video_id, ctx.path,
                    d.start_frame, d.bowler_track_id, d.delivery_id, None,
                    "rejected:phase_order_violation", ctx.split,
                    d.annotation_confidence, uncertain=True, annotator_id=d.annotator_id,
                    extra={"violations": [list(v) for v in bad],
                           "detail": [f"{pb} frame {b} is BEFORE {pa} frame {a}"
                                      for pa, a, pb, b in bad]}))
                continue
            dur = float(d.end_frame - d.start_frame)
            rows.append({
                "sample_id": f"{ctx.video_id}/{d.delivery_id}",
                "video_id": ctx.video_id, "delivery_id": d.delivery_id,
                "bowler_track_id": d.bowler_track_id, "split": ctx.split,
                "start_frame": d.start_frame, "end_frame": d.end_frame,
                "window_len_frames": int(d.end_frame - d.start_frame + 1),
                "window_len_sec": (dur / ctx.fps) if ctx.fps else None,
                "key_frames": kf,
                "relative_key_frames": {
                    p: (None if f is None else (f - d.start_frame) / max(1.0, dur))
                    for p, f in kf.items()},
                "n_key_frames_present": len(present),
                "annotation_confidence": d.annotation_confidence,
                "uncertain": d.uncertain, "annotator_id": d.annotator_id,
                "notes": d.notes,
            })
            writer.add(ProvenanceRow(
                "delivery", f"{ctx.video_id}/{d.delivery_id}", ctx.video_id, ctx.path,
                d.start_frame, d.bowler_track_id, d.delivery_id, None,
                "identity:human_key_frames", ctx.split, d.annotation_confidence,
                uncertain=d.uncertain, annotator_id=d.annotator_id,
                extra={"n_key_frames_present": len(present)}))

    summary = writer.write(notes={
        "ordering_rule": " <= ".join(schema.ORDERED_PHASE_SEQUENCE),
        "rejected_phase_order_violations": rejected,
        "policy": "violations reported and excluded; NEVER auto-corrected",
    })
    if not dry_run:
        with open(os.path.join(out_dir, "deliveries.jsonl"), "w",
                  encoding="utf-8", newline="\n") as fh:
            for r in sorted(rows, key=lambda r: r["sample_id"]):
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update({"n_rows": len(rows), "n_rejected": len(rejected)})
    return summary


# --------------------------------------------------------------------------- #

GENERATORS = {
    "det": build_yolo_dataset,
    "role": build_role_dataset,
    "track_seq": build_track_sequence_dataset,
    "action": build_action_dataset,
    "delivery": build_delivery_dataset,
}
