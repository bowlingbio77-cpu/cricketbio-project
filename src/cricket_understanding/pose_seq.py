"""
Phase 8 -- pose-sequence extraction for every annotated bowler track.

Design constraints
------------------
* Reuses the **existing** pose pipeline (``src.pose_estimation.PoseEstimator``,
  MediaPipe ``pose_landmarker_heavy.task``) rather than replacing it.
* Runs pose on the **human-annotated bowler crop**, so the sequence belongs to
  the locked, human-identified bowler -- not to "the largest person".
* **Missing pose is data.**  When MediaPipe returns no pose for a frame, an
  explicit record is written with ``keypoints=null``,
  ``keypoint_visibility=null`` and ``observed=false``.  The frame is NOT
  dropped, NOT interpolated, and NOT back-filled.
* Per-keypoint visibility is stored, so a low-confidence wrist can be excluded
  downstream instead of silently steering a biomechanical angle.
* ``pose_confidence`` is the mean visibility of the observed landmarks -- an
  honest, reproducible proxy.  It is labelled a proxy, not a calibrated
  probability.

Output: ``data/cricket_understanding/extracted/pose/<video_id>/<delivery_id>.json``
plus a per-video index, both deterministic and content-addressed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from . import schema

POSE_DIR = schema.p("extracted", "pose")
POSE_EXTRACTOR_VERSION = "cricket_understanding_pose_v1"

#: A landmark below this visibility is stored but marked unreliable.  Downstream
#: code can exclude it; nothing is dropped here.
LOW_VISIBILITY = 0.5


def _crop_from_human_boxes(boxes: Dict[int, Sequence[float]], frame: int,
                           shape) -> Optional[tuple]:
    """Crop window from the HUMAN box for a frame, padded generously.

    Falls back to the nearest available human box within a small gap so a one
    or two-frame annotation slip does not silently drop a pose; the fallback is
    recorded in the output so it is never mistaken for a direct measurement.
    """
    import numpy as np
    h, w = shape[:2]
    b = boxes.get(frame)
    used_frame = frame
    if b is None:
        near = [f for f in boxes if abs(f - frame) <= 3]
        if not near:
            return None
        used_frame = min(near, key=lambda f: abs(f - frame))
        b = boxes[used_frame]
    x1, y1, x2, y2 = (float(v) for v in b)
    pw, ph = 0.30 * (x2 - x1), 0.30 * (y2 - y1)
    x1, x2 = max(0, int(x1 - pw)), min(w, int(x2 + pw))
    y1, y2 = max(0, int(y1 - ph)), min(h, int(y2 + ph))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return (x1, y1, x2, y2), used_frame


def extract_for_delivery(video_path: str, video_id: str, delivery, boxes_by_frame,
                         shape_fn, target_fps: Optional[float] = None,
                         context_size: str = "bowler_crop",
                         require_pose: bool = True) -> Dict[str, Any]:
    """Run the existing pose pipeline over one delivery window.

    ``boxes_by_frame`` is the human-annotated ``frame -> bbox`` map for the
    DELIVERY'S BOWLER TRACK only.  ``shape_fn(frame_id) -> np.ndarray|None``
    returns the decoded full frame.
    """
    import numpy as np
    from .. import config as paceai_config
    from .. import pose_estimation  # src.pose_estimation -- the existing pipeline

    estimator = None
    records: List[Dict[str, Any]] = []
    n_observed = n_missing = 0
    vis_sum = 0.0

    lo = delivery.start_frame
    hi = delivery.end_frame
    if hi < lo:
        lo, hi = hi, lo

    # MediaPipe VIDEO mode needs a monotonically increasing timestamp in ms.
    # The existing pipeline feeds 20 fps (src/config.py: TARGET_FPS), so the
    # same clock is used here to stay byte-compatible with it.
    stamp_ms = lambda f: int(round(1000.0 * f / 20.0))  # noqa: E731

    try:
        estimator = pose_estimation.PoseEstimator()
    except Exception as exc:
        return {
            "schema_version": schema.SCHEMA_VERSION,
            "extractor_version": POSE_EXTRACTOR_VERSION,
            "video_id": video_id, "delivery_id": delivery.delivery_id,
            "track_id": delivery.bowler_track_id,
            "error": f"{type(exc).__name__}: {exc}",
            "frames": [], "n_observed": 0, "n_missing": hi - lo + 1,
        }

    try:
        for f in range(lo, hi + 1):
            frame = shape_fn(f)
            base = {
                "video_id": video_id, "delivery_id": delivery.delivery_id,
                "track_id": delivery.bowler_track_id, "frame_id": f,
            }
            if frame is None:
                records.append({**base, "observed": False, "keypoints": None,
                                "keypoint_visibility": None, "pose_confidence": 0.0,
                                "context": "frame_unreadable"})
                n_missing += 1
                continue
            crop_info = _crop_from_human_boxes(boxes_by_frame, f, frame.shape)
            if crop_info is None:
                records.append({**base, "observed": False, "keypoints": None,
                                "keypoint_visibility": None, "pose_confidence": 0.0,
                                "context": "no_human_box_for_crop"})
                n_missing += 1
                continue
            (x1, y1, x2, y2), used_frame = crop_info
            crop = frame[y1:y2, x1:x2]
            try:
                pf = estimator.process_frame(crop, f, stamp_ms(f) / 1000.0)
            except Exception as exc:
                records.append({**base, "observed": False, "keypoints": None,
                                "keypoint_visibility": None, "pose_confidence": 0.0,
                                "context": f"pose_error:{type(exc).__name__}",
                                "crop_bbox": [x1, y1, x2, y2]})
                n_missing += 1
                continue
            if pf is None or pf.landmarks is None or len(pf.landmarks) == 0:
                records.append({**base, "observed": False, "keypoints": None,
                                "keypoint_visibility": None, "pose_confidence": 0.0,
                                "context": "no_person_in_crop", "crop_bbox": [x1, y1, x2, y2]})
                n_missing += 1
                continue
            lm = np.asarray(pf.landmarks, dtype=float)
            kp = [[round(float(v), 6) for v in row[:3]] for row in lm]
            vis = [round(float(row[3]), 6) for row in lm]
            conf = float(np.mean(vis)) if vis else 0.0
            rec_phase = None
            records.append({
                **base, "observed": True, "keypoints": kp,
                "keypoint_visibility": vis, "pose_confidence": round(conf, 6),
                "n_people_in_crop": int(getattr(pf, "n_people", 1)),
                "context": context_size, "crop_bbox": [x1, y1, x2, y2],
                "crop_source_frame": used_frame,
                "low_confidence_landmarks": int(sum(1 for v in vis if v < LOW_VISIBILITY)),
            })
            n_observed += 1
            vis_sum += conf
            del rec_phase
    finally:
        try:
            estimator.close()
        except Exception:
            pass

    return {
        "schema_version": schema.SCHEMA_VERSION,
        "extractor_version": POSE_EXTRACTOR_VERSION,
        "video_id": video_id,
        "delivery_id": delivery.delivery_id,
        "track_id": delivery.bowler_track_id,
        "start_frame": lo, "end_frame": hi,
        "landmark_names": paceai_config.POSE_LANDMARK_NAMES,
        "low_visibility_threshold": LOW_VISIBILITY,
        "pose_confidence_is": "mean landmark visibility (proxy, NOT calibrated)",
        "missing_policy": "explicit record with keypoints=null, observed=false; "
                          "never interpolated, never dropped",
        "n_frames": hi - lo + 1,
        "n_observed": n_observed,
        "n_missing": n_missing,
        "mean_pose_confidence": (vis_sum / n_observed) if n_observed else 0.0,
        "frames": records,
    }


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def path_for(video_id: str, delivery_id: str) -> str:
    return os.path.join(POSE_DIR, video_id, f"{delivery_id}.json")


def save(payload: Dict[str, Any]) -> str:
    path = path_for(payload["video_id"], payload["delivery_id"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema.dumps(payload))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_payload(video_id: str, delivery_id: str) -> Optional[Dict[str, Any]]:
    path = path_for(video_id, delivery_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_sequence(video_id: str, delivery_id: str) -> List[Dict[str, Any]]:
    p = load_payload(video_id, delivery_id)
    return (p or {}).get("frames", [])


def index_path(video_id: str) -> str:
    return os.path.join(POSE_DIR, video_id, "index.json")


def save_index(video_id: str, entries: List[Dict[str, Any]]) -> str:
    path = index_path(video_id)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "extractor_version": POSE_EXTRACTOR_VERSION,
        "video_id": video_id,
        "n_deliveries": len(entries),
        "deliveries": sorted(entries, key=lambda e: e["delivery_id"]),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema.dumps(payload))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
