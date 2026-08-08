"""
Stage 3: Bowler Tracking (ByteTrack)

Ultralytics ships ByteTrack as a built-in tracker, so tracking is exposed via
`model.track(...)` rather than a separate library. This module wraps that,
and keeps the single track_id with the longest continuous run as "the bowler"
(handles fielders/umpires briefly entering frame).

Falls back to a lightweight IoU-based tracker (greedy nearest-bbox matching)
when ultralytics isn't installed, so the pipeline still runs end-to-end offline.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from . import config
from .detection import Detection, _bbox_area

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


@dataclass
class Track:
    track_id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[tuple] = field(default_factory=list)

    def __len__(self):
        return len(self.frames)


class BowlerTracker:
    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf_threshold: float = config.DETECTION_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.backend = "bytetrack" if _HAS_ULTRALYTICS else "iou_fallback"
        if self.backend == "bytetrack":
            self.model = YOLO(weights if weights.endswith(".pt") else "yolo11n.pt")
        self._iou_tracks: Dict[int, Track] = {}
        self._next_id = 0

    def track_video(self, video_path: str) -> Dict[int, Track]:
        """Run tracking over an entire video file, return {track_id: Track}."""
        if self.backend == "bytetrack":
            return self._track_bytetrack(video_path)
        raise RuntimeError(
            "IoU fallback tracker works frame-by-frame; call track_frame() in a loop instead."
        )

    def _track_bytetrack(self, video_path: str) -> Dict[int, Track]:
        tracks: Dict[int, Track] = {}
        results = self.model.track(
            source=video_path, classes=[config.BOWLER_CLASS_ID],
            conf=self.conf_threshold, tracker=config.BYTETRACK_CONFIG,
            persist=True, stream=True, verbose=False,
        )
        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue
            ids = r.boxes.id.int().tolist()
            xyxys = r.boxes.xyxy.tolist()
            for tid, box in zip(ids, xyxys):
                if tid not in tracks:
                    tracks[tid] = Track(track_id=tid)
                tracks[tid].frames.append(frame_idx)
                tracks[tid].bboxes.append(tuple(box))
        return tracks

    # --- Fallback: simple greedy IoU tracker, one frame at a time ---
    def track_frame(self, frame_idx: int, detections: List[Detection]) -> Dict[int, tuple]:
        assigned = {}
        used_tracks = set()
        for det in detections:
            best_id, best_iou = None, 0.3  # IoU threshold to continue a track
            for tid, tr in self._iou_tracks.items():
                if tid in used_tracks or not tr.bboxes:
                    continue
                iou = _iou(det.bbox, tr.bboxes[-1])
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                best_id = self._next_id
                self._iou_tracks[best_id] = Track(track_id=best_id)
                self._next_id += 1
            self._iou_tracks[best_id].frames.append(frame_idx)
            self._iou_tracks[best_id].bboxes.append(det.bbox)
            used_tracks.add(best_id)
            assigned[best_id] = det.bbox
        return assigned

    def get_iou_tracks(self) -> Dict[int, Track]:
        return self._iou_tracks


def select_bowler_track(tracks: Dict[int, Track]) -> Track | None:
    """Bowler = the track with the most frames and largest average bbox
    (filters out momentary detections of fielders/spectators)."""
    if not tracks:
        return None

    def score(tr: Track):
        avg_area = np.mean([_bbox_area(b) for b in tr.bboxes])
        return len(tr) * avg_area

    return max(tracks.values(), key=score)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
