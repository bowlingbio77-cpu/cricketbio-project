"""
Stage 2: Bowler Detection (YOLOv11)

Wraps Ultralytics YOLOv11 for person detection on each frame. Requires:
    pip install ultralytics
which will auto-download `yolo11n.pt` on first use (needs internet).

Falls back to an OpenCV HOG person-detector if ultralytics isn't installed,
so the rest of the pipeline can still be exercised offline/without a GPU.
"""
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
from . import config

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


def resolve_weights(weights: str = config.YOLO_WEIGHTS) -> str:
    """Find a usable YOLO weights file. Tries the configured path, the project
    root, and the working directory before letting Ultralytics auto-download."""
    candidates = [weights, os.path.join(config.PROJECT_ROOT, "yolo11n.pt"), "yolo11n.pt"]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return "yolo11n.pt"


@dataclass
class Detection:
    frame_idx: int
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float
    class_id: int


class BowlerDetector:
    def __init__(self, weights: str = config.YOLO_WEIGHTS,
                 conf_threshold: float = config.DETECTION_CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.backend = "yolov11" if _HAS_ULTRALYTICS else "hog_fallback"
        if self.backend == "yolov11":
            self.model = YOLO(resolve_weights(weights))
        else:
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        if self.backend == "yolov11":
            return self._detect_yolo(frame, frame_idx)
        return self._detect_hog(frame, frame_idx)

    def _detect_yolo(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        results = self.model.predict(
            frame, classes=[config.BOWLER_CLASS_ID],
            conf=self.conf_threshold, verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append(Detection(frame_idx, tuple(xyxy), conf, cls))
        return dets

    def _detect_hog(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = self.model.detectMultiScale(
            gray, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        dets = []
        for (x, y, w, h), conf in zip(rects, weights):
            if float(conf) < self.conf_threshold * 2:  # HOG confidence scale differs
                continue
            dets.append(Detection(frame_idx, (x, y, x + w, y + h), float(conf), 0))
        return dets

    def select_primary_bowler(self, detections: List[Detection]) -> Optional[Detection]:
        """Heuristic: the bowler is usually the largest, most-confident detection
        near the centre of frame during the run-up/delivery."""
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence * _bbox_area(d.bbox))


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)
