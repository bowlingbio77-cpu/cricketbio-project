"""
Video registry: probe real cricket clips and register them **by reference**.

Phase 1 requirement: do not duplicate videos.  This module writes one JSON
object per line to ``raw/metadata/videos.jsonl`` describing each clip, its
content-hash ``video_id``, and its path.  Nothing is copied.

A clip that cannot be probed is still registered, with explicit ``null`` metadata
and ``probe_status: "FAILED"``.  Un-probeable clips are never silently dropped --
dropping them would make the manifest lie about the corpus.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from . import schema

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
VIDEOS_JSONL = schema.p("raw", "metadata", "videos.jsonl")


# --------------------------------------------------------------------------- #

@dataclass
class VideoRecord:
    video_id: str
    path: str
    filename: str
    bytes: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    duration_sec: Optional[float] = None
    sha_id_basis: str = "size+full_content_sha256"
    probe_status: str = "OK"
    probe_error: str = ""
    # --- Phase 6 human/curated fields (never inferred) ---
    camera_view: str = "unknown"
    bowling_side_if_known: str = "unknown"
    bowler_id_if_known: str = ""
    number_of_deliveries: Optional[int] = None
    annotation_status: str = "NONE"
    review_status: str = "UNREVIEWED"
    notes: str = ""

    def __post_init__(self) -> None:
        self.camera_view = schema.CameraView(self.camera_view)
        self.bowling_side_if_known = schema.BowlingSide(self.bowling_side_if_known)
        self.annotation_status = schema.AnnotationStatus(self.annotation_status)
        self.review_status = schema.ReviewStatus(self.review_status)
        if self.review_status == "REVIEWED" and not self.notes:
            # Reviewed status without a reviewer is unfalsifiable; require a note.
            raise ValueError(
                f"{self.video_id}: review_status=REVIEWED requires a note "
                "naming who reviewed it."
            )

    @property
    def group_key(self) -> str:
        """Bowler grouping key for leakage-safe splitting (Phase 5).

        Preference order:

        1. an explicit ``bowler_id_if_known`` -- the strongest possible group,
           since the same bowler must never straddle a split;
        2. the filename's **session prefix** with the trailing clip number
           stripped, e.g. ``fast_left_00000001.avi`` -> ``view:fast_left``.
           This is the camera-view/session bucket, which is a *conservative*
           group: it can only merge more, never split a real bowler across
           train and test;
        3. ``view:unknown`` when neither is available.
        """
        if self.bowler_id_if_known:
            return f"bowler:{self.bowler_id_if_known}"
        stem = os.path.splitext(self.filename)[0]
        m = re.match(r"^(.*?)_\d+$", stem)
        prefix = (m.group(1) if m else stem).strip("_")
        return f"view:{prefix or 'unknown'}"

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #

def probe_video(path: str) -> Dict:
    """Probe with OpenCV. Never raises."""
    import cv2  # local import: keeps this module importable without cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {"probe_status": "FAILED", "probe_error": "VideoCapture could not open file"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # Prefer metadata count; fall back to a decode walk for containers that
        # report 0 (common for short .avi clips written by opencv itself).
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            walked = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                walked += 1
            count = walked
        if count <= 0 or width <= 0 or height <= 0:
            return {
                "probe_status": "FAILED",
                "probe_error": f"degenerate metadata (frames={count}, {width}x{height})",
            }
        duration = (count / fps) if fps and fps > 0 else None
        return {
            "probe_status": "OK",
            "probe_error": "",
            "fps": round(fps, 6) if fps > 0 else None,
            "width": width,
            "height": height,
            "frame_count": count,
            "duration_sec": round(duration, 4) if duration else None,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"probe_status": "FAILED", "probe_error": f"{type(exc).__name__}: {exc}"}
    finally:
        cap.release()


def iter_videos(source: str) -> Iterable[str]:
    """Yield video file paths from a directory (recursive) or a single file."""
    if os.path.isfile(source):
        yield source
        return
    for root, _dirs, files in os.walk(source):
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXTENSIONS):
                yield os.path.join(root, fn)


def make_record(path: str, project_root: Optional[str] = None) -> VideoRecord:
    path = os.path.abspath(path)
    vid = schema.video_id_for_path(path)
    info = probe_video(path)
    stored = os.path.relpath(path, project_root) if project_root else path
    return VideoRecord(
        video_id=vid,
        path=os.path.abspath(path),
        filename=os.path.basename(path),
        bytes=os.path.getsize(path),
        fps=info.get("fps"),
        width=info.get("width"),
        height=info.get("height"),
        frame_count=info.get("frame_count"),
        duration_sec=info.get("duration_sec"),
        probe_status=info.get("probe_status", "FAILED"),
        probe_error=info.get("probe_error", ""),
    )


# --------------------------------------------------------------------------- #
# JSONL store
# --------------------------------------------------------------------------- #

def load_records(path: str = VIDEOS_JSONL) -> List[VideoRecord]:
    if not os.path.exists(path):
        return []
    out: List[VideoRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(VideoRecord(**json.loads(line)))
    return out


def save_records(records: List[VideoRecord], path: str = VIDEOS_JSONL) -> str:
    payload = "".join(
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for r in sorted(records, key=lambda r: r.video_id)
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def index_by_id(records: List[VideoRecord]) -> Dict[str, VideoRecord]:
    return {r.video_id: r for r in records}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
