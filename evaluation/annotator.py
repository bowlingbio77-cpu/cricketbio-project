"""
PaceAI P1 -- Human Ground-Truth Annotation Tool.

A lightweight OpenCV-based tool that lets a *human* annotator independently
measure a bowling delivery from a real clip and store it in
`evaluation/ground_truth/ground_truth.csv`.

INDEPENDENCE:
  * The tool NEVER loads or displays PaceAI predictions.
  * It only writes what the annotator clicks / types.
  * Angles are computed from anatomical points the annotator selects on the
    frame (a human measurement, per ANNOTATION_PROTOCOL.md), or typed by the
    annotator when measured with a protractor.
  * Frames are 0-indexed in the preprocessed space (20 FPS, 640x360), matching
    the Space PaceAI's pipeline runs in (src.preprocessing.preprocess_video).

Controls:
  left-click          select anatomical point or ball centre (active mode)
  j / l (or arrows)   step -1 / +1           [ / ]     step -5 / +5
  g                   jump to frame number    space     play / pause
  r                   mark current frame = RELEASE frame
  c                   mark current frame = FRONT-FOOT CONTACT frame
  e                   elbow mode (click shoulder, elbow, wrist)
  k                   knee mode (click hip, knee, ankle of front leg)
  t                   trunk mode (click hip centre, shoulder centre)
  f                   stride mode (click front foot, back foot)
  b                   ball mode (click ball centre on >=2 frames after release)
  v                   enter a measured value manually (console prompt)
  1-5                 cycle visibility flags release/knee/elbow/trunk
  x                   set annotation_confidence 1-5 (console prompt)
  z                   add / edit notes (console prompt)
  s                   save (stays open)
  q                   save + quit
  ESC                 quit without saving

Usage:
  python evaluation/annotator.py --video corrected_all_data/bowling/x.avi --annotator_id A
  python evaluation/annotator.py --video ... --annotator_id B   # independent second pass
"""
import argparse
import csv
import datetime
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import preprocessing  # noqa: E402
from evaluation.ground_truth_validation import (  # noqa: E402
    GT_COLUMNS, GT_PATH, ANNOTATION_PROTOCOL_VERSION,
)

WINDOW = "paceai-annotator"

VALIDITY_OPTIONS = ["good", "partial", "occluded", "out_of_frame", "not_established"]


# --------------------------------------------------------------------------- #
# Geometry helpers (human point measurements -> planned values)
# --------------------------------------------------------------------------- #

def _safe_int(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return -1


def angle_at_vertex(a, b, c):
    """Internal angle (degrees, 0..180) at point b formed by segments a-b and c-b."""
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    dot = u[0] * v[0] + u[1] * v[1]
    nu = math.hypot(*u)
    nv = math.hypot(*v)
    if nu == 0 or nv == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (nu * nv)))
    return math.degrees(math.acos(cos_angle))


def trunk_lean_from_points(hip, shoulder):
    """Trunk lean (deg) relative to the upward image vertical (0, -1)."""
    vec = (shoulder[0] - hip[0], shoulder[1] - hip[1])
    norm = math.hypot(*vec)
    if norm == 0:
        return None
    dot = (vec[0] * 0 + vec[1] * -1) / norm  # dot with (0, -1)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def stride_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def release_angle_from_ball(p1, p2):
    """Ball release angle (deg) from two ball centres, positive = downward.
    Magnitude of the angle the segment makes with the horizontal."""
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return None
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return abs(math.degrees(math.atan2(dy, abs(dx))))


# --------------------------------------------------------------------------- #
# CSV read/write (resume-safe, multi-annotator)
# --------------------------------------------------------------------------- #

def read_ground_truth_records(path: str) -> tuple:
    """Return (records_by_key, columns). records_by_key maps
    (video_id, annotator_id, delivery_id) -> dict row."""
    columns = GT_COLUMNS
    records = {}
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                columns = reader.fieldnames
            for row in reader:
                key = (row.get("video_id", ""), row.get("annotator_id", ""),
                       row.get("delivery_id", ""))
                records[key] = {c: row.get(c, "") for c in columns}
    return records, columns


def upsert_ground_truth_row(path: str, row: dict, columns=None) -> bool:
    """Insert or replace a single annotation row keyed by
    (video_id, annotator_id, delivery_id)."""
    columns = columns or GT_COLUMNS
    records, existing_columns = read_ground_truth_records(path)
    if existing_columns and set(existing_columns) >= set(columns):
        columns = existing_columns

    key = (row.get("video_id", ""), row.get("annotator_id", ""),
           row.get("delivery_id", ""))
    if not key[0] or not key[1]:
        return False
    records[key] = {c: row.get(c, "") for c in columns}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for k in sorted(records.keys()):
            writer.writerow(records[k])
    return True


# --------------------------------------------------------------------------- #
# Annotation session state
# --------------------------------------------------------------------------- #

class AnnotationSession:
    def __init__(self, video_path: str, annotator_id: str, delivery_id: str,
                 video_id: str = None, out_path: str = GT_PATH):
        self.video_path = video_path
        self.annotator_id = annotator_id
        self.delivery_id = delivery_id
        self.video_id = video_id or os.path.splitext(os.path.basename(video_path))[0]
        self.out_path = out_path

        self.records, self.columns = read_ground_truth_records(out_path)
        saved = self.records.get((self.video_id, annotator_id, delivery_id), {})
        self.annotation_protocol_version = saved.get(
            "annotation_protocol_version", ANNOTATION_PROTOCOL_VERSION)
        self.annotation_date = saved.get(
            "annotation_date", datetime.date.today().isoformat())
        self.release_frame = saved.get("release_frame", "")
        self.front_contact_frame = saved.get("front_contact_frame", "")
        self.front_knee_angle_deg = saved.get("front_knee_angle_deg", "")
        self.elbow_flexion_deg = saved.get("elbow_flexion_deg", "")
        self.trunk_lean_deg = saved.get("trunk_lean_deg", "")
        self.stride_length_px = saved.get("stride_length_px", "")
        self.release_angle_deg = saved.get("release_angle_deg", "")
        self.release_speed_mps = saved.get("release_speed_mps", "")
        self.visibility = {
            "release": saved.get("visibility_release", ""),
            "knee": saved.get("visibility_knee", ""),
            "elbow": saved.get("visibility_elbow", ""),
            "trunk": saved.get("visibility_trunk", ""),
        }
        self.confidence = saved.get("annotation_confidence", "")
        self.notes = saved.get("notes", "")

        self.pending_points = []   # list of (point, tag) picked during a mode
        self.ball_points = []      # list of (frame, (x, y))

    # -- helpers to keep numbers typed as numbers ---------------------------
    def _num(self, key):
        try:
            return float(key)
        except (TypeError, ValueError):
            return ""

    def row(self) -> dict:
        return {
            "video_id": self.video_id,
            "delivery_id": self.delivery_id,
            "annotator_id": self.annotator_id,
            "annotation_protocol_version": self.annotation_protocol_version,
            "annotation_date": self.annotation_date,
            "release_frame": self.release_frame,
            "front_contact_frame": self.front_contact_frame,
            "front_knee_angle_deg": self.front_knee_angle_deg,
            "elbow_flexion_deg": self.elbow_flexion_deg,
            "trunk_lean_deg": self.trunk_lean_deg,
            "stride_length_px": self.stride_length_px,
            "release_angle_deg": self.release_angle_deg,
            "release_speed_mps": self.release_speed_mps,
            "visibility_release": self.visibility["release"],
            "visibility_knee": self.visibility["knee"],
            "visibility_elbow": self.visibility["elbow"],
            "visibility_trunk": self.visibility["trunk"],
            "annotation_confidence": self.confidence,
            "notes": self.notes,
        }

    def save(self) -> bool:
        ok = upsert_ground_truth_row(self.out_path, self.row(), self.columns)
        print("  [saved] ->", self.out_path)
        return ok


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

def run_gui(session: AnnotationSession, frames) -> None:
    import cv2

    n = len(frames)
    pos = 0
    playing = False
    mode = None
    WINDOW_NAME = WINDOW

    def say(msg):
        for line in msg.splitlines():
            print("  " + line)

    def on_click(event, x, y, flags, param):
        nonlocal mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "elbow":
                session.pending_points.append(((x, y), "point"))
                if len(session.pending_points) == 3:
                    s, e, w = [p[0] for p in session.pending_points]
                    ang = angle_at_vertex(s, e, w)
                    if ang is not None:
                        session.elbow_flexion_deg = session._num(ang)
                        print(f"  elbow_flexion_deg = {ang:.1f} deg")
                    session.pending_points = []
                    mode = None
            elif mode == "knee":
                session.pending_points.append(((x, y), "point"))
                if len(session.pending_points) == 3:
                    hip, knee, ankle = [p[0] for p in session.pending_points]
                    ang = angle_at_vertex(hip, knee, ankle)
                    if ang is not None:
                        session.front_knee_angle_deg = session._num(ang)
                        print(f"  front_knee_angle_deg = {ang:.1f} deg")
                    session.pending_points = []
                    mode = None
            elif mode == "trunk":
                session.pending_points.append(((x, y), "point"))
                if len(session.pending_points) == 2:
                    hip, shoulder = [p[0] for p in session.pending_points]
                    lean = trunk_lean_from_points(hip, shoulder)
                    if lean is not None:
                        session.trunk_lean_deg = session._num(lean)
                        print(f"  trunk_lean_deg = {lean:.1f} deg")
                    session.pending_points = []
                    mode = None
            elif mode == "stride":
                session.pending_points.append(((x, y), "point"))
                if len(session.pending_points) == 2:
                    f1, f2 = [p[0] for p in session.pending_points]
                    session.stride_length_px = session._num(stride_distance(f1, f2))
                    print(f"  stride_length_px = {stride_distance(f1, f2):.1f}")
                    session.pending_points = []
                    mode = None
            elif mode == "ball":
                session.ball_points.append((pos, (x, y)))
                if len(session.ball_points) >= 2:
                    p1 = session.ball_points[-2][1]
                    p2 = session.ball_points[-1][1]
                    ang = release_angle_from_ball(p1, p2)
                    if ang is not None:
                        session.release_angle_deg = session._num(ang)
                        print(f"  release_angle_deg = {ang:.1f} deg")
                print(f"  ball centre @ frame {pos}: ({x},{y})")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_click)

    while True:
        pos = max(0, min(n - 1, pos))
        _idx, _ts, frame = frames[pos]
        img = frame.copy()

        cv2.putText(img, "HUMAN ANNOTATION - measured independently, NOT model output",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "frame %d/%d   mode=%s" % (pos, n - 1, mode or "-"),
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "RELEASE %s    CONTACT %s" % (session.release_frame,
                                                       session.front_contact_frame),
                    (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "knee=%s elbow=%s lean=%s" % (
            session.front_knee_angle_deg, session.elbow_flexion_deg,
            session.trunk_lean_deg),
            (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "stride_px=%s rel_angle=%s conf=%s" % (
            session.stride_length_px, session.release_angle_deg, session.confidence),
            (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "vis R=%s K=%s E=%s T=%s" % (
            session.visibility["release"] or "-", session.visibility["knee"] or "-",
            session.visibility["elbow"] or "-", session.visibility["trunk"] or "-"),
            (8, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        if _safe_int(session.release_frame) == pos:
            cv2.putText(img, "<- RELEASE", (8, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if _safe_int(session.front_contact_frame) == pos:
            cv2.putText(img, "<- FRONT CONTACT", (8, 158), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        for (x, y), tag in session.pending_points:
            cv2.circle(img, (x, y), 5, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)
        if playing:
            key = cv2.waitKey(33) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF
        if key == 255:
            continue

        if key == 27:  # ESC - quit without saving
            break
        elif key == ord("q"):
            session.save()
            break
        elif key == ord("j") or key == ord("a"):
            pos -= 1
        elif key == ord("l") or key == ord("d"):
            pos += 1
        elif key == 82:  # up arrow
            pos -= 1
        elif key == 84:  # down arrow
            pos += 1
        elif key == ord("["):
            pos -= 5
        elif key == ord("]"):
            pos += 5
        elif key == ord(" "):
            playing = not playing
        elif key == ord("g"):
            try:
                pos = int(input("frame: "))
            except (ValueError, EOFError):
                pass
        elif key == ord("r"):
            session.release_frame = pos
            print(f"  release_frame = {pos}")
        elif key == ord("c"):
            session.front_contact_frame = pos
            print(f"  front_contact_frame = {pos}")
        elif key == ord("e"):
            mode = "elbow"; session.pending_points = []
            say("3 clicks on release frame: shoulder, elbow, wrist")
        elif key == ord("k"):
            mode = "knee"; session.pending_points = []
            say("3 clicks on release frame: front hip, knee, ankle")
        elif key == ord("t"):
            mode = "trunk"; session.pending_points = []
            say("2 clicks on release frame: hip centre, shoulder centre")
        elif key == ord("s"):
            if mode is not None:
                mode = None
                session.pending_points = []
            else:
                session.save()
        elif key == ord("f"):
            mode = "stride"; session.pending_points = []
            say("2 clicks on contact frame: front foot, back foot")
        elif key == ord("b"):
            mode = "ball"; session.pending_points = []
            say("click ball centre on >=2 frames after release")
        elif key == ord("v"):
            try:
                field = input("field: ").strip()
                value = input("value: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if field in ["front_knee_angle_deg", "elbow_flexion_deg",
                         "trunk_lean_deg", "release_angle_deg",
                         "stride_length_px"]:
                setattr(session, field, session._num(value) if value else "")
                print(f"  {field} = {value}")
            elif field == "release_speed_mps":
                print("  NOTE: release_speed_mps is NOT measurable from "
                      "uncalibrated footage -- left blank per protocol.")
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            mapping = {ord("1"): "release", ord("2"): "knee",
                       ord("3"): "elbow", ord("4"): "trunk", ord("5"): "rel_speed"}
            target = mapping[key]
            targets = {"release": "release", "knee": "knee", "elbow": "elbow",
                       "trunk": "trunk"}
            if target == "rel_speed":
                continue
            current = session.visibility[target]
            idx = VALIDITY_OPTIONS.index(current) if current in VALIDITY_OPTIONS else -1
            session.visibility[target] = VALIDITY_OPTIONS[(idx + 1) % len(VALIDITY_OPTIONS)]
            print(f"  visibility_{target} = {session.visibility[target]}")
        elif key == ord("x"):
            try:
                session.confidence = input("confidence 1-5: ").strip()[:3]
            except (EOFError, KeyboardInterrupt):
                continue
        elif key == ord("z"):
            try:
                session.notes = input("notes: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="PaceAI P1 ground-truth annotator")
    parser.add_argument("--video", default=None, help="path to a real clip")
    parser.add_argument("--annotator_id", default="",
                        help="required annotator label, e.g. ANNOTATOR_A "
                             "(no personal names unless intentionally recorded)")
    parser.add_argument("--delivery_id", default="1", help="delivery id within the clip")
    parser.add_argument("--video_id", default=None, help="override clip id")
    parser.add_argument("--out", default=GT_PATH, help="ground-truth CSV path")
    parser.add_argument("--list", action="store_true", help="list the 8 pilot clips")
    args = parser.parse_args()

    if args.list:
        import glob
        found = []
        for pattern in ("*.avi", "*.mp4"):
            found.extend(glob.glob(
                os.path.join("corrected_all_data", "bowling", pattern)))
        for p in sorted(found)[:12]:
            print(p)
        return

    if not args.annotator_id.strip():
        sys.exit("annotator_id is required (e.g. --annotator_id ANNOTATOR_A)")

    if not args.video:
        sys.exit("--video is required (or use --list)")

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")

    frames = list(preprocessing.preprocess_video(args.video))
    if len(frames) < 3:
        sys.exit(f"too few preprocessed frames: {len(frames)}")

    session = AnnotationSession(
        video_path=args.video, annotator_id=args.annotator_id.strip(),
        delivery_id=args.delivery_id, video_id=args.video_id, out_path=args.out)
    print(f"clip {session.video_id} | annotator {session.annotator_id} | "
          f"delivery {session.delivery_id} | protocol "
          f"{session.annotation_protocol_version} | {len(frames)} preprocessed "
          f"frames (20fps); resume = "
          f"{bool(session.release_frame or session.notes)}")
    print("e: elbow  k: knee  t: trunk  f: stride  b: ball  r: release  c: contact "
          "s: save  q: save+quit")
    run_gui(session, frames)
    print("done.")


if __name__ == "__main__":
    main()