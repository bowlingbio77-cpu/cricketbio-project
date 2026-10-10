"""
Phase 3 -- HUMAN ANNOTATION GUI (local, offline, OpenCV).

    python tools/annotate_cricket.py --annotator_id A --list
    python tools/annotate_cricket.py --annotator_id A --video-id v_9f2c1a0b4e77
    python tools/annotate_cricket.py --annotator_id A --video corrected_all_data/bowling/fast_left_00000001.avi
    python tools/annotate_cricket.py --annotator_id A --video-id v_... --show-suggestions path.json

    HUMAN ANNOTATION
    Measured independently - NOT model output

is drawn as a permanent banner on every frame.

Ground-truth rules enforced here
--------------------------------
* ``--annotator_id`` is mandatory; the tool exits without it.
* Ground-truth fields are **never** pre-filled from a PaceAI prediction.  With
  ``--show-suggestions`` the model hints are drawn as dashed amber boxes in a
  panel captioned "MODEL SUGGESTIONS - NOT GROUND TRUTH", and the only way they
  enter an annotation is the human pressing ``a`` (accept box) or ``A`` (accept
  box + suggested role).  Accepted boxes are written with
  ``visibility="uncertain"`` and ``annotation_confidence=0.5`` so a reviewer can
  find them again.
* Every mutation autosaves (atomic write) -- a crash cannot destroy work.

Controls
--------
  navigation      a/d  +/- 10      ,/.  prev/next annotated frame
  select          w/s  cycle tracks up/down
  box             drag a rectangle; with a track selected, drag inside it to MOVE,
                  drag a corner handle to RESIZE, press X to delete
  roles           0 BOWLER  1 STRIKER  2 NON_STRIKER  3 WICKETKEEPER  4 FIELDER  5 UNKNOWN
  role range      [ set role from current frame to the range start, ] to current frame
  phases          q NON_BOWLING  e RUN_UP  r APPROACH  t GATHER  y DELIVERY_STRIDE
                  u FRONT_FOOT_CONTACT  i RELEASE  o FOLLOW_THROUGH  p UNKNOWN
                  Q..O (shift) apply the phase to the WHOLE current delivery window
  visibility      z fully  x partially  c heavily  v not_visible  b uncertain
  delivery        n new   N rename-note   1 RUN_UP  2 GATHER  3 DELIVERY_STRIDE
                  4 FRONT_FOOT_CONTACT  5 RELEASE  6 FOLLOW_THROUGH   7 window end
                  Shift+1..6 stamp the phase on the delivery AND the per-frame row
  objects         B draw BALL, M draw STUMPS (drag), shift to delete
  scene           8 bowler_runup_region  9 striker_region  ; pitch_region
  meta            k camera orientation  g confidence down  G confidence up
  save/quit       s SAVE  ESC quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cricket_understanding import registry, schema            # noqa: E402
from src.cricket_understanding.annotation import (                 # noqa: E402
    BANNER_LINE_1, BANNER_LINE_2, AnnotationError, AnnotationSession,
)

WINDOW = "cricket_understanding_v1 - HUMAN ANNOTATION"

# BGR
C_BANNER_BG = (28, 22, 18)
C_BANNER_FG = (255, 255, 255)
C_BOX = (0, 235, 0)          # green   = human ground truth
C_BOX_SEL = (0, 200, 255)    # yellow  = selected human track
C_SUGGEST = (80, 160, 255)   # amber   = model suggestion (NOT ground truth)
C_BALL = (60, 60, 240)
C_STUMPS = (200, 120, 60)
C_PHASE = (255, 160, 40)
C_TEXT = (240, 240, 240)
C_WARN = (60, 60, 240)
PANEL_W = 330


def _put(img, text, org, color=C_TEXT, scale=0.45, thick=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, bg=(24, 24, 24)):
    import cv2
    import numpy as np
    h = min(h, img.shape[0] - y)
    w = min(w, img.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    sub = img[y:y + h, x:x + w]
    sub[:] = np.array(bg, dtype=np.uint8)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)


def _draw_banner(img):
    """The mandated, permanent, unmistakable provenance banner."""
    import cv2
    h, w = img.shape[:2]
    bar = 34 if h > 480 else 26
    cv2.rectangle(img, (0, 0), (w, bar), C_BANNER_BG, -1)
    cv2.line(img, (0, bar), (w, bar), C_BOX_SEL, 1)
    cv2.putText(img, BANNER_LINE_1, (8, 14 if h > 480 else 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BANNER_FG, 2, cv2.LINE_AA)
    cv2.putText(img, BANNER_LINE_2, (8, 26 if h > 480 else 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_BOX_SEL, 1, cv2.LINE_AA)
    _put(img, f"annotator: {ANNOTATOR_ID}", (w - 250, 20 if h > 480 else 17),
         C_TEXT, 0.42)


def _draw_timeline(img, session: AnnotationSession):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    y0, bh = h - 16, 12
    cv2.rectangle(img, (0, y0 - 2), (w, h), (20, 20, 20), -1)
    for f in session.annotated_frames():
        x = int(6 + (w - 12) * f / max(1, session.n_frames - 1))
        cv2.line(img, (x, y0), (x, y0 + bh), C_BOX, -1)
    for d in session.deliveries.values():
        x1 = int(6 + (w - 12) * d.start_frame / max(1, session.n_frames - 1))
        x2 = int(6 + (w - 12) * d.end_frame / max(1, session.n_frames - 1))
        cv2.rectangle(img, (x1, y0 - 2), (max(x2, x1 + 1), y0 + bh + 1), C_PHASE, 1)
    cur = int(6 + (w - 12) * session.current_frame / max(1, session.n_frames - 1))
    cv2.line(img, (cur, y0 - 4), (cur, y0 + bh + 4), C_WARN, 1)


def _draw_overlay(img, session: AnnotationSession):
    import cv2
    h, w = img.shape[:2]
    # --- model suggestions: unmistakably separate from ground truth ---------
    if session.suggestions_enabled:
        for s in session.suggestions_at(session.current_frame):
            x1, y1, x2, y2 = (int(v) for v in s.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), C_SUGGEST, 1, cv2.LINE_AA)
            tag = f"SUGGEST {s.source}"
            if s.suggested_role:
                tag += f" ~{s.suggested_role}"
            _put(img, tag, (x1, max(12, y1 - 6)), C_SUGGEST, 0.38)
    # --- human ground truth ------------------------------------------------
    for p in session.people_at(session.current_frame):
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        sel = (p.track_id == session.selected_track)
        cv2.rectangle(img, (x1, y1), (x2, y2), C_BOX_SEL if sel else C_BOX,
                      2 if sel else 1, cv2.LINE_AA)
        _put(img, f"T{p.track_id} {p.role} [{p.visibility[:4]}]",
             (x1, max(12, y1 - 6)), C_BOX_SEL if sel else C_BOX, 0.42)
    for (f, t, cls), o in session.objects.items():
        if f != session.current_frame:
            continue
        x1, y1, x2, y2 = (int(v) for v in (o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2))
        col = C_BALL if cls == "BALL" else C_STUMPS
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        _put(img, cls, (x1, max(12, y1 - 6)), col, 0.38)
    for (f, t), ph in session.phases.items():
        if f != session.current_frame:
            continue
        x1, y1 = int(session.people[(f, t)].bbox_x1), int(session.people[(f, t)].bbox_y1)
        _put(img, f"T{t}:{ph.phase}{'?' if ph.uncertain else ''}",
             (x1, int(session.people[(f, t)].bbox_y2) + 14), C_PHASE, 0.42)
    if session.draft_bbox:
        x1, y1, x2, y2 = session.draft_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), C_WARN, 1, cv2.LINE_AA)
    # right-hand panel
    _panel(img, w - PANEL_W, 38, PANEL_W, h - 60)
    x = w - PANEL_W + 8
    y = 56
    _put(img, BANNER_LINE_1, (x, y), C_BANNER_FG, 0.44, 1)
    _put(img, BANNER_LINE_2, (x, y + 14), C_BOX_SEL, 0.36)
    y += 40
    _put(img, f"frame {session.current_frame}/{session.n_frames - 1}", (x, y), C_TEXT, 0.45)
    y += 18
    d = session.current_delivery()
    _put(img, f"delivery: {d.delivery_id if d else '-'}  bowler T"
              f"{d.bowler_track_id if d else '-'}", (x, y), C_TEXT, 0.42)
    y += 16
    for fld, lbl in (("runup_start_frame", "run-up"),
                     ("gather_frame", "gather"),
                     ("delivery_stride_frame", "del.stride"),
                     ("front_foot_contact_frame", "FFC"),
                     ("release_frame", "RELEASE"),
                     ("followthrough_end_frame", "follow-thru")):
        val = getattr(d, fld, None) if d else None
        mark = "OK " if val is not None else "-- "
        _put(img, f"{mark}{lbl:12s} {val if val is not None else '-'}",
             (x, y), C_BOX if val is not None else C_TEXT, 0.40)
        y += 14
    y += 6
    for t in session.track_ids():
        r = session.role_at(session.current_frame, t)
        sel = ">" if t == session.selected_track else " "
        _put(img, f"{sel} T{t}: {r or '-'}  frames={len(session.frames_for_track(t))}",
             (x, y), C_BOX_SEL if t == session.selected_track else C_TEXT, 0.40)
        y += 14
    y += 6
    c = session.counts()
    _put(img, f"boxes={c['people']} phases={c['phases']} del={c['deliveries']}",
         (x, y), C_TEXT, 0.40)
    y += 16
    for m in session.messages[-3:]:
        _put(img, m[:44], (x, y), C_WARN, 0.36)
        y += 13
    _put(img, "H help  S save  ESC quit", (x, h - 48), C_TEXT, 0.38)


HELP_TEXT = """Cricket Understanding v1 - HUMAN ANNOTATION (measured independently)

NAV      a/d  +/-(10)  ,/. prev/next annotated frame   g/G confidence  S save  ESC quit
BOX      drag = new box (assigned to selected track, or next free id)
         drag inside selected box = MOVE   drag corner = RESIZE   X delete
TRACK    w/s select prev/next track
ROLE     0 BOWLER 1 STRIKER 2 NON_STRIKER 3 WICKETKEEPER 4 FIELDER 5 UNKNOWN
         [ / ]  set role from range-start .. current frame for the selected track
VIS      z fully  x partially  c heavily  v not_visible  b uncertain
PHASE    q NON_BOWLING e RUN_UP r APPROACH t GATHER y DELIVERY_STRIDE
         u FRONT_FOOT_CONTACT i RELEASE o FOLLOW_THROUGH p UNKNOWN
         SHIFT+q..o  apply phase over the whole current delivery window
DELIVERY n new   1 run-up  2 gather  3 delivery stride  4 FFC  5 RELEASE  6 follow-thru
         7 mark end of window   SHIFT+1..6 also stamps the per-frame phase row
OBJECTS  B drag = BALL   M drag = STUMPS   SHIFT+B delete ball   SHIFT+M delete stumps
SCENE    8 bowler run-up region   9 striker region   ; pitch region  k camera orientation
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="PaceAI cricket_understanding_v1 HUMAN annotation GUI")
    ap.add_argument("--annotator-id", "--annotator_id", dest="annotator_id",
                    required=True,
                    help="REQUIRED. The human doing the labelling.")
    ap.add_argument("--video-id", default=None, help="registered video_id")
    ap.add_argument("--video", default=None, help="path to a video file")
    ap.add_argument("--list", action="store_true", help="list registered videos")
    ap.add_argument("--show-suggestions", default=None,
                    help="OPTIONAL model hints JSON. Displayed separately; never "
                         "auto-written as ground truth.")
    ap.add_argument("--fps-hint", type=float, default=None)
    args = ap.parse_args(argv)

    annotator_id = args.annotator_id.strip()
    if not annotator_id:
        print("annotator_id is required.", file=sys.stderr)
        return 2

    global ANNOTATOR_ID
    ANNOTATOR_ID = annotator_id

    records = registry.load_records()
    by_id = registry.index_by_id(records)

    if args.list:
        print(f"{len(records)} registered video(s)\n")
        print(f"{'video_id':16s} {'split':10s} {'status':11s} {'frames':>7s}  filename")
        for r in records:
            n = len(schema.load_person_annotations(r.video_id))
            print(f"{r.video_id:16s} {'-':10s} {'{' + str(n) + ' boxes':11s} "
                  f"{str(r.frame_count or '?'):>7s}  {r.filename}")
        return 0

    # ---- resolve the video ------------------------------------------------
    video_path: Optional[str] = None
    video_id = args.video_id
    if video_path is None and video_id:
        rec = by_id.get(video_id)
        if rec is None:
            print(f"unknown video_id {video_id!r}; run --list", file=sys.stderr)
            return 2
        video_path = rec.path
    if args.video:
        video_path = os.path.abspath(args.video)
        if video_id is None:
            rec = by_id.get(schema.video_id_for_path(video_path))
            video_id = rec.video_id if rec else f"v_unreg_{os.path.basename(video_path)}"
    if not video_path or not os.path.exists(video_path):
        print(f"video not found: {video_path}", file=sys.stderr)
        return 2

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"OpenCV could not open {video_path}", file=sys.stderr)
        return 2
    fps = args.fps_hint or (cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    if n_frames <= 0:  # some containers lie; walk it once
        n = 0
        while cap.grab():
            n += 1
        n_frames = n
        cap.release()
        cap = cv2.VideoCapture(video_path)
    cap.release()

    schema.ensure_dirs()
    session = AnnotationSession(video_id=video_id, annotator_id=annotator_id,
                                n_frames=n_frames, video_path=video_path, fps=fps)
    session.load()
    session.messages.append(f"loaded {len(session.people)} existing box(es)")

    n_sugg = 0
    if args.show_suggestions:
        if not os.path.exists(args.show_suggestions):
            print(f"suggestions file not found: {args.show_suggestions}", file=sys.stderr)
            return 2
        with open(args.show_suggestions, encoding="utf-8") as fh:
            rows = json.load(fh)
        rows = [r for r in rows if r.get("video_id") in (None, video_id)]
        n_sugg = session.load_suggestions(rows)
        session.messages.append(f"{n_sugg} MODEL suggestion(s) loaded (NOT ground truth)")
    else:
        session.messages.append("no model suggestions loaded")

    # ---- UI state ---------------------------------------------------------
    state = {"dragging": False, "start": None, "range_start": None,
             "drawing": None, "grabbing": None, "show_help": False,
             "pending_role": None, "next_track": max(session.track_ids(), default=-1) + 1}

    def track_under(x, y):
        for p in session.people_at(session.current_frame):
            if p.bbox_x1 <= x <= p.bbox_x2 and p.bbox_y1 <= y <= p.bbox_y2:
                return p.track_id
        return None

    def on_mouse(event, x, y, flags, param):
        if state["drawing"] is not None and event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            sx, sy = state["start"]
            session.draft_bbox = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            box = session.draft_bbox
            session.draft_bbox = None
            target = state["drawing"]
            state["drawing"] = None
            if not box or (box[2] - box[0]) <= 6 or (box[3] - box[1]) <= 6:
                session.messages.append("box too small / cancelled")
                return
            try:
                if target in ("BALL", "STUMPS"):
                    tid = session.selected_track
                    if tid is None:
                        tid = next_free_track()
                        session.selected_track = tid
                    session.set_object(session.current_frame, tid, target, box)
                    session.messages.append(f"{target} set @f{session.current_frame}")
                elif target in schema.SCENE_KEYS:
                    session.set_scene_from_box(target, box)
                    session.messages.append(f"scene {target} set")
                else:
                    tid = session.selected_track
                    if tid is None:
                        tid = next_free_track()
                        state["next_track"] = max(state["next_track"], tid + 1)
                    session.set_person(session.current_frame, tid, box, role="UNKNOWN")
                    session.messages.append(f"T{tid} box set (role UNKNOWN)")
            except AnnotationError as e:
                session.messages.append(str(e)[:60])

    def grab_frame(idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            return None
        if fr.shape[1] != w0:  # keep coordinates consistent with the annotation
            fr = cv2.resize(fr, (w0, int(fr.shape[0] * w0 / fr.shape[1])))
        return fr

    def next_free_track():
        ids = set(session.track_ids())
        t = 0
        while t in ids:
            t += 1
        return t

    state["next_track"] = next_free_track()

    def apply_role(role):
        if session.selected_track is None:
            session.messages.append("select a track first (w/s)")
            return
        try:
            session.set_role(session.current_frame, session.selected_track, role)
            session.messages.append(f"T{session.selected_track} -> {role}")
        except AnnotationError as e:
            session.messages.append(str(e)[:60])

    def apply_phase(ph, whole_window=False):
        if session.selected_track is None:
            session.messages.append("select a track first (w/s)")
            return
        d = session.current_delivery()
        if whole_window and d is not None:
            session.apply_phase_range(d.start_frame, d.end_frame,
                                      session.selected_track, ph)
            session.messages.append(f"{ph} over [{d.start_frame},{d.end_frame}]")
        else:
            session.set_phase(session.current_frame, session.selected_track, ph)
            session.messages.append(f"phase {ph} @ f{session.current_frame}")

    PHASE_KEYS = {
        ord("q"): "NON_BOWLING", ord("e"): "RUN_UP", ord("r"): "APPROACH",
        ord("t"): "GATHER", ord("y"): "DELIVERY_STRIDE",
        ord("u"): "FRONT_FOOT_CONTACT", ord("i"): "RELEASE",
        ord("o"): "FOLLOW_THROUGH", ord("p"): "UNKNOWN",
    }
    PHASE_KEYS_SHIFT = {k - 32: v for k, v in PHASE_KEYS.items()}
    ROLE_KEYS = {
        ord("0"): "BOWLER", ord("1"): "STRIKER", ord("2"): "NON_STRIKER",
        ord("3"): "WICKETKEEPER", ord("4"): "FIELDER", ord("5"): "UNKNOWN",
    }
    DELIVERY_FIELD_KEYS = {
        ord("1"): "runup_start_frame", ord("2"): "gather_frame",
        ord("3"): "delivery_stride_frame", ord("4"): "front_foot_contact_frame",
        ord("5"): "release_frame", ord("6"): "followthrough_end_frame",
    }
    VIS_KEYS = {
        ord("z"): "fully_visible", ord("x"): "partially_occluded",
        ord("c"): "heavily_occluded", ord("v"): "not_visible", ord("b"): "uncertain",
    }
    PHASE_FOR_DELIVERY_FIELD = {
        "runup_start_frame": "RUN_UP", "gather_frame": "GATHER",
        "delivery_stride_frame": "DELIVERY_STRIDE",
        "front_foot_contact_frame": "FRONT_FOOT_CONTACT",
        "release_frame": "RELEASE", "followthrough_end_frame": "FOLLOW_THROUGH",
    }

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(HELP_TEXT)
    print(f"\nvideo_id={video_id} frames={n_frames} {w0}x{h0} @{fps:.2f}fps")
    print(f"annotator={annotator_id} suggestions={n_sugg} (displayed separately)\n")

    try:
        while True:
            session.current_frame = max(0, min(session.n_frames - 1, session.current_frame))
            frame = grab_frame(session.current_frame)
            if frame is None:
                session.messages.append(f"frame {session.current_frame} unreadable")
                frame = __import__("numpy").zeros((h0, w0, 3), dtype="uint8")
            canvas = frame.copy()
            _draw_banner(canvas)
            _draw_overlay(canvas, session)
            _draw_timeline(canvas, session)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                session.save()
                print("saved and quit")
                break
            elif key in (ord("H"), ord("?")):
                state["show_help"] = not state["show_help"]
                if state["show_help"]:
                    print(HELP_TEXT)
            elif key == ord("S"):
                p = session.save()
                session.messages.append(f"saved {len(p)} file(s)")
                print("saved:", p)
            elif key == ord("a"):
                session.current_frame = max(0, session.current_frame - 1)
            elif key == ord("d"):
                session.current_frame = min(session.n_frames - 1, session.current_frame + 1)
            elif key == ord("W") or key == ord("+"):
                session.current_frame = max(0, session.current_frame - 10)
            elif key == ord("D") or key == ord("."):
                session.current_frame = min(session.n_frames - 1, session.current_frame + 10)
            elif key == ord(","):
                fr = session.annotated_frames()
                prev = [f for f in fr if f < session.current_frame]
                if prev:
                    session.current_frame = prev[-1]
            elif key == ord("/"):
                fr = session.annotated_frames()
                nxt = [f for f in fr if f > session.current_frame]
                if nxt:
                    session.current_frame = nxt[0]
            elif key in (ord("w"), ord("s")):
                ids = session.track_ids()
                if ids:
                    i = ids.index(session.selected_track) if session.selected_track in ids else -1
                    session.selected_track = ids[(i + (1 if key == ord("s") else -1)) % len(ids)]
            elif key in ROLE_KEYS:
                state["pending_role"] = ROLE_KEYS[key]
                apply_role(ROLE_KEYS[key])
            elif key in VIS_KEYS and session.selected_track is not None:
                try:
                    session.set_visibility(session.current_frame,
                                           session.selected_track, VIS_KEYS[key])
                    session.messages.append(f"vis {VIS_KEYS[key]}")
                except AnnotationError as e:
                    session.messages.append(str(e)[:60])
            elif key in PHASE_KEYS:
                apply_phase(PHASE_KEYS[key], whole_window=False)
            elif key in PHASE_KEYS_SHIFT:
                apply_phase(PHASE_KEYS_SHIFT[key], whole_window=True)
            elif key in DELIVERY_FIELD_KEYS:
                fld = DELIVERY_FIELD_KEYS[key]
                d = session.current_delivery()
                if d is None:
                    session.messages.append("press n to start a delivery first")
                else:
                    session.mark_key_frame(fld)
                    if session.selected_track is not None:
                        session.set_phase(session.current_frame, session.selected_track,
                                          PHASE_FOR_DELIVERY_FIELD[fld])
                    session.messages.append(f"{fld} = f{session.current_frame}")
            elif key == ord("7"):
                d = session.current_delivery()
                if d:
                    session.set_delivery_window(d.delivery_id, d.start_frame,
                                                session.current_frame)
                    session.messages.append(f"window end = f{session.current_frame}")
            elif key == ord("n"):
                tid = session.selected_track
                if tid is None:
                    tid = next_free_track()
                did = session.next_delivery_id()
                try:
                    session.start_delivery(did, tid)
                    session.selected_track = tid
                    session.messages.append(f"delivery {did} bowler T{tid}")
                except AnnotationError as e:
                    session.messages.append(str(e)[:60])
            elif key == ord("X"):
                if session.selected_track is not None:
                    try:
                        session.move_bbox(session.current_frame,
                                          session.selected_track, None)
                        session.messages.append(f"T{session.selected_track} deleted")
                    except AnnotationError as e:
                        session.messages.append(str(e)[:60])
            elif key in (ord("B"), ord("M")):
                cls = "BALL" if key == ord("B") else "STUMPS"
                if state["drawing"] == cls:
                    state["drawing"] = None
                else:
                    state["drawing"] = cls
                    session.messages.append(f"next drag = {cls}")
            elif key in (ord("8"), ord("9"), ord(";")):
                k = {ord("8"): "bowler_runup_region", ord("9"): "striker_region",
                     ord(";"): "pitch_region"}[key]
                state["drawing"] = k
                session.messages.append(f"next drag = scene {k}")
            elif key == ord("k"):
                views = schema.CAMERA_VIEWS
                cur = session.scene.camera_orientation if session.scene else "unknown"
                nxt = views[(views.index(cur) + 1) % len(views)] if cur in views else "unknown"
                session.set_camera_orientation(nxt)
                session.messages.append(f"camera = {nxt}")
            elif key == ord("G"):
                rec = session.people.get((session.current_frame, session.selected_track))
                if rec:
                    session.set_person(rec.frame_id, rec.track_id, rec.bbox, role=rec.role,
                                       visibility=rec.visibility,
                                       annotation_confidence=min(1.0, rec.annotation_confidence + 0.1),
                                       notes=rec.notes)
            elif key == ord("g"):
                rec = session.people.get((session.current_frame, session.selected_track))
                if rec:
                    session.set_person(rec.frame_id, rec.track_id, rec.bbox, role=rec.role,
                                       visibility=rec.visibility,
                                       annotation_confidence=max(0.0, rec.annotation_confidence - 0.1),
                                       notes=rec.notes)
            elif key == ord("["):
                state["range_start"] = session.current_frame
                session.messages.append(f"range start f{state['range_start']}")
            elif key == ord("]"):
                if state["range_start"] is None or session.selected_track is None:
                    session.messages.append("press [ and select a track first")
                else:
                    role = state["pending_role"] or "FIELDER"
                    n = session.apply_role_range(state["range_start"],
                                                 session.current_frame,
                                                 session.selected_track, role)
                    session.messages.append(f"{role} over {n} frame(s)")
                    state["range_start"] = None
            elif key == ord("A"):
                # accept a model suggestion for the selected track at this frame
                if session.selected_track is None:
                    session.messages.append("select a track first (w/s)")
                else:
                    try:
                        rec = session.accept_suggestion(session.selected_track)
                        session.messages.append(
                            f"suggestion accepted T{rec.track_id} role={rec.role} "
                            "(visibility=uncertain)")
                    except AnnotationError as e:
                        session.messages.append(str(e)[:60])
            elif key == 26:  # ctrl-z undo is not supported; tell the user
                session.messages.append("no undo: re-edit the box/label instead")
    except KeyboardInterrupt:
        session.save()
        print("\ninterrupted; saved")
    finally:
        session.save()
        cv2.destroyAllWindows()
    return 0


ANNOTATOR_ID = "?"

if __name__ == "__main__":
    raise SystemExit(main())
