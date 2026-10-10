"""
Interactive review tool for the auto-labeled bowler/batsman dataset.

Shows each labeled frame with the two role boxes (red = bowler, blue = batsman)
and lets you:
  * set the active role            [1] bowler   [2] batsman
  * add / replace a box            drag a rectangle over the person (uses active role)
  * delete the box under a click   draw over it, then press [d]
  * jump frames                    [] prev  [] next   g=goto
  * save all                       [s]               save + quit [q]

Corrected labels are written to data/bowler_batsman_dataset/reviewed/ -- the
trainer prefers reviewed/ over auto_labeled/ when present.

Usage:
    python scripts/review_role_labels.py
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTO_DIR = os.path.join("data", "bowler_batsman_dataset", "auto_labeled")
REVIEW_DIR = os.path.join("data", "bowler_batsman_dataset", "reviewed")
CLASS_NAMES = ["bowler", "batsman"]
COLORS = {0: (0, 0, 255), 1: (255, 0, 0)}  # BGR


def _norm(box):
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [max(0.0, x1), max(0.0, y1), x2, y2]


def _to_norm(nx, ny, nw, nh):
    x1, y1 = max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))
    nw, nh = max(0.0, min(1.0 - x1, nw)), max(0.0, min(1.0 - y1, nh))
    return [x1, y1, nw, nh]


def _to_abs(box, w, h):
    nx, ny, nw, nh = box
    return [int(nx * w), int(ny * h), int((nx + nw) * w), int((ny + nh) * h)]


def _box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def load_entries(src):
    img_dir = os.path.join(src, "images")
    lbl_dir = os.path.join(src, "labels")
    entries = []
    for img_name in sorted(os.listdir(img_dir)):
        base, ext = os.path.splitext(img_name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    boxes.append([int(float(parts[0])), [float(v) for v in parts[1:]]])
        entries.append((os.path.join(img_dir, img_name), boxes))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=AUTO_DIR)
    ap.add_argument("--out", default=REVIEW_DIR)
    args = ap.parse_args()

    entries = load_entries(args.src)
    if not entries:
        sys.exit("no labeled frames found -- run prepare_bowler_batsman_dataset.py first")

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    win = "review_roles"
    cv2.namedWindow(win)
    drag = {"active": False, "start": None, "cur": None, "sel": None}
    state = {"role": 0, "msg": "active role: bowler (1)", "dirty": False}
    last_drag_center = None

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["active"] = True
            drag["start"] = (x, y)
            drag["cur"] = (x, y)
            drag["sel"] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag["active"]:
            drag["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["active"]:
            drag["cur"] = (x, y)
            drag["sel"] = _norm([drag["start"][0], drag["start"][1],
                                 drag["cur"][0], drag["cur"][1]])
            drag["active"] = False

    cv2.setMouseCallback(win, on_click)

    idx = 0

    def render(img_path, boxes):
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]
        for cls, box in boxes:
            x1, y1, x2, y2 = _to_abs(box, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cls], 2)
            cv2.putText(frame, CLASS_NAMES[cls], (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2, cv2.LINE_AA)
        if drag["active"] and drag["start"] and drag["cur"]:
            x1, y1 = drag["start"]
            x2, y2 = drag["cur"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(frame, f"{os.path.basename(img_path)}  [{idx}/{len(entries)-1}]",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, state["msg"], (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 255), 1, cv2.LINE_AA)
        return frame

    def save_all():
        for i in range(len(entries)):
            img_path, boxes = entries[i]
            img_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(out_img, img_name), cv2.imread(img_path))
            with open(os.path.join(out_lbl, os.path.splitext(img_name)[0] + ".txt"), "w") as f:
                for cls, box in boxes:
                    f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        print(f"  saved {len(entries)} frames -> {args.out}")

    while True:
        img_path, boxes = entries[idx]
        frame = render(img_path, boxes)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if drag["sel"] is not None:
            sel = drag["sel"]
            drag["sel"] = None
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            box = _to_norm(sel[0] / w, sel[1] / h,
                           (sel[2] - sel[0]) / w, (sel[3] - sel[1]) / h)
            if box[2] <= 0 or box[3] <= 0:
                pass
            else:
                boxes.append([state["role"], box])
                last_drag_center = (sel[0] / w + sel[2] / w) / 2, (sel[1] / h + sel[3] / h) / 2
                if state["role"] == 1:
                    state["msg"] = f"added {CLASS_NAMES[state['role']]} box (press [d] to delete)"
                else:
                    state["msg"] = f"added {CLASS_NAMES[state['role']]} box"
                state["dirty"] = True

        if key == ord("q"):
            break
        elif key == ord("s"):
            save_all()
            state["msg"] = "saved all"
        elif key == ord("1"):
            state["role"] = 0
            state["msg"] = "active role: bowler (1)"
        elif key == ord("2"):
            state["role"] = 1
            state["msg"] = "active role: batsman (2)"
        elif key == ord("d") and last_drag_center is not None:
            cx, cy = last_drag_center
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            before = len(boxes)
            keep = []
            for cls, box in boxes:
                x1, y1, x2, y2 = _to_abs(box, w, h)
                bcx, bcy = _box_center([x1, y1, x2, y2])
                # normalized center distance from the last drag
                d = (((bcx / w) - cx) ** 2 + ((bcy / h) - cy) ** 2) ** 0.5
                if d <= 0.15:
                    continue
                keep.append([cls, box])
            boxes[:] = keep
            if len(boxes) < before:
                state["msg"] = f"deleted {before - len(boxes)} box(es) near last drag"
                state["dirty"] = True
            else:
                state["msg"] = "no box near last drag to delete"
        elif key == ord("]"):
            idx = min(len(entries) - 1, idx + 1)
        elif key == ord("["):
            idx = max(0, idx - 1)
        elif key == ord("g"):
            try:
                v = int(input("frame index: "))
                if 0 <= v < len(entries):
                    idx = v
            except (ValueError, EOFError):
                pass

    cv2.destroyAllWindows()
    save_all()
    print("Done. Reviewed labels in", args.out)


if __name__ == "__main__":
    main()