"""Interactive ball-labeling tool for real cricket clips.

Click on the ball in each frame to record its centre; frames you skip are
treated as "ball not visible" (they don't count in eval recall). Optionally
seed the labels with the v2 tracker's own guess so you only correct it.

Output layout (consumed directly by scripts/evaluate_ball_tracking.py):
  data/gt_clips/<name>/
    source_path.txt   absolute path of the source video
    gt.json           {"fps", "release_idx"?, "impact_idx"?, "frames": {idx: [x, y]}}

Controls:
  left-click          set ball position at current frame, then advance
  n / right-click     skip current frame (ball not visible), advance
  p / b / <-          1 frame back            [ ]        jump -5/+5
  g                   jump to a frame number   u          remove label at frame
  r                   mark current frame as release       i  mark as impact
  s                   save                    q          save + quit

Usage:
  python scripts/label_ball.py --video corrected_all_data/bowling/xxx.avi --name ball_1
  python scripts/label_ball.py --video corrected_all_data/bowling/xxx.avi --seed --name ball_1
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import preprocessing                 # noqa: E402
from src import ball_tracking_v2 as bt2       # noqa: E402


def load_ball(video_path):
    """Return (name, frames) at tracker coordinate space."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    frames = list(preprocessing.preprocess_video(video_path))
    return name, frames


def run_seed(frames):
    tracker = bt2.BallTracker(model=None)
    track, stats = tracker.track(frames, fps=20.0)
    return {p.frame_idx: (p.x, p.y) for p in track}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="source video (avi/mp4)")
    ap.add_argument("--name", help="clip id (default: video basename)")
    ap.add_argument("--out", default=os.path.join("data", "gt_clips"),
                    help="output directory for the labeled clip")
    ap.add_argument("--seed", action="store_true",
                    help="preload the v2 tracker's guess and correct it")
    ap.add_argument("--list", action="store_true",
                    help="list candidate videos and exit")
    args = ap.parse_args()

    if args.list:
        root = "corrected_all_data/bowling"
        for v in sorted(os.listdir(root)):
            print(os.path.join(root, v))
        return

    if not args.video or not os.path.exists(args.video):
        sys.exit("provide an existing --video (or --list to see candidates)")

    name, frames = load_ball(args.video)
    name = args.name or name
    clip_dir = os.path.join(args.out, name)
    os.makedirs(clip_dir, exist_ok=True)

    labels = {}
    release_idx = impact_idx = None
    if args.seed:
        seed, seed_stats = run_seed(frames)
        labels.update(seed)
        print(f"  [seed] preloaded {len(seed)} tracker points "
              f"(outcome={seed_stats.get('outcome')}, rel={seed_stats.get('release_idx')}, "
              f"imp={seed_stats.get('impact_idx')}) -- correct them, or 'u' to clear a frame")
    print(f"  [{name}] {len(frames)} frames at 640x360@20 -- click the ball, "
          f"'n' to skip, 'q' to save+quit")

    pos = 0
    n = len(frames)
    win = "label_ball"
    cv2.namedWindow(win)

    click_pos = [None]

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_pos[0] = "skip"

    cv2.setMouseCallback(win, on_click)

    while True:
        if pos < 0:
            pos = 0
        if pos >= n:
            pos = n - 1
        _idx, _ts, frame = frames[pos]
        img = frame.copy()
        cv2.putText(img, f"frame {pos}/{n - 1}  labeled={len(labels)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        if release_idx is not None:
            cv2.putText(img, f"RELEASE {release_idx}", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if impact_idx is not None:
            cv2.putText(img, f"IMPACT {impact_idx}", (8, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if pos in labels:
            x, y = labels[pos]
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
        # faint trail of nearby labels (for continuity)
        for p in range(max(0, pos - 6), min(n, pos + 7)):
            if p in labels:
                x, y = labels[p]
                cv2.circle(img, (int(x), int(y)), 1, (255, 200, 0), -1)

        cv2.imshow(win, img)
        key = cv2.waitKey(25) & 0xFF

        hit = click_pos[0]
        if hit is not None:
            click_pos[0] = None
            if hit == "skip":
                pos += 1
            else:
                x, y = hit
                labels[pos] = (float(x), float(y))
                pos += 1

        if key == ord("q"):
            break
        elif key == ord("s"):
            save(labels, release_idx, impact_idx, clip_dir, args.video)
        elif key == ord("g"):
            try:
                pos = int(input("frame: "))
            except (ValueError, EOFError):
                pass
        elif key == ord("u"):
            labels.pop(pos, None)
        elif key in (ord("n"),):
            pos += 1
        elif key in (ord("p"), ord("b")):
            pos -= 1
        elif key == ord("["):
            pos -= 5
        elif key == ord("]"):
            pos += 5
        elif key == ord("r"):
            release_idx = pos
        elif key == ord("i"):
            impact_idx = pos

    cv2.destroyAllWindows()
    save(labels, release_idx, impact_idx, clip_dir, args.video)
    print(f"  saved {len(labels)} labeled frames -> {clip_dir}")


def save(labels, release_idx, impact_idx, clip_dir, video_path):
    gt = {"fps": 20.0, "frames": {str(k): [float(v[0]), float(v[1])]
                                  for k, v in sorted(labels.items())}}
    if release_idx is not None:
        gt["release_idx"] = release_idx
    if impact_idx is not None:
        gt["impact_idx"] = impact_idx
    with open(os.path.join(clip_dir, "gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    with open(os.path.join(clip_dir, "source_path.txt"), "w") as f:
        f.write(os.path.abspath(video_path) + "\n")


if __name__ == "__main__":
    main()
