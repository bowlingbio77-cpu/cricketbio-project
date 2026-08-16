"""Generate synthetic ball-tracking clips with known ground truth.

Each clip is 640x360 @ 20 fps (matches src.config so the tracker sees the
same coordinate space as real clips) and contains a red cricket ball on a
known trajectory with a release point and (optionally) an impact deflection.
Clips can also include realistic distractors: a bowler-ish person, a growing
person (box-growth false track), a fixed round object (stationary false
track), and mid-flight occlusions (re-seed test).

Output, per clip under data/gt_clips/<name>/:
  clip.avi   rendered frames (exactly what the tracker will be run on)
  gt.json    {"fps", "release_idx", "impact_idx", "frames": {idx: [x, y]}}
  spec.json  oracle-detector parameters used by the eval harness

Usage: python scripts/gen_gt_clips.py [--out data/gt_clips] [--seed 42]
"""
import argparse
import json
import math
import os
import random

import cv2
import numpy as np

W, H, FPS = 640, 360, 20


def make_background(rng):
    """Static textured pitch + grass + crowd strip. Static => no motion noise."""
    nrng = np.random.default_rng(rng.randrange(0, 2**31))
    grad = np.zeros((H, W, 3), np.float32)
    grad[:, :, 0] = np.linspace(45, 30, H)[:, None]
    grad[:, :, 1] = np.linspace(120, 95, H)[:, None]
    grad[:, :, 2] = np.linspace(38, 30, H)[:, None]
    img = grad + nrng.normal(0, 4, (H, W, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (180, 110), (460, 345), (185, 165, 120), -1)  # pitch
    cv2.rectangle(img, (180, 110), (460, 345), (120, 100, 70), 1)
    for _ in range(160):  # crowd along the top
        cv2.circle(img, (rng.randint(0, W), rng.randint(4, 42)),
                   rng.randint(1, 3),
                   (rng.randint(25, 55), rng.randint(35, 65), rng.randint(45, 75)), -1)
    for _ in range(30):  # field marks
        cv2.line(img, (rng.randint(0, W), rng.randint(60, 350)),
                 (rng.randint(0, W), rng.randint(60, 350)),
                 (60, 95, 55), 1)
    return img


def draw_person(img, cx, cy, height, jitter=0, rng=None):
    """A simple bowler-ish figure (legs + torso + head). Box ~ height*0.3 wide."""
    jx = int(jitter) if jitter else 0
    jy = int(jitter) if jitter else 0
    torso_h = int(height * 0.42)
    torso_w = int(height * 0.16)
    cv2.ellipse(img, (cx + jx, cy - torso_h // 2), (torso_w, torso_h),
                0, 0, 360, (225, 225, 235), -1)
    cv2.circle(img, (cx + jx, cy - int(height * 0.70) + jy),
               int(height * 0.10), (170, 150, 140), -1)
    cv2.rectangle(img, (cx - torso_w, cy), (cx + torso_w, cy + int(height * 0.10)),
                  (40, 40, 45), -1)
    return img


def draw_ball(img, x, y, r, color=(35, 40, 170), seam=(245, 245, 245)):
    """A solid dark-red cricket ball.

    Solid fill only (no outline, no seam ellipses): a seam or outline renders
    low-contrast pixels INSIDE the disk, so the motion diff-blob of a moving
    ball fragments into 7-20px pieces instead of one ~2r-wide contour. Real
    clips show stable ball-sized blobs (off_left_216: 5-9px), and the box
    growth heuristic depends on that.
    """
    cv2.circle(img, (int(x), int(y)), r, color, -1)
    cv2.circle(img, (int(x), int(y)), max(1, r - 2), seam, 2)


def ball_positions(cfg):
    """Return {frame_idx: [x, y]} for the ball's trajectory.

    The ball appears at `release_idx` (hand-off), travels with velocity v
    (px/frame) plus gravity g, optionally deflects once at `impact_idx`
    (rotate + scale), and is removed from GT while occluded.
    """
    x, y = cfg["start"]
    vx, vy = cfg["v"]
    g = cfg.get("g", 0.4)
    deflect = cfg.get("deflect")
    occlude = cfg.get("occlude")  # (start, end) frame range ball is hidden
    end = cfg.get("end_idx", 130)
    out = {}
    for i in range(cfg["release_idx"], end):
        if deflect is not None and i == cfg["impact_idx"]:
            ang = math.radians(deflect[0])
            c, s = math.cos(ang), math.sin(ang)
            vx, vy = (vx * c - vy * s) * deflect[1], (vx * s + vy * c) * deflect[1]
        x += vx
        y += vy
        vy += g
        if not (0 <= x < W and 0 <= y < H):
            break
        if occlude and occlude[0] <= i < occlude[1]:
            continue
        out[i] = [float(x), float(y)]
    return out


def build_clip(cfg, rng):
    """Render one clip; return (frames, gt_dict)."""
    name = cfg["name"]
    person = cfg.get("person")           # (cx, cy, height, growth_rate_per_frame)
    fixed_obj = cfg.get("fixed_obj")     # (cx, cy, r)
    arm = cfg.get("arm")                 # (frames_start, frames_end) moving blob near release
    occlude = cfg.get("occlude")

    frames = []
    for i in range(cfg.get("n_frames", 100)):
        img = make_background(rng)
        if person:
            cx, cy, h, growth = person
            h = h * (1.0 + growth * i)
            jitter = rng.uniform(-1.2, 1.2) if person[3] else rng.uniform(-0.4, 0.4)
            draw_person(img, cx, cy, h, jitter=jitter, rng=rng)
        if fixed_obj:
            fx, fy, fr = fixed_obj
            cv2.circle(img, (fx, fy), fr, (200, 205, 215), -1)
            cv2.circle(img, (fx, fy), fr, (0, 0, 0), 1)
        if arm and arm[0] <= i <= arm[1]:
            # a fast 'arm/hand' blob sweeping toward the release point
            t = (i - arm[0]) / max(1, arm[1] - arm[0])
            ax = int(arm[2][0] + (arm[3][0] - arm[2][0]) * t)
            ay = int(arm[2][1] + (arm[3][1] - arm[2][1]) * t)
            cv2.ellipse(img, (ax, ay), (9, 13), 0, 0, 360, (200, 140, 130), -1)

        positions = ball_positions(cfg)
        if i in positions and not (occlude and occlude[0] <= i < occlude[1]):
            x, y = positions[i]
            r = cfg.get("radius", 6)
            draw_ball(img, x, y, r)

        frames.append(img)

    return frames, positions


CLIP_SPECS = [
    # Per-frame speeds are capped near the ball's own diameter so consecutive
    # positions OVERLAP: a moving 12px ball then yields one contiguous
    # motion blob ~ its own size, matching what real clips produce
    # (off_left_216: attached motion widths 5-9, growth <=1.8x). At 20fps with
    # a 26px/frame ball the diff-blob fragments into 5-16px lunes -- an
    # artifact no real clip showed, and one that flips the growth heuristic.
    dict(name="fast_left_clean", start=(580, 170), v=(-12, -6), g=0.4,
         release_idx=40, impact_idx=52, deflect=(38, 0.55), radius=6),
    dict(name="slow_left_clean", start=(420, 200), v=(-9, -3), g=0.2,
         release_idx=20, impact_idx=44, deflect=(45, 0.5), radius=6),
    dict(name="far_small_ball", start=(540, 120), v=(-7, 1), g=0.15,
         release_idx=90, impact_idx=120, radius=4, end_idx=150),
    dict(name="short_flight", start=(400, 160), v=(-12, -4), g=0.4,
         release_idx=30, impact_idx=36, deflect=(30, 0.6), radius=6, n_frames=70),
    dict(name="ball_leaves_frame", start=(620, 200), v=(-14, -2), g=0.4,
         release_idx=10, impact_idx=None, radius=6, n_frames=70),
    dict(name="impact_sharp", start=(560, 180), v=(-12, -6), g=0.4,
         release_idx=30, impact_idx=48, deflect=(70, 0.45), radius=6),
    dict(name="with_person", start=(560, 170), v=(-12, -5), g=0.4,
         release_idx=40, impact_idx=52, deflect=(38, 0.55), radius=6,
         person=(300, 300, 60, 0.0)),
    dict(name="person_grows", start=(500, 200), v=(-12, -4), g=0.3,
         release_idx=30, impact_idx=48, deflect=(40, 0.5), radius=6,
         person=(280, 300, 55, 0.012)),           # grows 60px -> ~2.1x
    dict(name="stationary_object", start=(560, 170), v=(-12, -5), g=0.4,
         release_idx=40, impact_idx=52, deflect=(38, 0.55), radius=6,
         fixed_obj=(460, 290, 22)),               # fixed ball-like blob
    dict(name="occluded_short", start=(560, 180), v=(-12, -4), g=0.35,
         release_idx=30, impact_idx=50, deflect=(40, 0.5), radius=6,
         occlude=(40, 43)),                       # hidden 3 frames
    dict(name="occluded_long", start=(560, 180), v=(-12, -4), g=0.35,
         release_idx=30, impact_idx=70, deflect=(40, 0.5), radius=6,
         occlude=(44, 56)),                       # hidden 12 frames -> re-seed test
    dict(name="arm_sweep", start=(520, 200), v=(-12, -5), g=0.35,
         release_idx=35, impact_idx=55, deflect=(42, 0.5), radius=6,
         person=(360, 300, 62, 0.0),
         arm=(28, 40, (320, 205), (150, 180))),
    dict(name="bright_fast", start=(600, 140), v=(-14, -5), g=0.5,
         release_idx=20, impact_idx=40, deflect=(35, 0.5), radius=7),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "gt_clips"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for cfg in CLIP_SPECS:
        rng = random.Random(args.seed + len(cfg["name"]))
        frames, positions = build_clip(cfg, rng)
        clip_dir = os.path.join(args.out, cfg["name"])
        os.makedirs(clip_dir, exist_ok=True)

        vid_path = os.path.join(clip_dir, "clip.avi")
        vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()

        gt = {
            "fps": float(FPS),
            "release_idx": cfg["release_idx"],
            "impact_idx": cfg["impact_idx"],
            "frames": {str(k): v for k, v in sorted(positions.items())},
        }
        with open(os.path.join(clip_dir, "gt.json"), "w") as f:
            json.dump(gt, f, indent=1)

        spec = {
            "det_conf": 0.75,
            "conf_jitter": 0.18,
            "miss_prob": 0.12,          # YOLO intermittently misses the ball
            "low_conf_prob": 0.18,      # occasionally a sub-seed (0.2-0.29) det
            "seed": args.seed,
        }
        with open(os.path.join(clip_dir, "spec.json"), "w") as f:
            json.dump(spec, f, indent=1)

        print(f"{cfg['name']:22s} frames={len(positions):3d} "
              f"release={gt['release_idx']} impact={gt['impact_idx']}")


if __name__ == "__main__":
    main()
