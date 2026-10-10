"""
Stage 2: fine-tune YOLOv11-nano on the bowler/batsman 2-class dataset.

Uses the auto-labeled (or reviewed) frames produced by
scripts/prepare_bowler_batsman_dataset.py. A clip-aware split keeps frames of
one held-out clip entirely out of training so the val metrics are honest.

Usage:
    python scripts/train_bowler_batsman.py --epochs 60 --imgsz 640 --batch 16
    python scripts/train_bowler_batsman.py --use-reviewed   # prefer reviewed/ labels
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join("data", "bowler_batsman_dataset")
AUTO_DIR = os.path.join(DATASET_ROOT, "auto_labeled")
REVIEW_DIR = os.path.join(DATASET_ROOT, "reviewed")
YOLO_DIR = os.path.join(DATASET_ROOT, "yolo_dataset")
PREVIEW_DIR = os.path.join(DATASET_ROOT, "preview")

CLASS_NAMES = ["bowler", "batsman"]
SEED = 42


def pick_source(use_reviewed: bool) -> str:
    """Reviewed labels win when present and the user asks for them."""
    if use_reviewed and os.path.isdir(os.path.join(REVIEW_DIR, "images")):
        return REVIEW_DIR
    if os.path.isdir(os.path.join(AUTO_DIR, "images")):
        return AUTO_DIR
    sys.exit("no signed dataset found -- run prepare_bowler_batsman_dataset.py first "
             "(or --use-reviewed after reviewing)")


def build_dataset(src_dir: str) -> dict:
    """Copy images+labels into YOLO train/val, holding out whole clips for val."""
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    img_dir = os.path.join(src_dir, "images")
    lbl_dir = os.path.join(src_dir, "labels")

    frames = []
    for fname in os.listdir(img_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lbl = os.path.join(lbl_dir, base + ".txt")
        if os.path.isfile(lbl):
            clip = base.split("_")[0] + "_" + base.split("_")[1]
            frames.append((clip, os.path.join(img_dir, fname),
                           os.path.join(lbl_dir, base + ".txt")))

    random.seed(SEED)
    clips = sorted({c for c, _, _ in frames})
    n_val_clips = max(1, len(clips) // 4)
    val_clips = set(random.sample(clips, n_val_clips))

    os.makedirs(os.path.join(YOLO_DIR, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "val", "labels"), exist_ok=True)

    shuffled = list(frames)
    random.shuffle(shuffled)
    n_train = n_val = 0
    counts = {"train": [0, 0], "val": [0, 0]}  # [bowler, batsman] box counts
    for i, (clip, img, lbl) in enumerate(shuffled):
        split = "val" if clip in val_clips else "train"
        new_img = os.path.join(YOLO_DIR, split, "images", f"frame_{i:06d}.jpg")
        new_lbl = os.path.join(YOLO_DIR, split, "labels", f"frame_{i:06d}.txt")
        shutil.copy2(img, new_img)
        shutil.copy2(lbl, new_lbl)
        if split == "val":
            n_val += 1
        else:
            n_train += 1
        with open(new_lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    cls = int(float(parts[0]))
                    counts[split][cls] += 1

    data_yaml = os.path.join(YOLO_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(YOLO_DIR, 'train', 'images'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(YOLO_DIR, 'val', 'images'))}\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"  held-out val clips: {sorted(val_clips)}")
    print(f"  train frames: {n_train} (boxes bowler={counts['train'][0]}, batsman={counts['train'][1]})")
    print(f"  val frames:   {n_val} (boxes bowler={counts['val'][0]}, batsman={counts['val'][1]})")
    return {"yaml": data_yaml, "n_train": n_train, "n_val": n_val, **counts}


def train(yaml_path: str, epochs: int, imgsz: int, batch: int) -> str:
    from ultralytics import YOLO

    print(f"\nLoading pretrained YOLOv11-nano weights...")
    base = os.path.join("models", "yolo11n.pt")
    model = YOLO(base if os.path.exists(base) else "yolo11n.pt")

    trainer = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="bowler_batsman",
        project=os.path.join("models", "runs"),
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True,
    )
    save_dir = getattr(trainer, "save_dir", None) or os.path.join(
        "models", "runs", "bowler_batsman")
    best = os.path.join(str(save_dir), "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join("models", "runs", "bowler_batsman",
                            "weights", "best.pt")
    out = os.path.join("models", "bowler_batsman_yolo11n.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy2(best, out)
    print(f"  best weights ({best}) copied to {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--use-reviewed", action="store_true",
                    help="train on reviewed/ labels instead of auto_labeled/")
    args = ap.parse_args()

    src = pick_source(args.use_reviewed)
    print("=" * 60)
    print("STEP 1: Build YOLO 2-class dataset")
    print(f"  source: {src}")
    print("=" * 60)
    info = build_dataset(src)

    print("\n" + "=" * 60)
    print(f"STEP 2: Fine-tune YOLOv11-nano ({args.epochs} epochs, {args.imgsz}px)")
    print("=" * 60)
    best = train(info["yaml"], args.epochs, args.imgsz, args.batch)

    print("\n" + "=" * 60)
    print("STEP 3: Validate")
    print("=" * 60)
    from ultralytics import YOLO
    model = YOLO(best)
    metrics = model.val(data=info["yaml"], imgsz=args.imgsz)
    summary = {
        "train_frames": info["n_train"],
        "val_frames": info["n_val"],
        "epochs": args.epochs,
        "names": CLASS_NAMES,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "best_weights": best,
        "app_weights": os.path.join("models", "bowler_batsman_yolo11n.pt"),
    }
    with open(os.path.join(DATASET_ROOT, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {os.path.join(DATASET_ROOT, 'training_summary.json')}")
    print("\nPoint config.YOLO_WEIGHTS at models/bowler_batsman_yolo11n.pt to "
          "use it in the app (or run the wiring in scripts/use_role_model.py).")


if __name__ == "__main__":
    main()