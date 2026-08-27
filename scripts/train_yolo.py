"""
Prepare YOLO training dataset from auto-labels + synthetic GT clips.
Then fine-tune YOLOv11-nano on cricket ball detection.

Splits data 80/20 train/val, creates data.yaml, trains for 50 epochs.
"""
import os
import json
import random
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTO_LABELS_DIR = os.path.join("data", "cricket_ball_dataset", "auto_labeled")
GT_CLIPS_DIR = os.path.join("data", "gt_clips")
DATASET_DIR = os.path.join("data", "cricket_ball_dataset", "yolo_dataset")
BALL_CLASS_ID = 0
VAL_SPLIT = 0.2
SEED = 42


def extract_gt_frames_as_labels():
    """Extract ball bounding boxes from synthetic GT clips (gt.json)."""
    frames_dir = os.path.join(DATASET_DIR, "gt_raw_images")
    labels_dir = os.path.join(DATASET_DIR, "gt_raw_labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    count = 0
    import cv2
    for clip_name in sorted(os.listdir(GT_CLIPS_DIR)):
        clip_dir = os.path.join(GT_CLIPS_DIR, clip_name)
        gt_path = os.path.join(clip_dir, "gt.json")
        clip_video = os.path.join(clip_dir, "clip.avi")
        if not os.path.isfile(gt_path) or not os.path.isfile(clip_video):
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        gt_frames = gt.get("frames", {})
        if not gt_frames:
            continue

        cap = cv2.VideoCapture(clip_video)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fstr = str(frame_idx)
            if fstr in gt_frames:
                ball = gt_frames[fstr]
                if isinstance(ball, dict):
                    cx, cy = ball.get("x", 0), ball.get("y", 0)
                    half = ball.get("half", 5)
                elif isinstance(ball, (list, tuple)) and len(ball) >= 2:
                    cx, cy = float(ball[0]), float(ball[1])
                    half = 5
                else:
                    frame_idx += 1
                    continue
                h, w = frame.shape[:2]

                img_name = f"gt_{clip_name}_f{frame_idx:04d}.png"
                lbl_name = f"gt_{clip_name}_f{frame_idx:04d}.txt"

                cv2.imwrite(os.path.join(frames_dir, img_name), frame)

                # YOLO format
                nx = cx / w
                ny = cy / h
                nw = (half * 2) / w
                nh = (half * 2) / h
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                nw = max(0.01, min(0.5, nw))
                nh = max(0.01, min(0.5, nh))

                with open(os.path.join(labels_dir, lbl_name), "w") as f:
                    f.write(f"{BALL_CLASS_ID} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                count += 1
            frame_idx += 1
        cap.release()

    print(f"  Extracted {count} labeled frames from {len(os.listdir(GT_CLIPS_DIR))} GT clips")
    return count


def build_dataset():
    """Combine auto-labels + GT into train/val split."""
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    # Collect all (image_path, label_path) pairs
    pairs = []

    # 1. Auto-labeled real clips
    auto_img_dir = os.path.join(AUTO_LABELS_DIR, "images")
    auto_lbl_dir = os.path.join(AUTO_LABELS_DIR, "labels")
    for fname in os.listdir(auto_img_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            img = os.path.join(auto_img_dir, fname)
            lbl = os.path.join(auto_lbl_dir, base + ".txt")
            if os.path.isfile(lbl):
                pairs.append((img, lbl, "auto"))

    # 2. Synthetic GT clips
    print("Extracting synthetic GT frames...")
    extract_gt_frames_as_labels()
    gt_img_dir = os.path.join(DATASET_DIR, "gt_raw_images")
    gt_lbl_dir = os.path.join(DATASET_DIR, "gt_raw_labels")
    for fname in os.listdir(gt_img_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            img = os.path.join(gt_img_dir, fname)
            lbl = os.path.join(gt_lbl_dir, base + ".txt")
            if os.path.isfile(lbl):
                pairs.append((img, lbl, "gt"))

    print(f"  Total pairs: {len(pairs)} (auto: {sum(1 for _,_,s in pairs if s=='auto')}, gt: {sum(1 for _,_,s in pairs if s=='gt')})")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Create directory structure
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i, (img, lbl, _) in enumerate(split_pairs):
            new_img = os.path.join(img_dir, f"frame_{i:06d}{os.path.splitext(img)[1]}")
            new_lbl = os.path.join(lbl_dir, f"frame_{i:06d}.txt")
            shutil.copy2(img, new_img)
            shutil.copy2(lbl, new_lbl)

    print(f"  Train: {len(train_pairs)} images, Val: {len(val_pairs)} images")

    # Write data.yaml
    data_yaml = f"""train: {os.path.abspath(os.path.join(DATASET_DIR, 'train', 'images'))}
val: {os.path.abspath(os.path.join(DATASET_DIR, 'val', 'images'))}

nc: 1
names: ['cricket_ball']
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml)
    print(f"  data.yaml: {yaml_path}")

    # Cleanup temp GT dirs
    shutil.rmtree(gt_img_dir, ignore_errors=True)
    shutil.rmtree(gt_lbl_dir, ignore_errors=True)

    return yaml_path, len(train_pairs), len(val_pairs)


def train_yolo(yaml_path, epochs=50, imgsz=640, batch=16):
    """Fine-tune YOLOv11-nano on cricket ball dataset."""
    from ultralytics import YOLO

    print(f"\nLoading YOLOv11-nano pretrained weights...")
    model = YOLO("yolo11n.pt")

    print(f"Fine-tuning for {epochs} epochs at {imgsz}px, batch={batch}...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="cricket_ball_v1",
        project=os.path.join("models", "runs"),
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True,
    )

    best_path = os.path.join("models", "runs", "cricket_ball_v1", "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_path}")
    return best_path


def main():
    print("=" * 60)
    print("STEP 1: Build YOLO Training Dataset")
    print("=" * 60)
    yaml_path, n_train, n_val = build_dataset()

    print(f"\n{'='*60}")
    print("STEP 2: Fine-tune YOLOv11-nano")
    print("=" * 60)
    best_weights = train_yolo(yaml_path, epochs=50, imgsz=640, batch=16)

    print(f"\n{'='*60}")
    print("STEP 3: Validate on held-out set")
    print("=" * 60)
    from ultralytics import YOLO
    model = YOLO(best_weights)
    metrics = model.val(data=yaml_path, imgsz=640)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    # Save results summary
    summary = {
        "train_images": n_train,
        "val_images": n_val,
        "epochs": 50,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "weights": best_weights,
    }
    summary_path = os.path.join("models", "runs", "cricket_ball_v1", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
