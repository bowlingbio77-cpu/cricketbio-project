"""
Download the DeepSportradar Cricket Bowl Release Dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Get API token from https://www.kaggle.com/settings
3. Save kaggle.json to ~/.kaggle/kaggle.json

Usage:
    python evaluation/single_video_benchmark/download_dataset.py
"""
import os
import subprocess
import sys
import json
from pathlib import Path

DATASET_SLUG = "dzambrano/cricket-bowlrelease-dataset"
DOWNLOAD_DIR = Path(__file__).parent / "dataset"
EXTRACT_DIR = DOWNLOAD_DIR


def check_kaggle_credentials():
    """Check if Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found.")
        print(f"Expected: {kaggle_json}")
        print()
        print("To set up:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print(f"3. Save the downloaded kaggle.json to {kaggle_json.parent}")
        return False
    return True


def download_dataset():
    """Download the dataset from Kaggle."""
    if not check_kaggle_credentials():
        return False

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_SLUG}")
    print(f"Target directory: {DOWNLOAD_DIR}")
    print()

    # Download
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(DOWNLOAD_DIR),
        "--unzip"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        return False

    print("Download complete!")
    print()

    # List contents
    print("Dataset contents:")
    for item in sorted(DOWNLOAD_DIR.rglob("*")):
        if item.is_file():
            rel = item.relative_to(DOWNLOAD_DIR)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {rel} ({size_mb:.1f} MB)")

    return True


def verify_dataset():
    """Verify the downloaded dataset has expected structure."""
    print()
    print("Verifying dataset structure...")

    # Check for video files
    videos = list(DOWNLOAD_DIR.rglob("*.mp4"))
    if not videos:
        videos = list(DOWNLOAD_DIR.rglob("*.avi"))

    # Check for annotation files
    annotations = list(DOWNLOAD_DIR.rglob("*.json"))

    print(f"  Videos found: {len(videos)}")
    print(f"  Annotation files found: {len(annotations)}")

    if videos:
        print()
        print("Sample videos:")
        for v in videos[:5]:
            rel = v.relative_to(DOWNLOAD_DIR)
            print(f"  {rel}")

    if annotations:
        print()
        print("Sample annotations:")
        for a in annotations[:5]:
            rel = a.relative_to(DOWNLOAD_DIR)
            # Try to read and show structure
            try:
                with open(a) as f:
                    data = json.load(f)
                keys = list(data.keys())
                print(f"  {rel} (keys: {keys})")
            except Exception as e:
                print(f"  {rel} (error reading: {e})")

    return len(videos) > 0 and len(annotations) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("DeepSportradar Cricket Bowl Release Dataset Downloader")
    print("=" * 60)
    print()

    success = download_dataset()
    if success:
        verify_dataset()
        print()
        print("Ready to run validation:")
        print("  python evaluation/single_video_benchmark/run_validation.py")
    else:
        print()
        print("Download failed. Please check your Kaggle credentials.")
        sys.exit(1)
