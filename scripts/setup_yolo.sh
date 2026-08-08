#!/usr/bin/env bash
# One-time YOLOv11 setup for the bowler-detection/tracking stages.
# Run this on a machine with internet access (this project's dev sandbox
# had none, so this was written but not executed here -- see README).
set -e

echo "== Installing ultralytics =="
pip install --upgrade ultralytics

echo ""
echo "== Downloading YOLO11 nano weights (auto-cached by ultralytics) =="
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')   # downloads to the ultralytics cache on first use
print('Loaded:', model.model_name if hasattr(model, \"model_name\") else 'yolo11n.pt')
print('Classes:', model.names.get(0), '(class 0 should be person)')
"

echo ""
echo "== Verifying ByteTrack tracker config is available =="
python3 -c "
from ultralytics.utils import checks
import ultralytics, os
cfg_dir = os.path.join(os.path.dirname(ultralytics.__file__), 'cfg', 'trackers')
print('Tracker configs found:', os.listdir(cfg_dir))
assert 'bytetrack.yaml' in os.listdir(cfg_dir), 'bytetrack.yaml missing!'
print('bytetrack.yaml OK')
"

echo ""
echo "Done. Test it against a real clip with:"
echo "  python scripts/test_yolo_detection.py path/to/your_video.mp4"
