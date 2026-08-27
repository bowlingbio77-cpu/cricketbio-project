"""Quick smoke test of the video pipeline to find what breaks."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, preprocessing, tracking, pose_estimation
from src import ball_tracking_v2 as ball_tracking
from src import feature_engineering as feateng
from src import ml_models
import numpy as np
import cv2

video_path = os.path.join(os.path.dirname(__file__), "test_synthetic.mp4")

# Create a tiny synthetic test video (no real bowler, just to test each stage)
if not os.path.exists(video_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 360))
    for i in range(60):
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print("Synthetic test video created")

# Step 1: preprocess
try:
    frames = list(preprocessing.preprocess_video(video_path, target_fps=20, resize_dim=(640, 360)))
    print(f"1. Preprocessing OK: {len(frames)} frames")
except Exception as e:
    print(f"1. Preprocessing FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: ball tracking
try:
    track, track_stats = ball_tracking.track_ball(frames)
    print(f"2. Ball tracking: {len(track)} points, outcome={track_stats.get('outcome')}")
except Exception as e:
    print(f"2. Ball tracking FAILED: {e}")
    traceback.print_exc()

# Step 3: bowler tracking
try:
    tracker = tracking.BowlerTracker()
    tracks = tracker.track_frames([(idx, fr) for idx, ts, fr in frames])
    bowler = tracking.select_bowler_track(tracks)
    print(f"3. Bowler tracking: {len(tracks)} tracks, bowler={bowler}")
except Exception as e:
    print(f"3. Bowler tracking FAILED: {e}")
    traceback.print_exc()

# Step 4: pose estimation
try:
    with pose_estimation.PoseEstimator() as estimator:
        pose_sequence = estimator.process_video_frames(iter(frames))
    print(f"4. Pose estimation: {len(pose_sequence)} pose frames")
except Exception as e:
    print(f"4. Pose estimation FAILED: {e}")
    traceback.print_exc()

# Step 5: feature engineering
if pose_sequence:
    try:
        feature_vector, diagnostics = feateng.analyze_delivery(
            pose_sequence, bowling_arm="right", camera_view="behind"
        )
        print(f"5. Feature engineering OK: reliable={diagnostics.get('reliable')}")
    except Exception as e:
        print(f"5. Feature engineering FAILED: {e}")
        traceback.print_exc()
else:
    print("5. Feature engineering SKIPPED (no pose frames)")

# Step 6: full pipeline
try:
    from src.pipeline import analyze_video
    perf_bundle = ml_models.load_bundle(os.path.join(config.MODEL_DIR, "performance_random_forest.joblib"))
    injury_bundle = ml_models.load_bundle(os.path.join(config.MODEL_DIR, "injury_random_forest.joblib"))
    result = analyze_video(
        video_path, bowling_arm="right",
        performance_bundle=perf_bundle, injury_bundle=injury_bundle,
        run_ml=True,
    )
    print(f"6. Full pipeline OK: perf={result.performance_score:.1f}, risk={result.injury_risk}")
except Exception as e:
    print(f"6. Full pipeline FAILED: {e}")
    traceback.print_exc()
