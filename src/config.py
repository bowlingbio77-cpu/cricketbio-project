"""
Central configuration for the Cricket Bowling Biomechanics AI pipeline.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# --- Video preprocessing ---
TARGET_FPS = 20
RESIZE_DIM = (640, 360)          # (width, height)
DENOISE = False

# --- Detection / Tracking ---
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo11n.pt")   # auto-downloaded by ultralytics on first run
DETECTION_CONF_THRESHOLD = 0.4
BOWLER_CLASS_ID = 0              # "person" class in COCO-pretrained YOLO
BYTETRACK_CONFIG = "bytetrack.yaml"  # shipped with ultralytics

# --- Pose estimation ---
# Path to the MediaPipe Pose Landmarker task file. Not bundled (network needed to fetch it).
# Download once with:
#   wget -O models/pose_landmarker_heavy.task \
#     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# The 33 MediaPipe Pose landmarks, in official index order.
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
assert len(POSE_LANDMARK_NAMES) == 33

# --- Biomechanical feature list (matches the architecture diagram) ---
FEATURE_NAMES = [
    "shoulder_rotation_deg",
    "elbow_flexion_deg",
    "wrist_angle_deg",
    "hip_rotation_deg",
    "knee_flexion_deg",
    "trunk_lean_deg",
    "stride_length_norm",
    "release_angle_deg",
    "angular_velocity_deg_s",
    "ground_contact_time_s",
]

# --- ML module ---
RANDOM_STATE = 42
PERFORMANCE_TARGET = "performance_score"   # regression, 0-100
# Injury risk is defined as: the chance that an injury occurs if the SAME
# action is performed long-term (repetitive exposure), graded by the SEVERITY
# of the injury that would occur. The training target is an ordinal outcome
# (0=no injury, 1=minor, 2=moderate, 3=severe) simulated over a per-row long-term
# exposure. The model reports P(injury) for the exposure chosen at inference,
# the expected severity, and a combined 0-1 risk score = P(injury) x severity.
INJURY_TARGET = "injury_severity"                        # ordinal 0-3 (no/minor/moderate/severe)
INJURY_EXPOSURE_FEATURE = "long_term_exposure_deliveries"  # deliveries performed over time
INJURY_EXPOSURE_DELIVERIES = 500      # default long-term exposure assumed at inference
INJURY_RISK_LOW = 0.33                # risk_score (0-1) below this -> "low" long-term risk
INJURY_RISK_HIGH = 0.66               # risk_score above this -> "high"

# Elbow flexion legal delivery threshold (ICC law of cricket: <=15 degrees extension)
ICC_ELBOW_EXTENSION_LIMIT_DEG = 15.0

# --- Multimodal Sports Injury Dataset (Kaggle: anjalibhegam/multimodal-sports-injury-dataset) ---
# 15,420 per-session rows from 156 athletes; target is 3-class injury risk.
SPORTS_INJURY_DATA = os.path.join(DATA_DIR, "multimodal_sports_injury_dataset.csv")
SPORTS_INJURY_TARGET = "injury_occurred"
SPORTS_INJURY_LABEL_MAP = {0: "healthy", 1: "low_risk", 2: "high_risk_injured"}
SPORTS_INJURY_GROUP_COL = "athlete_id"
SPORTS_INJURY_TIME_COL = "session_id"
SPORTS_INJURY_NUMERIC_FEATURES = [
    "heart_rate", "body_temperature", "hydration_level", "sleep_quality",
    "recovery_score", "stress_level", "muscle_activity", "joint_angles",
    "gait_speed", "cadence", "step_count", "jump_height",
    "ground_reaction_force", "range_of_motion", "ambient_temperature",
    "humidity", "altitude", "training_intensity", "training_duration",
    "training_load", "fatigue_index", "age", "bmi",
]
SPORTS_INJURY_CATEGORICAL_FEATURES = ["sport_type", "gender", "playing_surface"]
SPORTS_INJURY_SEQUENCE_WINDOW = 10       # sessions of context for sequence models
SPORTS_INJURY_ROLLING_WINDOW = 3         # short rolling stats (lag-based, no leakage)

# --- Cricket Injury Dataset (player-season level, one row per player per season) ---
CRICKET_INJURY_DATA = os.path.join(DATA_DIR, "cricket_injury_dataset.csv")
CRICKET_INJURY_TARGET = "injury_status"                       # binary 0/1
CRICKET_INJURY_GROUP_COL = "player_id"
CRICKET_INJURY_SEASON_COL = "season"
CRICKET_INJURY_NUMERIC_FEATURES = ["age", "match_exposure_hours", "player_match_days"]
CRICKET_INJURY_CATEGORICAL_FEATURES = [
    "playing_level", "playing_role", "match_format",
]
# Ordinal severity derived from match_days_lost: 0 none, 1 minor (<7 days), 2 major (>=7 days)
CRICKET_INJURY_SEVERITY_TARGET = "injury_severity"
CRICKET_INJURY_SEVERITY_LABEL_MAP = {0: "none", 1: "minor", 2: "major"}
CRICKET_INJURY_SEVERITY_DAYS = {"minor": 7}
# High-detail injury columns only present on injured rows (used for explainability, not as features)
CRICKET_INJURY_DETAIL_COLUMNS = ["injury_type", "body_site", "new_or_recurrent",
                                 "match_or_training", "recovered", "match_days_lost"]
