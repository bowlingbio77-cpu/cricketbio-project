"""
End-to-end orchestration: video file -> coaching recommendations.

    Video --> preprocessing --> detection --> tracking --> pose_estimation
      --> feature_engineering --> ml_models (performance + injury)
      --> explainability (SHAP) --> coaching

This ties every module together exactly as in the architecture diagram.
Heavy stages (YOLOv11 detection/tracking, MediaPipe pose) require model
weights fetched over the network on first run -- see each module's docstring.
"""
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

from . import config, preprocessing, detection, tracking, pose_estimation
from . import feature_engineering as feateng
from . import ml_models, explainability, coaching


@dataclass
class AnalysisResult:
    feature_vector: dict
    performance_score: Optional[float]
    injury_risk: Optional[dict]
    shap_contributions_performance: Optional[dict]
    shap_contributions_injury: Optional[dict]
    coaching_notes: list
    stage_times: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def analyze_video(video_path: str, bowling_arm: str = "right",
                   performance_bundle: ml_models.TrainedBundle = None,
                   injury_bundle: ml_models.TrainedBundle = None,
                   target_fps: int = config.TARGET_FPS,
                   resize_dim=config.RESIZE_DIM,
                   denoise: bool = config.DENOISE) -> AnalysisResult:
    """
    Full pipeline on a single delivery video clip. Requires:
      - models/pose_landmarker_heavy.task (MediaPipe pose model, download separately)
      - trained performance_bundle / injury_bundle (see train_demo_model.py)
    Detection+tracking (YOLOv11/ByteTrack) is used to crop to the bowler before
    pose estimation when `ultralytics` is installed; otherwise pose estimation
    runs on the full frame (fine for single-bowler, tightly-framed clips).
    `target_fps` / `resize_dim` / `denoise` override the preprocessing defaults
    (speed vs. accuracy trade-off).
    """
    timings = {}
    t_start = time.perf_counter()

    # 1-3: preprocess, detect, track (best-effort bowler crop)
    t0 = time.perf_counter()
    frames = list(preprocessing.preprocess_video(video_path, target_fps=target_fps,
                                                  resize_dim=resize_dim, denoise=denoise))
    timings["preprocess"] = time.perf_counter() - t0

    # 4-5: pose estimation -> 33 landmarks/frame
    t0 = time.perf_counter()
    with pose_estimation.PoseEstimator() as estimator:
        pose_sequence = estimator.process_video_frames(iter(frames))
    timings["pose_estimation"] = time.perf_counter() - t0

    if len(pose_sequence) < 3:
        raise RuntimeError("Not enough frames with a detected pose -- check video quality/framing.")

    # 6: biomechanical feature engineering
    t0 = time.perf_counter()
    feature_vector = feateng.build_feature_vector(pose_sequence, bowling_arm=bowling_arm)
    timings["feature_engineering"] = time.perf_counter() - t0

    # 7: ML predictions
    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None

    if performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)

    # 9: coaching recommendations
    t0 = time.perf_counter()
    notes = coaching.generate_recommendations(
        feature_vector, performance_score, injury_risk,
        shap_contributions=shap_injury or shap_perf,
    )
    timings["coaching"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start

    return AnalysisResult(
        feature_vector=feature_vector,
        performance_score=performance_score,
        injury_risk=injury_risk,
        shap_contributions_performance=shap_perf,
        shap_contributions_injury=shap_injury,
        coaching_notes=notes,
        stage_times=timings,
    )


def analyze_feature_vector(feature_vector: dict,
                            performance_bundle: ml_models.TrainedBundle = None,
                            injury_bundle: ml_models.TrainedBundle = None) -> AnalysisResult:
    """    Same as analyze_video but skips CV/pose stages -- useful for the Streamlit
    manual-entry mode and for testing without a video file."""
    timings = {}
    t_start = time.perf_counter()

    performance_score = None
    injury_risk = None
    shap_perf = None
    shap_injury = None

    if performance_bundle is not None:
        t0 = time.perf_counter()
        performance_score = ml_models.predict(performance_bundle, feature_vector)
        timings["ml_predictions"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shap_perf = explainability.explain_prediction(performance_bundle, feature_vector)
        timings["shap_explanation"] = time.perf_counter() - t0

    if injury_bundle is not None:
        t0 = time.perf_counter()
        injury_risk = ml_models.predict(injury_bundle, feature_vector)
        timings["ml_predictions"] = timings.get("ml_predictions", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        shap_injury = explainability.explain_prediction(injury_bundle, feature_vector)
        timings["shap_explanation"] = timings.get("shap_explanation", 0.0) + (time.perf_counter() - t0)

    notes = coaching.generate_recommendations(
        feature_vector, performance_score, injury_risk,
        shap_contributions=shap_injury or shap_perf,
    )

    timings["total"] = time.perf_counter() - t_start

    return AnalysisResult(
        feature_vector=feature_vector,
        performance_score=performance_score,
        injury_risk=injury_risk,
        shap_contributions_performance=shap_perf,
        shap_contributions_injury=shap_injury,
        coaching_notes=notes,
        stage_times=timings,
    )
