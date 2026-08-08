"""
Stage 1: Video Preprocessing
Frame extraction, resizing, and denoising for cricket bowling videos.
"""
import os
import cv2
import numpy as np
from . import config


def extract_frames(video_path: str, target_fps: int = config.TARGET_FPS):
    """
    Generator that yields (frame_index, timestamp_sec, frame) tuples,
    sub-sampled/interpolated to target_fps.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, round(source_fps / target_fps))

    idx = 0
    out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            timestamp = idx / source_fps
            yield out_idx, timestamp, frame
            out_idx += 1
        idx += 1
    cap.release()


def resize_frame(frame: np.ndarray, dim=config.RESIZE_DIM) -> np.ndarray:
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def denoise_frame(frame: np.ndarray) -> np.ndarray:
    """Fast non-local-means denoising, tuned to preserve edges (needed for pose landmarks)."""
    return cv2.fastNlMeansDenoisingColored(frame, None, h=5, hColor=5,
                                            templateWindowSize=7, searchWindowSize=21)


def preprocess_video(video_path: str, target_fps: int = config.TARGET_FPS,
                      resize_dim=config.RESIZE_DIM, denoise: bool = config.DENOISE):
    """
    Full preprocessing pass. Yields (frame_index, timestamp_sec, processed_frame).
    """
    for idx, ts, frame in extract_frames(video_path, target_fps):
        frame = resize_frame(frame, resize_dim)
        if denoise:
            frame = denoise_frame(frame)
        yield idx, ts, frame


def preprocess_video_to_array(video_path: str, **kwargs):
    """Convenience wrapper returning (timestamps, frames_array)."""
    timestamps, frames = [], []
    for idx, ts, frame in preprocess_video(video_path, **kwargs):
        timestamps.append(ts)
        frames.append(frame)
    return np.array(timestamps), np.array(frames)
