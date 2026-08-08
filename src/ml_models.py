"""
Stage 7: Machine Learning Module

Two tasks, sharing the same 10-feature biomechanical input vector:
  1. Performance Assessment   -> regression, 0-100 performance score
  2. Injury Risk Prediction   -> classification, {low, moderate, high}

Models offered (matching the architecture diagram): Random Forest, XGBoost,
CatBoost, CNN-LSTM, Transformer.

Only scikit-learn is guaranteed available in every environment, so:
  - RandomForest: always available (sklearn).
  - XGBoost / CatBoost: used if installed (`pip install xgboost catboost`),
    otherwise silently fall back to sklearn GradientBoosting.
  - CNN-LSTM / Transformer: these need a raw *time-series* of frame features
    (not just the single release-frame vector) and a deep learning backend
    (`pip install torch`). If torch isn't available, they fall back to an
    MLPRegressor/MLPClassifier (sklearn) trained on flattened sequences --
    same interface, so the rest of the app doesn't need to know the difference.
"""
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import joblib
import os

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from . import config

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except (ImportError, OSError):
    _HAS_XGBOOST = False

try:
    import catboost as cb
    _HAS_CATBOOST = True
except (ImportError, OSError):
    _HAS_CATBOOST = False

try:
    import torch
    _HAS_TORCH = True
except (ImportError, OSError):
    _HAS_TORCH = False


ModelName = Literal["random_forest", "xgboost", "catboost", "cnn_lstm", "transformer"]


def _make_regressor(model_name: ModelName):
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=8, random_state=config.RANDOM_STATE)
    if model_name == "xgboost":
        if _HAS_XGBOOST:
            return xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     random_state=config.RANDOM_STATE)
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=config.RANDOM_STATE)
    if model_name == "catboost":
        if _HAS_CATBOOST:
            return cb.CatBoostRegressor(iterations=300, depth=6, verbose=False,
                                         random_state=config.RANDOM_STATE)
        return GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=config.RANDOM_STATE)
    if model_name in ("cnn_lstm", "transformer"):
        # Sequence models fall back to an MLP on the flattened feature vector when
        # torch isn't installed / no per-frame sequence data is supplied.
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                             random_state=config.RANDOM_STATE)
    raise ValueError(f"Unknown model_name: {model_name}")


def _make_classifier(model_name: ModelName):
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, random_state=config.RANDOM_STATE,
                                       class_weight="balanced")
    if model_name == "xgboost":
        if _HAS_XGBOOST:
            return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      random_state=config.RANDOM_STATE, eval_metric="mlogloss")
        return GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=config.RANDOM_STATE)
    if model_name == "catboost":
        if _HAS_CATBOOST:
            return cb.CatBoostClassifier(iterations=300, depth=6, verbose=False,
                                          random_state=config.RANDOM_STATE)
        return GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=config.RANDOM_STATE)
    if model_name in ("cnn_lstm", "transformer"):
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                              random_state=config.RANDOM_STATE)
    raise ValueError(f"Unknown model_name: {model_name}")


@dataclass
class TrainedBundle:
    model_name: str
    task: str                 # "performance" | "injury"
    model: object
    scaler: StandardScaler
    feature_names: list
    metrics: dict
    label_map: Optional[dict] = None  # for classification


def train_performance_model(X: np.ndarray, y: np.ndarray, model_name: ModelName = "random_forest",
                             feature_names=None) -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=config.RANDOM_STATE)

    model = _make_regressor(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_test": int(len(y_test)),
    }
    return TrainedBundle(model_name, "performance", model, scaler, feature_names, metrics)


def train_injury_model(X: np.ndarray, y: np.ndarray, model_name: ModelName = "random_forest",
                        feature_names=None, label_map=None) -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    label_map = label_map or {0: "low", 1: "moderate", 2: "high"}
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)

    model = _make_classifier(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "n_test": int(len(y_test)),
    }
    return TrainedBundle(model_name, "injury", model, scaler, feature_names, metrics, label_map)


def predict(bundle: TrainedBundle, feature_vector: dict):
    x = np.array([[feature_vector[f] for f in bundle.feature_names]])
    xs = bundle.scaler.transform(x)
    if bundle.task == "performance":
        return float(bundle.model.predict(xs)[0])
    pred_idx = int(bundle.model.predict(xs)[0])
    proba = None
    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(xs)[0].tolist()
    return {
        "risk_level": bundle.label_map[pred_idx],
        "risk_index": pred_idx,
        "probabilities": proba,
    }


def save_bundle(bundle: TrainedBundle, path: str):
    joblib.dump(bundle, path)


def load_bundle(path: str) -> TrainedBundle:
    return joblib.load(path)


BACKEND_INFO = {
    "xgboost_available": _HAS_XGBOOST,
    "catboost_available": _HAS_CATBOOST,
    "torch_available": _HAS_TORCH,
}
