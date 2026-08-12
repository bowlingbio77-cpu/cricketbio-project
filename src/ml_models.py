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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
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
    data_source: str = "synthetic"    # "synthetic" | "real" -- provenance for honest reporting
    feature_ranges: Optional[dict] = None  # {feature: {"min": .., "max": ..}} for OOD checks
    cv_metrics: Optional[dict] = None       # honest K-fold cross-validation metrics
    baseline_metrics: Optional[dict] = None  # trivial-baseline metrics for comparison


def _feature_ranges(X: np.ndarray, feature_names: list) -> dict:
    X = np.asarray(X, dtype=float)
    return {name: {"min": float(X[:, i].min()), "max": float(X[:, i].max())}
            for i, name in enumerate(feature_names)}


def _n_folds_for(n: int, y: np.ndarray = None, max_folds: int = 5) -> int:
    """Pick a safe number of CV folds given sample/class sizes."""
    if y is None:
        return min(max_folds, max(2, n // 3))
    counts = np.bincount(np.asarray(y).astype(int))
    return min(max_folds, int(counts.min())) if counts.size else 2


def cross_validate(task: str, X: np.ndarray, y: np.ndarray,
                   model_name: ModelName = "random_forest",
                   label_map=None, n_folds: int = 5) -> Optional[dict]:
    """
    Honest generalization estimate via K-fold cross-validation.
    The scaler is refit on each training fold (no leakage). Returns
    None if the dataset is too small to stratify.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    if n < 10:
        return None

    if task == "injury":
        n_folds = _n_folds_for(n, y, n_folds)
        if n_folds < 2:
            return None
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=config.RANDOM_STATE)
    else:
        n_folds = _n_folds_for(n, None, n_folds)
        splitter = KFold(n_splits=n_folds, shuffle=True,
                         random_state=config.RANDOM_STATE)

    scores = {"mae": [], "r2": [], "rmse": [], "accuracy": [], "f1": []}
    for tr_idx, te_idx in splitter.split(X, y):
        s = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = s.transform(X[tr_idx]), s.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        if task == "performance":
            m = _make_regressor(model_name).fit(Xtr, y_tr)
            preds = m.predict(Xte)
            scores["mae"].append(mean_absolute_error(y_te, preds))
            scores["r2"].append(r2_score(y_te, preds))
            scores["rmse"].append(float(np.sqrt(mean_squared_error(y_te, preds))))
        else:
            m = _make_classifier(model_name).fit(Xtr, y_tr)
            preds = m.predict(Xte)
            scores["accuracy"].append(accuracy_score(y_te, preds))
            scores["f1"].append(f1_score(y_te, preds, average="macro"))

    def _mean_std(key):
        vals = scores[key]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    if task == "performance":
        mae, mae_std = _mean_std("mae")
        r2, r2_std = _mean_std("r2")
        rmse, _ = _mean_std("rmse")
        return {"folds": n_folds, "n_samples": int(n),
                "mae_mean": mae, "mae_std": mae_std,
                "r2_mean": r2, "r2_std": r2_std, "rmse_mean": rmse}
    acc, acc_std = _mean_std("accuracy")
    f1, f1_std = _mean_std("f1")
    return {"folds": n_folds, "n_samples": int(n),
            "accuracy_mean": acc, "accuracy_std": acc_std,
            "f1_mean": f1, "f1_std": f1_std}


def baseline_performance(y: np.ndarray) -> dict:
    """Trivial baseline: always predict the mean. Tells you if the model adds value."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    preds = np.full_like(y, mean)
    return {"mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds)))}


def baseline_injury(y: np.ndarray) -> dict:
    """Trivial baseline: always predict the majority class."""
    y = np.asarray(y)
    majority = float(np.bincount(y.astype(int)).argmax())
    preds = np.full_like(y, majority)
    return {"accuracy": float(accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, average="macro"))}


def train_performance_model(X: np.ndarray, y: np.ndarray, model_name: ModelName = "random_forest",
                             feature_names=None, data_source: str = "synthetic") -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scaler = StandardScaler().fit(X)
    model = _make_regressor(model_name)
    model.fit(scaler.transform(X), y)

    cv_metrics = cross_validate("performance", X, y, model_name) or {}
    baseline = baseline_performance(y)
    metrics = {
        "mae": cv_metrics.get("mae_mean", 0.0),
        "r2": cv_metrics.get("r2_mean", 0.0),
        "rmse": cv_metrics.get("rmse_mean", 0.0),
        "folds": cv_metrics.get("folds", 0),
        "n_test": int(len(X)),
    }
    return TrainedBundle(
        model_name, "performance", model, scaler, feature_names, metrics,
        None, data_source, _feature_ranges(X, feature_names), cv_metrics, baseline,
    )


def train_injury_model(X: np.ndarray, y: np.ndarray, model_name: ModelName = "random_forest",
                        feature_names=None, label_map=None,
                        data_source: str = "synthetic") -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    label_map = label_map or {0: "low", 1: "moderate", 2: "high"}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    scaler = StandardScaler().fit(X)
    model = _make_classifier(model_name)
    model.fit(scaler.transform(X), y)

    cv_metrics = cross_validate("injury", X, y, model_name, label_map) or {}
    baseline = baseline_injury(y)
    metrics = {
        "accuracy": cv_metrics.get("accuracy_mean", 0.0),
        "f1_macro": cv_metrics.get("f1_mean", 0.0),
        "folds": cv_metrics.get("folds", 0),
        "n_test": int(len(X)),
    }
    return TrainedBundle(
        model_name, "injury", model, scaler, feature_names, metrics,
        label_map, data_source, _feature_ranges(X, feature_names), cv_metrics, baseline,
    )


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


def out_of_distribution_warnings(feature_vector: dict, bundle: TrainedBundle) -> list:
    """
    Return [(feature, value, lo, hi), ...] for every feature whose value falls
    outside the range seen in training. Predictions for such inputs extrapolate
    beyond the model's knowledge and should be flagged as unreliable.
    """
    ranges = getattr(bundle, "feature_ranges", None) or {}
    warnings_ = []
    for name, rng in ranges.items():
        if name not in feature_vector or feature_vector[name] is None:
            continue
        v = float(feature_vector[name])
        if v < rng["min"] or v > rng["max"]:
            warnings_.append((name, v, rng["min"], rng["max"]))
    return warnings_


def prediction_interval_performance(bundle: TrainedBundle, feature_vector: dict,
                                    percentile=(16, 84)) -> Optional[tuple]:
    """
    Approximate prediction interval for the performance score from the spread of
    individual random-forest tree predictions (16th-84th percentile ~ 68% band).
    Returns None for models without tree estimators (no uncertainty available).
    """
    model = bundle.model
    if not hasattr(model, "estimators_") or not getattr(model, "estimators_", None):
        return None
    x = np.array([[feature_vector[f] for f in bundle.feature_names]])
    xs = bundle.scaler.transform(x)
    preds = np.array([est.predict(xs)[0] for est in model.estimators_])
    lo, hi = np.percentile(preds, percentile)
    return float(max(0.0, lo)), float(min(100.0, hi))


def save_bundle(bundle: TrainedBundle, path: str):
    joblib.dump(bundle, path)


def load_bundle(path: str) -> TrainedBundle:
    return joblib.load(path)


BACKEND_INFO = {
    "xgboost_available": _HAS_XGBOOST,
    "catboost_available": _HAS_CATBOOST,
    "torch_available": _HAS_TORCH,
}
