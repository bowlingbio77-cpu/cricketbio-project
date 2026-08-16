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
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, f1_score,
    recall_score, precision_score, roc_auc_score, classification_report,
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

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
    import torch.nn as nn
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
                   label_map=None, n_folds: int = 5,
                   groups: Optional[np.ndarray] = None,
                   imputer: Optional[SimpleImputer] = None,
                   class_weight: Optional[dict] = None) -> Optional[dict]:
    """
    Honest generalization estimate via K-fold cross-validation.
    The scaler (and imputer, if supplied) are refit on each training fold
    (no leakage). When `groups` is supplied (e.g. athlete_id / player_id),
    StratifiedGroupKFold keeps every group in a single fold so correlated
    repeated measurements cannot leak across train/test boundaries.
    `class_weight` ({class: weight}) is applied as per-sample weights to counter
    class imbalance. For classifiers the result also includes macro
    precision/recall, OVR ROC-AUC and an out-of-fold per-class report.
    Returns None if the dataset is too small to stratify.
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
        if groups is not None:
            groups = np.asarray(groups)
            n_groups = len(np.unique(groups))
            if n_groups < n_folds:
                n_folds = max(2, n_groups)
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                            random_state=config.RANDOM_STATE)
        else:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=config.RANDOM_STATE)
    else:
        n_folds = _n_folds_for(n, None, n_folds)
        splitter = KFold(n_splits=n_folds, shuffle=True,
                         random_state=config.RANDOM_STATE)

    scores = {"mae": [], "r2": [], "rmse": [], "accuracy": [], "f1": [],
              "precision": [], "recall": [], "roc_auc": []}
    oof = np.full(n, -1) if task == "injury" else None

    for tr_idx, te_idx in splitter.split(X, y, groups=groups):
        imp = clone(imputer) if imputer is not None else None
        Xtr, Xte = X[tr_idx], X[te_idx]
        if imp is not None:
            Xtr = imp.fit_transform(Xtr)
            Xte = imp.transform(Xte)
        s = StandardScaler().fit(Xtr)
        Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        y_tr, y_te = y[tr_idx], y[te_idx]

        if task == "performance":
            m = _make_regressor(model_name).fit(Xtr, y_tr)
            preds = m.predict(Xte)
            scores["mae"].append(mean_absolute_error(y_te, preds))
            scores["r2"].append(r2_score(y_te, preds))
            scores["rmse"].append(float(np.sqrt(mean_squared_error(y_te, preds))))
        else:
            m = _make_classifier(model_name)
            fit_kwargs = {}
            # RandomForest already applies "balanced" class_weight in its
            # constructor -- skip the extra sample weights to avoid double-counting.
            if class_weight and getattr(m, "class_weight", None) != "balanced":
                sw = np.array([class_weight.get(int(v), 1.0) for v in y_tr])
                fit_kwargs["sample_weight"] = sw
            m.fit(Xtr, y_tr, **fit_kwargs)
            preds = np.asarray(m.predict(Xte)).ravel()
            oof[te_idx] = preds
            scores["accuracy"].append(accuracy_score(y_te, preds))
            scores["f1"].append(f1_score(y_te, preds, average="macro", zero_division=0))
            scores["precision"].append(precision_score(y_te, preds, average="macro", zero_division=0))
            scores["recall"].append(recall_score(y_te, preds, average="macro", zero_division=0))
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(Xte)
                try:
                    if len(np.unique(y)) == 2:
                        scores["roc_auc"].append(roc_auc_score(y_te, proba[:, 1]))
                    else:
                        scores["roc_auc"].append(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
                except ValueError:
                    pass

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
    prec, prec_std = _mean_std("precision")
    rec, rec_std = _mean_std("recall")
    auc, auc_std = _mean_std("roc_auc")
    report = {}
    if oof is not None:
        valid = oof >= 0
        if valid.any():
            report = classification_report(y[valid], oof[valid], output_dict=True,
                                           labels=sorted(np.unique(y[valid])), zero_division=0)
    return {"folds": n_folds, "n_samples": int(n),
            "accuracy_mean": acc, "accuracy_std": acc_std,
            "f1_mean": f1, "f1_std": f1_std,
            "precision_mean": prec, "recall_mean": rec, "roc_auc_mean": auc,
            "per_class_report": report}


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
                             feature_names=None, data_source: str = "synthetic",
                             groups=None, imputer=None) -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scaler = StandardScaler().fit(X)
    model = _make_regressor(model_name)
    model.fit(scaler.transform(X), y)

    cv_metrics = cross_validate("performance", X, y, model_name,
                                groups=groups, imputer=imputer) or {}
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
                        data_source: str = "synthetic",
                        groups=None, imputer=None,
                        class_weight: Optional[dict] = None) -> TrainedBundle:
    feature_names = feature_names or config.FEATURE_NAMES
    label_map = label_map or {0: "no injury", 1: "injury"}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if class_weight is None:
        counts = np.bincount(y)
        inv = 1.0 / np.maximum(counts, 1)
        class_weight = {int(i): float(inv[i] / inv.max())
                        for i in range(len(counts)) if counts[i] > 0}
    scaler = StandardScaler().fit(X)
    model = _make_classifier(model_name)
    fit_kwargs = {}
    if getattr(model, "class_weight", None) != "balanced":
        fit_kwargs["sample_weight"] = np.array([class_weight.get(int(v), 1.0) for v in y])
    model.fit(scaler.transform(X), y, **fit_kwargs)

    cv_metrics = cross_validate("injury", X, y, model_name, label_map,
                                groups=groups, imputer=imputer,
                                class_weight=class_weight) or {}
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
    raw = np.asarray(bundle.model.predict(xs)).ravel()
    if bundle.task == "performance":
        return float(raw[0])
    pred_idx = int(raw[0])
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


# ---------------------------------------------------------------------------
# Sequence models (CNN-LSTM / Transformer) for per-athlete session time-series.
# These operate on (n_samples, window, n_features) inputs instead of the flat
# feature vectors used by the tree/MLP models above, so the sample rows of one
# athlete (ordered by session_id) form a rolling context window that predicts
# the injury class of the FOLLOWING session. StratifiedGroupKFold (by athlete)
# is used for evaluation so sessions of one athlete never appear in both
# training and test folds.
# ---------------------------------------------------------------------------

class _SequenceClassifierBase:
    """Sklearn-style classifier over (n, window, features) tensors using torch."""

    def __init__(self, n_features, n_classes, window, epochs=30, lr=3e-4,
                 batch_size=256, dropout=0.2, random_state=config.RANDOM_STATE,
                 patience=7, val_frac=0.12):
        self.n_features = n_features
        self.n_classes = n_classes
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.patience = patience
        self.val_frac = val_frac
        torch.manual_seed(random_state)
        self._rng = torch.Generator().manual_seed(random_state)

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def _class_weights(self, y: np.ndarray) -> torch.Tensor:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (self.n_classes * counts)
        return torch.as_tensor(w, dtype=torch.float32)

    def fit(self, X, y, verbose=False):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        self.classes_ = np.unique(y)
        # Standardize each feature channel using ONLY the training rows (the
        # scaler is stored on the model and re-applied in predict_proba, so the
        # fold-wise CV in cross_validate_sequences stays leakage-free). The raw
        # per-session sensor scales differ by orders of magnitude (heart_rate ~
        # 1e2 vs step_count ~ 1e4) and cripple attention/conv init otherwise.
        self._scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
        X = self._scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X = np.asarray(X, dtype=np.float32)
        self.net = self._build_net()
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.val_frac, stratify=y, random_state=self.random_state)
        Xtr, ytr = torch.as_tensor(Xtr), torch.as_tensor(ytr)
        Xva, yva = torch.as_tensor(Xva), torch.as_tensor(yva)
        loss_fn = nn.CrossEntropyLoss(weight=self._class_weights(ytr.numpy()))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best_loss, best_state, no_improve = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=self._rng)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net(Xtr[b]), ytr[b])
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(Xva), yva).item())
            if val_loss < best_loss - 1e-4:
                best_loss, no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(f"    early stop at epoch {epoch + 1} (val loss {best_loss:.4f})")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        if getattr(self, "_scaler", None) is not None:
            X = self._scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self.net(Xt), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class _CNNLSTMNet(nn.Module):
    def __init__(self, n_features, hidden, n_classes, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):  # (B, W, F)
        x = x.transpose(1, 2)      # (B, F, W)
        x = self.conv(x)           # (B, 64, W)
        x = x.transpose(1, 2)      # (B, W, 64)
        out, _ = self.lstm(x)      # (B, W, 2*hidden)
        return self.head(out.mean(dim=1))  # mean-pool over time


class _TransformerNet(nn.Module):
    def __init__(self, n_features, window, d_model, nhead, n_layers, n_classes, dropout):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=dropout,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):  # (B, W, F)
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        x = self.enc(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class _CNNLSTMClassifier(_SequenceClassifierBase):
    def __init__(self, n_features, n_classes, window, hidden=64, **kwargs):
        self.hidden = hidden
        super().__init__(n_features, n_classes, window, **kwargs)

    def _build_net(self) -> nn.Module:
        return _CNNLSTMNet(self.n_features, self.hidden, self.n_classes, self.dropout)


class _TransformerClassifier(_SequenceClassifierBase):
    def __init__(self, n_features, n_classes, window, d_model=64, nhead=4,
                 n_layers=2, **kwargs):
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        super().__init__(n_features, n_classes, window, **kwargs)

    def _build_net(self) -> nn.Module:
        return _TransformerNet(self.n_features, self.window, self.d_model, self.nhead,
                               self.n_layers, self.n_classes, self.dropout)


def make_sequence_classifier(model_name: ModelName, n_features: int, n_classes: int,
                             window: int, epochs: int = 25,
                             random_state: int = config.RANDOM_STATE):
    """Return a torch sequence classifier, or raise if torch is unavailable."""
    if not _HAS_TORCH:
        raise RuntimeError(
            "Sequence models (cnn_lstm/transformer) require torch. "
            "Install it with `pip install torch` or use a tree model instead.")
    kwargs = dict(epochs=epochs, random_state=random_state)
    if model_name == "cnn_lstm":
        return _CNNLSTMClassifier(n_features, n_classes, window, hidden=32, **kwargs)
    if model_name == "transformer":
        return _TransformerClassifier(n_features, n_classes, window, **kwargs)
    raise ValueError(f"Unknown sequence model_name: {model_name}")


def _aggregate_classification_scores(scores: dict, oof: np.ndarray, y: np.ndarray,
                                     n_folds: int, label_map) -> dict:
    def _mean_std(key):
        vals = scores[key]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    acc, acc_std = _mean_std("accuracy")
    f1, f1_std = _mean_std("f1")
    prec, prec_std = _mean_std("precision")
    rec, rec_std = _mean_std("recall")
    auc, auc_std = _mean_std("roc_auc")
    report = {}
    if oof is not None:
        valid = oof >= 0
        if valid.any():
            report = classification_report(y[valid], oof[valid], output_dict=True,
                                           labels=sorted(np.unique(y[valid])), zero_division=0)
    return {"folds": n_folds, "n_samples": int(len(y)),
            "accuracy_mean": acc, "accuracy_std": acc_std,
            "f1_mean": f1, "f1_std": f1_std,
            "precision_mean": prec, "recall_mean": rec, "roc_auc_mean": auc,
            "per_class_report": report}


def cross_validate_sequences(X_seq: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             model_name: ModelName = "cnn_lstm", n_folds: int = 5,
                             n_classes: int = 3, epochs: int = 25,
                             random_state: int = config.RANDOM_STATE) -> dict:
    """
    StratifiedGroupKFold (by athlete/player) evaluation of a sequence classifier.
    Returns aggregated metrics plus an out-of-fold per-class report.
    """
    X_seq = np.asarray(X_seq, dtype=np.float32)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    n_folds = _n_folds_for(len(y), y, n_folds)
    n_folds = min(n_folds, max(2, len(np.unique(groups))))
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=random_state)
    scores = {"accuracy": [], "f1": [], "precision": [], "recall": [], "roc_auc": []}
    oof = np.full(len(y), -1)
    for tr_idx, te_idx in splitter.split(X_seq, y, groups):
        m = make_sequence_classifier(model_name, X_seq.shape[2], n_classes,
                                     X_seq.shape[1], epochs=epochs,
                                     random_state=random_state)
        m.fit(X_seq[tr_idx], y[tr_idx])
        preds = m.predict(X_seq[te_idx])
        oof[te_idx] = preds
        scores["accuracy"].append(accuracy_score(y[te_idx], preds))
        scores["f1"].append(f1_score(y[te_idx], preds, average="macro", zero_division=0))
        scores["precision"].append(precision_score(y[te_idx], preds, average="macro", zero_division=0))
        scores["recall"].append(recall_score(y[te_idx], preds, average="macro", zero_division=0))
        proba = m.predict_proba(X_seq[te_idx])
        if len(np.unique(y)) == 2:
            scores["roc_auc"].append(roc_auc_score(y[te_idx], proba[:, 1]))
        else:
            scores["roc_auc"].append(roc_auc_score(y[te_idx], proba, multi_class="ovr", average="macro"))
    return _aggregate_classification_scores(scores, oof, y, n_folds, None)


def train_sequence_model(X_seq: np.ndarray, y: np.ndarray, model_name: ModelName,
                         label_map: dict, groups=None, n_folds: int = 5,
                         epochs: int = 25, data_source: str = "real") -> TrainedBundle:
    """Train + honestly evaluate a sequence classifier and bundle it."""
    X_seq = np.asarray(X_seq, dtype=np.float32)
    y = np.asarray(y).astype(int)
    n_classes = len(np.unique(y)) if label_map is None else len(label_map)
    model = make_sequence_classifier(model_name, X_seq.shape[2], n_classes,
                                     X_seq.shape[1], epochs=epochs)
    model.fit(X_seq, y)
    cv_metrics = (cross_validate_sequences(X_seq, y, groups, model_name, n_folds=n_folds,
                                           n_classes=n_classes, epochs=epochs) or {})
    baseline = baseline_injury(y)
    metrics = {
        "accuracy": cv_metrics.get("accuracy_mean", 0.0),
        "f1_macro": cv_metrics.get("f1_mean", 0.0),
        "folds": cv_metrics.get("folds", 0),
        "n_test": int(len(X_seq)),
    }
    feature_names = [f"t{i}" for i in range(X_seq.shape[1])]
    return TrainedBundle(
        model_name, "injury", model, None, feature_names, metrics,
        label_map, data_source, None, cv_metrics, baseline,
    )


def predict_sequence(bundle: TrainedBundle, X_seq: np.ndarray) -> dict:
    """Predict injury risk for one (window, n_features) sequence."""
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim == 2:
        X_seq = X_seq[None, :, :]
    proba = bundle.model.predict_proba(X_seq)[0]
    pred_idx = int(np.argmax(proba))
    return {
        "risk_level": bundle.label_map[pred_idx],
        "risk_index": pred_idx,
        "probabilities": proba.tolist(),
    }


BACKEND_INFO = {
    "xgboost_available": _HAS_XGBOOST,
    "catboost_available": _HAS_CATBOOST,
    "torch_available": _HAS_TORCH,
    "sequence_models": _HAS_TORCH,
}
