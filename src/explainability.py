"""
Stage 8: Explainable AI (SHAP)

Wraps SHAP (`pip install shap`) for per-prediction and global feature attribution.
Falls back to scikit-learn permutation importance (always available) when SHAP
isn't installed, so the dashboard's "why" panel always has something to show.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from . import config

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

SHAP_AVAILABLE = _HAS_SHAP


def explain_prediction(bundle, feature_vector: dict, background_X: np.ndarray = None):
    """
    Returns {feature_name: contribution} for a single prediction.
    Positive contribution = pushes prediction up (higher performance score /
    higher injury-risk class).
    """
    feature_names = bundle.feature_names
    x = np.array([[feature_vector[f] for f in feature_names]])
    xs = bundle.scaler.transform(x)

    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(bundle.model, background_X) if background_X is not None \
                else shap.Explainer(bundle.model)
            sv = explainer(xs)
            values = np.array(sv.values).reshape(-1)
            # classification models can return per-class arrays; take the predicted class / first output
            if values.ndim > 1 or len(values) != len(feature_names):
                values = np.array(sv.values)[0].mean(axis=-1) if np.array(sv.values).ndim > 2 \
                    else np.array(sv.values)[0]
            return {name: float(v) for name, v in zip(feature_names, values.flatten()[:len(feature_names)])}
        except Exception:
            pass  # fall through to permutation importance below

    # --- Fallback: local sensitivity via finite-difference perturbation ---
    base_pred = _scalar_predict(bundle, xs)
    contributions = {}
    eps = 0.5  # perturb by 0.5 std (features are already standardized)
    for i, name in enumerate(feature_names):
        x_pert = xs.copy()
        x_pert[0, i] += eps
        pert_pred = _scalar_predict(bundle, x_pert)
        contributions[name] = float(pert_pred - base_pred)
    return contributions


def explain_global(bundle, X: np.ndarray, y: np.ndarray, n_repeats: int = 10):
    """
    Global feature importance across a validation set. Uses SHAP mean(|value|)
    if available, otherwise sklearn permutation importance.
    """
    feature_names = bundle.feature_names
    Xs = bundle.scaler.transform(X)

    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(bundle.model, Xs)
            sv = explainer(Xs)
            values = np.abs(np.array(sv.values))
            if values.ndim == 3:  # (n_samples, n_features, n_classes)
                values = values.mean(axis=-1)
            importance = values.mean(axis=0)
            return {name: float(v) for name, v in zip(feature_names, importance)}
        except Exception:
            pass

    result = permutation_importance(
        bundle.model, Xs, y, n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, scoring=None,
    )
    return {name: float(v) for name, v in zip(feature_names, result.importances_mean)}


def _scalar_predict(bundle, xs: np.ndarray) -> float:
    if bundle.task == "performance":
        return float(bundle.model.predict(xs)[0])
    if hasattr(bundle.model, "predict_proba"):
        return float(np.max(bundle.model.predict_proba(xs)[0]))
    return float(bundle.model.predict(xs)[0])
