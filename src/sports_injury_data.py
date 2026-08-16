"""
Data loading + preprocessing for the Kaggle Multimodal Sports Injury Dataset
(15,420 per-session rows from 156 athletes, 3-class injury-risk target).

Key correctness choices:
  * Repeated sessions of one athlete are strongly correlated, so all evaluation
    is grouped by ``athlete_id`` (see ml_models.cross_validate / StratifiedGroupKFold)
    to prevent athletes leaking across train/test folds.
  * Missing values are imputed fold-by-fold (SimpleImputer refit on training
    folds only) or, for sequence construction, per-athlete using that athlete's
    own median -- never the whole dataset median -- so no global leakage.
  * Temporal/rolling features use ONLY past sessions (shift(1)) so a row's
    features never incorporate its own or future labels/sensors.
  * Sequence models get a rolling (window x n_features) context of the previous
    ``window`` sessions and predict the injury class of the FOLLOWING session.
"""
import os
import numpy as np
import pandas as pd

from . import config


def load_sports_injury(path=None) -> pd.DataFrame:
    """Load the CSV with consistent dtypes. Returns the raw frame (unsorted)."""
    path = path or config.SPORTS_INJURY_DATA
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from "
            "https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-dataset "
            "and save it as data/multimodal_sports_injury_dataset.csv")
    df = pd.read_csv(path)
    for c in config.SPORTS_INJURY_NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[config.SPORTS_INJURY_GROUP_COL] = df[config.SPORTS_INJURY_GROUP_COL].astype(int)
    df[config.SPORTS_INJURY_TIME_COL] = pd.to_numeric(
        df[config.SPORTS_INJURY_TIME_COL], errors="coerce")
    df[config.SPORTS_INJURY_TARGET] = pd.to_numeric(
        df[config.SPORTS_INJURY_TARGET], errors="coerce").astype(int)
    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag/rolling features computed strictly from an athlete's PAST sessions
    (sorted by session_id). Nothing here uses the current or future rows.
    """
    df = df.sort_values(
        [config.SPORTS_INJURY_GROUP_COL, config.SPORTS_INJURY_TIME_COL]).copy()
    g = df.groupby(config.SPORTS_INJURY_GROUP_COL, sort=False)

    df["acute_load_7"] = g["training_load"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    df["chronic_load_28"] = g["training_load"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).sum())
    df["acwr"] = df["acute_load_7"] / df["chronic_load_28"].replace(0, np.nan)
    df["prev_fatigue_index"] = g["fatigue_index"].shift(1)
    df["fatigue_delta"] = df["fatigue_index"] - df["prev_fatigue_index"]
    df["recovery_trend_3"] = g["recovery_score"].transform(
        lambda s: s.shift(1).rolling(config.SPORTS_INJURY_ROLLING_WINDOW, min_periods=1).mean())
    df["prev_session_high_risk"] = g[config.SPORTS_INJURY_TARGET].shift(1).fillna(0)
    df["cumulative_load"] = g["training_load"].cumsum().shift(1)
    return df


def categorical_columns(df: pd.DataFrame) -> list:
    """One-hot encode the categorical columns; returns the new feature names."""
    parts = []
    for c in config.SPORTS_INJURY_CATEGORICAL_FEATURES:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype(str), prefix=c)
            parts.append(d)
    return parts


def preprocess_sports_injury(df: pd.DataFrame = None, include_temporal: bool = True):
    """
    Build the flat feature matrix for tree/MLP models.

    Returns (X, y, groups, feature_names) with rows grouped by athlete and
    sorted by session (temporal features need the ordering).
    """
    df = load_sports_injury() if df is None else df
    df = engineer_temporal_features(df)
    df = df.dropna(subset=[config.SPORTS_INJURY_TARGET])

    feature_names = list(config.SPORTS_INJURY_NUMERIC_FEATURES)
    cat_parts = categorical_columns(df)
    if include_temporal:
        feature_names += [
            "acute_load_7", "chronic_load_28", "acwr", "prev_fatigue_index",
            "fatigue_delta", "recovery_trend_3", "prev_session_high_risk",
            "cumulative_load",
        ]

    X = df[feature_names].astype(float).to_numpy()
    if cat_parts:
        X = np.hstack([X] + [p.astype(float).to_numpy() for p in cat_parts])
        for p in cat_parts:
            feature_names += list(p.columns)

    y = df[config.SPORTS_INJURY_TARGET].to_numpy(dtype=int)
    groups = df[config.SPORTS_INJURY_GROUP_COL].to_numpy(dtype=int)
    return X, y, groups, feature_names


def impute_per_athlete(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Fill NaNs using each athlete's OWN column median (no global leakage)."""
    df = df.copy()
    g = df.groupby(config.SPORTS_INJURY_GROUP_COL, sort=False)
    for c in feature_names:
        if df[c].isna().any():
            df[c] = g[c].transform(lambda s: s.fillna(s.median()))
    return df


def build_sequences(df: pd.DataFrame = None, window: int = None):
    """
    Build per-athlete rolling sequences for the sequence models.

    Each sample is the previous ``window`` sessions (features) and the target
    is the injury class of the FOLLOWING session. Returns
    (X_seq, y_seq, groups_seq, feature_names).
    """
    window = window or config.SPORTS_INJURY_SEQUENCE_WINDOW
    df = load_sports_injury() if df is None else df
    df = engineer_temporal_features(df)
    df = df.dropna(subset=[config.SPORTS_INJURY_TARGET])

    feature_names = list(config.SPORTS_INJURY_NUMERIC_FEATURES)
    cat_parts = categorical_columns(df)
    feature_names += [
        "acute_load_7", "chronic_load_28", "acwr", "prev_fatigue_index",
        "fatigue_delta", "recovery_trend_3", "prev_session_high_risk",
        "cumulative_load",
    ]
    df = impute_per_athlete(df, feature_names)
    if cat_parts:
        for i, p in enumerate(cat_parts):
            df[p.columns] = p.astype(float)
            feature_names += list(p.columns)

    X_seq, y_seq, groups_seq = [], [], []
    for aid, sub in df.groupby(config.SPORTS_INJURY_GROUP_COL, sort=False):
        sub = sub.sort_values(config.SPORTS_INJURY_TIME_COL)
        vals = sub[feature_names].to_numpy(dtype=float)
        labels = sub[config.SPORTS_INJURY_TARGET].to_numpy(dtype=int)
        n = len(vals)
        if n <= window:
            continue
        for t in range(window, n):
            X_seq.append(vals[t - window:t])
            y_seq.append(labels[t])
            groups_seq.append(aid)
    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=int),
            np.asarray(groups_seq, dtype=int),
            feature_names)


def dataset_summary(df: pd.DataFrame = None) -> dict:
    """Quick EDA summary printed by the training script."""
    df = load_sports_injury() if df is None else df
    return {
        "n_samples": int(len(df)),
        "n_athletes": int(df[config.SPORTS_INJURY_GROUP_COL].nunique()),
        "n_features": len(config.SPORTS_INJURY_NUMERIC_FEATURES)
                      + len(config.SPORTS_INJURY_CATEGORICAL_FEATURES),
        "missing_frac": float(df.isna().mean().mean()),
        "class_dist": df[config.SPORTS_INJURY_TARGET].value_counts().sort_index().to_dict(),
    }
