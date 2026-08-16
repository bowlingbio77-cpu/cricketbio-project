"""
Data loading + preprocessing for the Cricket Injury Dataset
(1,272 player-season rows; binary injury_status target).

Notes:
  * Every player appears exactly once, so evaluation uses ordinary stratified
    CV (grouped-by-player CV degenerates to the same thing here).
  * ``recovered`` is a perfect duplicate of ``injury_status`` and the
    injury-detail columns (injury_type, body_site, ...) only exist on injured
    rows, so they are excluded from the feature set (kept for explainability).
  * An ordinal severity target (0 none / 1 minor / 2 major) is derived from
    match_days_lost so models can predict injury severity, not just incidence.
"""
import os
import numpy as np
import pandas as pd

from . import config


def load_cricket_injury(path=None) -> pd.DataFrame:
    path = path or config.CRICKET_INJURY_DATA
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df[config.CRICKET_INJURY_TARGET] = df[config.CRICKET_INJURY_TARGET].astype(int)
    for c in config.CRICKET_INJURY_NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_severity_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ordinal severity (0/1/2) from match_days_lost."""
    days = config.CRICKET_INJURY_SEVERITY_DAYS["minor"]
    df[config.CRICKET_INJURY_SEVERITY_TARGET] = df["match_days_lost"].apply(
        lambda d: 0 if d == 0 else (1 if d < days else 2))
    return df


def preprocess_cricket_injury(df: pd.DataFrame = None):
    """
    Build the feature matrix. Returns (X, y, groups, feature_names, df) where
    groups is player_id (all unique here) and the frame has the derived
    severity target attached for multi-class training.
    """
    df = load_cricket_injury() if df is None else df
    df = add_severity_target(df)
    df = df.sort_values([config.CRICKET_INJURY_SEASON_COL]).copy()

    feature_names = list(config.CRICKET_INJURY_NUMERIC_FEATURES)
    parts = []
    for c in config.CRICKET_INJURY_CATEGORICAL_FEATURES:
        if c in df.columns:
            parts.append(pd.get_dummies(df[c].astype(str), prefix=c))

    X = df[feature_names].astype(float).to_numpy()
    if parts:
        X = np.hstack([X] + [p.astype(float).to_numpy() for p in parts])
        for p in parts:
            feature_names += list(p.columns)

    y = df[config.CRICKET_INJURY_TARGET].to_numpy(dtype=int)
    groups = df[config.CRICKET_INJURY_GROUP_COL].to_numpy()
    return X, y, groups, feature_names, df


def dataset_summary(df: pd.DataFrame = None) -> dict:
    df = load_cricket_injury() if df is None else df
    return {
        "n_samples": int(len(df)),
        "n_players": int(df[config.CRICKET_INJURY_GROUP_COL].nunique()),
        "n_features": len(config.CRICKET_INJURY_NUMERIC_FEATURES)
                      + len(config.CRICKET_INJURY_CATEGORICAL_FEATURES),
        "class_dist": df[config.CRICKET_INJURY_TARGET].value_counts().sort_index().to_dict(),
        "seasons": sorted(df[config.CRICKET_INJURY_SEASON_COL].unique().tolist()),
    }
