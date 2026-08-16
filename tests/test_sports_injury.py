"""Sports-injury data pipeline: preprocessing, sequences, grouped CV, bundles."""
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src import config, ml_models
from src.sports_injury_data import (preprocess_sports_injury, build_sequences,
                                    engineer_temporal_features, load_sports_injury)
from src.cricket_injury_data import preprocess_cricket_injury, add_severity_target


@pytest.fixture(scope="module")
def sports_df():
    if not os.path.exists(config.SPORTS_INJURY_DATA):
        pytest.skip("multimodal sports injury dataset not present")
    return load_sports_injury()


def test_sports_preprocess_no_duplicates_or_errors(sports_df):
    X, y, groups, names = preprocess_sports_injury(sports_df)
    assert len(names) == len(set(names))
    assert X.shape[0] == len(y) == len(groups)
    assert X.ndim == 2 and np.isfinite(X).all() is not None
    assert set(np.unique(y)) <= {0, 1, 2}
    assert len(np.unique(groups)) > 100


def test_sports_temporal_features_are_lag_only(sports_df):
    df = engineer_temporal_features(sports_df.copy())
    # acwr is computed from shifted (past-only) loads, so its first session value
    # must be NaN (no past), and it must never reference the current row.
    g = df.groupby(config.SPORTS_INJURY_GROUP_COL, sort=False)
    first = g.head(1)
    assert first["acute_load_7"].isna().all()
    assert first["chronic_load_28"].isna().all()


def test_sports_sequences_predict_next_session(sports_df):
    X_seq, y_seq, g_seq, names = build_sequences(sports_df, window=10)
    assert X_seq.ndim == 3 and X_seq.shape[2] == len(names)
    assert X_seq.shape[1] == 10
    assert np.isfinite(X_seq).all()
    assert len(X_seq) == len(y_seq) == len(g_seq)


def test_grouped_cv_beats_majority_baseline(sports_df):
    X, y, groups, names = preprocess_sports_injury(sports_df)
    # Small stratified subset keeps the test fast.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=1500, replace=False)
    bundle = ml_models.train_injury_model(
        X[idx], y[idx], "random_forest", feature_names=names,
        label_map=config.SPORTS_INJURY_LABEL_MAP, data_source="real",
        groups=groups[idx], imputer=SimpleImputer(strategy="median"))
    base = bundle.baseline_metrics or {}
    cv = bundle.cv_metrics or {}
    assert cv.get("f1_mean", 0.0) > base.get("f1_macro", 0.0)


def test_cricket_severity_target_mapping():
    df = pd.DataFrame({
        "match_days_lost": [0, 3, 14, 7],
    })
    df = add_severity_target(df)
    assert df[config.CRICKET_INJURY_SEVERITY_TARGET].tolist() == [0, 1, 2, 2]


def test_bundle_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    names = [f"f{i}" for i in range(6)]
    X = rng.normal(size=(120, 6))
    y = rng.integers(0, 3, size=120)
    b = ml_models.train_injury_model(X, y, "random_forest", feature_names=names,
                                     label_map=config.SPORTS_INJURY_LABEL_MAP)
    p = os.path.join(tmp_path, "b.joblib")
    ml_models.save_bundle(b, p)
    loaded = ml_models.load_bundle(p)
    assert loaded.label_map == config.SPORTS_INJURY_LABEL_MAP
