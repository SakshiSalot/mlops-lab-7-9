"""Data loading, cleaning and splitting (A4 checks: data loading, preprocessing, leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import schema
from src.data import (REQUIRED_COLUMNS, DataValidationError, load_data, normalize_formats,
                      preprocess_data, split_data)


def test_load_data_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"house_id": [1], "area_sqft": [1000]}).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_data(p)


def test_load_data_reads_expected_columns(tmp_path, raw_df):
    p = tmp_path / "raw.csv"
    raw_df.to_csv(p, index=False)
    df = load_data(p)
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert len(df) == len(raw_df)


def test_normalize_formats_fixes_inconsistent_values():
    raw = pd.DataFrame({
        "area_sqft": ["2,450", "1800 sqft", " 950 ", "large"],
        "has_garage": ["Yes", "N", "1", "maybe"],
        "locality": ["  north ", "CENTRAL", "West", "Uptown"],
    })
    out = normalize_formats(raw)
    assert out["area_sqft"].tolist()[:3] == [2450.0, 1800.0, 950.0]
    assert np.isnan(out["area_sqft"].iloc[3])
    assert out["has_garage"].tolist()[:3] == [1.0, 0.0, 1.0] and np.isnan(out["has_garage"].iloc[3])
    assert out["locality"].tolist()[:3] == ["North", "Central", "West"] and pd.isna(out["locality"].iloc[3])


def test_preprocess_removes_duplicates_and_invalid_values(raw_df):
    clean, report = preprocess_data(raw_df)
    assert report["exact_duplicates_removed"] == 30
    assert report["invalid_target_rows_removed"] == 10
    assert clean[schema.ID_COLUMN].is_unique
    assert clean[schema.TARGET].notna().all() and (clean[schema.TARGET] > 0).all()
    for col, (lo, hi) in schema.VALID_RANGES.items():
        present = clean[col].dropna()
        assert present.between(lo, hi).all(), f"{col} has out-of-range values after cleaning"
    assert set(clean["locality"].dropna()) <= set(schema.LOCALITIES)


def test_split_has_no_overlap_and_expected_size(split, cfg):
    train, test, _ = split
    assert not set(train[schema.ID_COLUMN]) & set(test[schema.ID_COLUMN])
    expected = round((len(train) + len(test)) * cfg["training"]["test_size"])
    assert abs(len(test) - expected) <= 1


def test_preprocessing_is_fitted_on_train_only(loaded, split):
    """Leakage check: imputer medians and scaler means come from the TRAIN split only."""
    train, test, _ = split
    pipe = loaded.pipeline
    fe = pipe.named_steps["features"]
    num = pipe.named_steps["preprocess"].named_transformers_["num"]
    cols = list(pipe.named_steps["preprocess"].transformers_[0][2])
    train_feats = fe.transform(train[schema.INPUT_FEATURES])[cols]
    all_feats = fe.transform(pd.concat([train, test])[schema.INPUT_FEATURES])[cols]
    np.testing.assert_allclose(num.named_steps["impute"].statistics_, train_feats.median().to_numpy())
    imputed = num.named_steps["impute"].transform(train_feats)
    np.testing.assert_allclose(num.named_steps["scale"].mean_, imputed.mean(axis=0))
    assert not np.allclose(num.named_steps["impute"].statistics_, all_feats.median().to_numpy()), \
        "medians equal full-data medians: test rows may have leaked into fitting"
