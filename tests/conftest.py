"""Shared fixtures. Tests train their own small model in a temp folder, so they need
neither DVC data nor a pre-built models/model.joblib, and never overwrite them."""
from __future__ import annotations

import pandas as pd
import pytest

from src import schema
from src.config import load_config
from src.data import preprocess_data, split_data, xy
from src.evaluate import evaluate_model
from src.make_dataset import make_raw
from src.predict import load_model
from src.train import save_model, train_model

VALID_HOUSE = dict(schema.EXAMPLE_HOUSE)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw().astype(str).replace({"<NA>": None, "None": None, "nan": None})


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("artifacts")
    c = load_config(overrides=[
        "model.gradient_boosting.n_estimators=60",          # small model -> fast tests
        f"artifacts.model_path={(out / 'model.joblib').as_posix()}",
        f"artifacts.metadata_path={(out / 'model_metadata.json').as_posix()}",
        f"artifacts.reports_dir={(out / 'reports').as_posix()}",
    ])
    return c


@pytest.fixture(scope="session")
def split(raw_df, cfg):
    clean, report = preprocess_data(raw_df)
    train, test = split_data(clean, cfg)
    report.update({"raw_md5": "test", "train_rows": len(train), "test_rows": len(test)})
    return train, test, report


@pytest.fixture(scope="session")
def model_path(cfg, split):
    train, test, report = split
    X_train, y_train = xy(train)
    X_test, y_test = xy(test)
    pipeline = train_model(X_train, y_train, cfg)
    metrics = evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline, cfg, metrics, report, {"cv_folds": 0, "cv_rmse_mean": 0.0, "cv_rmse_std": 0.0})
    return cfg["artifacts"]["model_path"]


@pytest.fixture(scope="session")
def loaded(model_path):
    return load_model(model_path)
