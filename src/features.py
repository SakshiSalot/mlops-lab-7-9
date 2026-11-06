"""Stage 3: feature engineering + preprocessing, packaged inside the saved Pipeline.

Everything here is fitted on the TRAINING split only and saved together with the
estimator in one scikit-learn Pipeline, so the API and batch scoring apply exactly
the same transformations as training.

    raw columns -> HouseFeatureEngineer -> ColumnTransformer -> estimator
                   (house_age, rooms,      (impute, scale,
                    area_per_room)          one-hot encode)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import schema

ENGINEERED = ["house_age", "rooms_total", "area_per_room"]
NUMERIC_MODEL_INPUTS = ["area_sqft", "bedrooms", "bathrooms", "distance_to_center_km", *ENGINEERED]


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-wise features derived from the raw columns.

    house_age     = reference_year - year_built  (fixed reference year -> reproducible)
    rooms_total   = bedrooms + bathrooms
    area_per_room = area_sqft / rooms_total
    Missing inputs give missing features, which the next step imputes.
    """

    def __init__(self, reference_year: int = schema.REFERENCE_YEAR):
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in schema.INPUT_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        self.feature_names_in_ = np.array(schema.INPUT_FEATURES, dtype=object)
        self.n_features_in_ = len(schema.INPUT_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HouseFeatureEngineer expects a pandas DataFrame with named columns")
        missing = [c for c in schema.INPUT_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        out = X[schema.INPUT_FEATURES].copy()
        for col in schema.NUMERIC_FEATURES + schema.BINARY_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        out["house_age"] = self.reference_year - out["year_built"]
        out["rooms_total"] = out["bedrooms"] + out["bathrooms"]
        out["area_per_room"] = out["area_sqft"] / out["rooms_total"]
        return out.drop(columns=["year_built"])

    def get_feature_names_out(self, input_features=None):
        cols = [c for c in schema.INPUT_FEATURES if c != "year_built"] + ENGINEERED
        return np.array(cols, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_MODEL_INPUTS),
         ("bin", binary, schema.BINARY_FEATURES),
         ("cat", categorical, schema.CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("features", HouseFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
