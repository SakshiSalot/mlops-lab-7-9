"""Stage 1-2 of the pipeline: load, validate, clean and split the data.

Functions (A2):
    load_data(path)              read the raw CSV and check the expected columns exist
    normalize_formats(df)        stateless format fixes, shared with batch prediction
    preprocess_data(df)          training-time cleaning: duplicates, invalid target, out-of-range values
    split_data(df, cfg)          train/test split AFTER cleaning and de-duplication

Row-level cleaning (dropping rows) happens here, on training data only. Anything
the model also needs at prediction time (feature engineering, imputation, encoding)
lives in the saved scikit-learn Pipeline (src/features.py), so training and
inference cannot drift apart.

Run on its own to inspect the cleaned split:
    python -m src.data
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import schema
from src.config import load_config, resolve

REQUIRED_COLUMNS = [schema.ID_COLUMN, *schema.INPUT_FEATURES, schema.TARGET]
GARAGE_VALUES = {"yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
                 "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0}


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected contract."""


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(resolve(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- loading
def load_data(path: str | Path, required: list[str] | None = None) -> pd.DataFrame:
    """Read the CSV as text (formats are fixed later) and verify the columns."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `dvc pull` or `python -m src.make_dataset` first.")
    df = pd.read_csv(p, dtype=str, keep_default_na=True)
    missing = [c for c in (required or REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise DataValidationError(f"{p.name} is missing required columns: {missing}")
    if df.empty:
        raise DataValidationError(f"{p.name} has no rows")
    return df


# ----------------------------------------------------------------------------- cleaning
def _to_number(series: pd.Series) -> pd.Series:
    """'2,450' / '2450 sqft' / ' 1800 ' -> float; anything unparseable -> NaN."""
    s = series.astype("string").str.strip().str.lower()
    s = s.str.replace(",", "", regex=False).str.replace(r"\s*(sq\.?\s*ft|sqft)$", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype(float)


def normalize_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent formats without dropping rows (safe for training AND batch scoring)."""
    out = df.copy()
    for col in schema.NUMERIC_FEATURES:
        if col in out:
            out[col] = _to_number(out[col])
    if "has_garage" in out:
        g = out["has_garage"].astype("string").str.strip().str.lower()
        out["has_garage"] = g.map(GARAGE_VALUES).astype(float)
    if "locality" in out:
        loc = out["locality"].astype("string").str.strip().str.title()
        out["locality"] = loc.where(loc.isin(schema.LOCALITIES)).astype(object)
        out.loc[out["locality"].isna(), "locality"] = np.nan
    if schema.TARGET in out:
        out[schema.TARGET] = _to_number(out[schema.TARGET])
    if schema.ID_COLUMN in out:
        out[schema.ID_COLUMN] = pd.to_numeric(out[schema.ID_COLUMN], errors="coerce").astype("Int64")
    return out


def out_of_range_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where a present value lies outside schema.VALID_RANGES."""
    mask = pd.DataFrame(False, index=df.index, columns=list(schema.VALID_RANGES))
    for col, (lo, hi) in schema.VALID_RANGES.items():
        if col in df:
            mask[col] = df[col].notna() & ~df[col].between(lo, hi)
    return mask


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Training-time cleaning. Returns the clean frame and a report of every change.

    - exact duplicate rows and repeated house_id are removed (BEFORE the split, so the
      same house cannot appear in both train and test)
    - rows with a missing or non-positive target are removed (cannot train on them)
    - out-of-range feature values become NaN and are imputed later by the pipeline,
      using statistics learned from the training split only
    """
    report: dict = {"raw_rows": int(len(df))}
    raw_missing = df[schema.INPUT_FEATURES].isna().sum()

    out = normalize_formats(df)
    unparseable = (out[schema.INPUT_FEATURES].isna().sum() - raw_missing).clip(lower=0)
    report["values_unparseable_set_to_nan"] = {k: int(v) for k, v in unparseable.items() if v}

    before = len(out)
    out = out.drop_duplicates()
    report["exact_duplicates_removed"] = int(before - len(out))
    before = len(out)
    out = out.drop_duplicates(subset=[schema.ID_COLUMN], keep="first")
    report["repeated_ids_removed"] = int(before - len(out))

    bad_target = out[schema.TARGET].isna() | (out[schema.TARGET] <= 0)
    report["invalid_target_rows_removed"] = int(bad_target.sum())
    out = out.loc[~bad_target]

    oor = out_of_range_mask(out)
    report["out_of_range_set_to_nan"] = {k: int(v) for k, v in oor.sum().items() if v}
    for col in oor.columns:
        out.loc[oor[col], col] = np.nan

    rule = out["bathrooms"] > out["bedrooms"] + schema.MAX_EXTRA_BATHROOMS
    report["business_rule_violations_set_to_nan"] = int(rule.sum())
    out.loc[rule, "bathrooms"] = np.nan

    out = out[REQUIRED_COLUMNS].reset_index(drop=True)
    report["clean_rows"] = int(len(out))
    report["missing_values_left_for_imputation"] = {
        k: int(v) for k, v in out[schema.INPUT_FEATURES].isna().sum().items() if v}
    report["target_missing_after_cleaning"] = int(out[schema.TARGET].isna().sum())
    return out, report


# ----------------------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split with a fixed seed (training.test_size, training.random_state)."""
    t = cfg["training"]
    train, test = train_test_split(df, test_size=t["test_size"], random_state=t["random_state"])
    overlap = set(train[schema.ID_COLUMN]) & set(test[schema.ID_COLUMN])
    if overlap:
        raise DataValidationError(f"{len(overlap)} house_id values are in both train and test")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[schema.INPUT_FEATURES], df[schema.TARGET]


def prepare(cfg: dict, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """load -> preprocess -> split, optionally writing data/processed/ and reports/data_quality.json."""
    raw_path = cfg["data"]["raw_path"]
    raw = load_data(raw_path)
    clean, report = preprocess_data(raw)
    train, test = split_data(clean, cfg)
    report.update({"raw_path": raw_path, "raw_md5": file_md5(raw_path),
                   "train_rows": int(len(train)), "test_rows": int(len(test))})
    if write:
        out_dir = resolve(cfg["data"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_dir / "train.csv", index=False, lineterminator="\n")
        test.to_csv(out_dir / "test.csv", index=False, lineterminator="\n")
        rep = resolve(cfg["artifacts"]["reports_dir"]) / "data_quality.json"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return train, test, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    _, _, report = prepare(load_config(args.config, args.overrides))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
