"""Stage 6: load the saved pipeline safely and score new records.

    load_model(path)          -> LoadedModel (pipeline + metadata), after integrity checks
    predict(model, records)   -> numpy array of predicted prices (INR lakhs)

Safe loading: joblib/pickle files can execute code when loaded, so this module only
loads an artifact whose SHA-256 matches models/model_metadata.json (written at
training time) and refuses to load it under a different scikit-learn minor version.
Never load a model file from an untrusted source.

Batch inference (separate process, B6):
    python -m src.predict --input examples/batch_houses.csv --output reports/batch_predictions.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import schema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The model artifact is missing, altered or incompatible."""


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict
    path: Path

    @property
    def version(self) -> str:
        return f"{self.metadata['model_version']}+{self.metadata['model_sha256'][:12]}"


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _minor(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split(".")[:2])


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Explicit argument > MODEL_PATH environment variable > models/model.joblib."""
    p = Path(model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)
    return p if p.is_absolute() else ROOT / p


def load_model(model_path: str | Path | None = None) -> LoadedModel:
    path = resolve_model_path(model_path)
    meta_path = path.with_name("model_metadata.json")
    if not path.exists():
        raise ModelLoadError(f"model file not found: {path}")
    if not meta_path.exists():
        raise ModelLoadError(f"metadata file not found next to the model: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    actual = sha256_of(path)
    if actual != metadata.get("model_sha256"):
        raise ModelLoadError("model checksum does not match model_metadata.json; refusing to load")
    trained_with = metadata["library_versions"]["scikit-learn"]
    if _minor(trained_with) != _minor(sklearn.__version__):
        raise ModelLoadError(f"model was trained with scikit-learn {trained_with}, but "
                             f"{sklearn.__version__} is installed; install the pinned requirements")
    if trained_with != sklearn.__version__:
        logger.warning("scikit-learn patch version differs (trained %s, running %s)",
                       trained_with, sklearn.__version__)

    pipeline = joblib.load(path)
    return LoadedModel(pipeline=pipeline, metadata=metadata, path=path)


def to_frame(records) -> pd.DataFrame:
    """Accept a DataFrame, one dict or a list of dicts; return columns in the training order."""
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(list(records))
    missing = [c for c in schema.INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    if df.empty:
        raise ValueError("no records to predict")
    df = df[schema.INPUT_FEATURES].copy()
    df["has_garage"] = df["has_garage"].map(lambda v: float(v) if v is not None and not pd.isna(v) else np.nan)
    return df


def predict(model: LoadedModel, records) -> np.ndarray:
    """Predicted price (INR lakhs) for each record, as a float array of shape (n,)."""
    pred = np.asarray(model.pipeline.predict(to_frame(records)), dtype=float)
    if pred.ndim != 1 or not np.all(np.isfinite(pred)):
        raise RuntimeError("model returned invalid predictions")
    return pred


def score_csv(model: LoadedModel, input_path: Path) -> pd.DataFrame:
    """Batch scoring: fix formats, validate each row, predict the valid ones, explain the rest."""
    from src.data import load_data, normalize_formats, out_of_range_mask

    raw = load_data(input_path, required=schema.INPUT_FEATURES)
    df = normalize_formats(raw)
    problems = pd.Series("", index=df.index, dtype=object)
    for col in schema.INPUT_FEATURES:
        problems[df[col].isna()] += f"{col} missing or unparseable; "
    oor = out_of_range_mask(df)
    for col in oor.columns:
        problems[oor[col]] += f"{col} out of range {schema.VALID_RANGES[col]}; "
    rule = df["bathrooms"] > df["bedrooms"] + schema.MAX_EXTRA_BATHROOMS
    problems[rule] += "bathrooms exceed bedrooms + 1; "

    ok = problems == ""
    out = raw.copy()
    out["predicted_price_lakhs"] = np.nan
    out["status"] = np.where(ok, "ok", "rejected")
    out["error"] = problems.str.rstrip("; ")
    if ok.any():
        out.loc[ok, "predicted_price_lakhs"] = predict(model, df.loc[ok])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with the input feature columns")
    ap.add_argument("--output", help="write predictions CSV here (default: print)")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    model = load_model(args.model_path)
    out = score_csv(model, Path(args.input))
    print(f"model {model.version}: {int((out.status == 'ok').sum())} scored, "
          f"{int((out.status != 'ok').sum())} rejected")
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False, lineterminator="\n", float_format="%.6f")
        print(f"wrote {p}")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()
