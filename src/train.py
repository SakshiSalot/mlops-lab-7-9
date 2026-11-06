"""Stage 4: read the config, train, save the full inference pipeline, evaluate, log to MLflow.

    python -m src.train                                    # use configs/config.yaml
    python -m src.train --set model.type=random_forest     # override any config key
    python -m src.train --no-mlflow                        # skip experiment tracking

Steps: load + clean + split (src.data) -> build Pipeline (src.features) -> 5-fold CV on
the train split -> fit on train -> save models/model.joblib + model_metadata.json ->
evaluate on the held-out test split (src.evaluate) -> reports/ -> MLflow run.

Importing this module does nothing; work starts only in main().
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from src import schema
from src.config import ROOT, load_config, resolve
from src.data import prepare, xy
from src.evaluate import evaluate_model, write_reports
from src.features import build_pipeline
from src.predict import sha256_of

logger = logging.getLogger("train")
ESTIMATORS = {"gradient_boosting": GradientBoostingRegressor, "random_forest": RandomForestRegressor,
              "ridge": Ridge}


def build_estimator(cfg: dict):
    kind = cfg["model"]["type"]
    if kind not in ESTIMATORS:
        raise ValueError(f"model.type must be one of {sorted(ESTIMATORS)}, got {kind!r}")
    params = dict(cfg["model"].get(kind) or {})
    if kind != "ridge":
        params["random_state"] = cfg["training"]["random_state"]
    return ESTIMATORS[kind](**params)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict):
    """Fit preprocessing + estimator on the training split only."""
    pipeline = build_pipeline(build_estimator(cfg))
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate(X_train, y_train, cfg: dict) -> dict:
    folds = cfg["training"]["cv_folds"]
    cv = KFold(folds, shuffle=True, random_state=cfg["training"]["random_state"])
    scores = -cross_val_score(build_pipeline(build_estimator(cfg)), X_train, y_train, cv=cv,
                              scoring="neg_root_mean_squared_error")
    return {"cv_folds": folds, "cv_rmse_mean": float(scores.mean()), "cv_rmse_std": float(scores.std())}


def save_model(pipeline, cfg: dict, metrics: dict, data_report: dict, cv: dict) -> dict:
    """Write the pipeline and a metadata file the loader uses for integrity checks."""
    model_path = resolve(cfg["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    kind = cfg["model"]["type"]
    metadata = {
        "model_name": "house-price-regressor",
        "model_version": str(cfg["project"]["model_version"]),
        "model_file": model_path.name,
        "model_sha256": sha256_of(model_path),
        "model_type": kind,
        "params": cfg["model"].get(kind) or {},
        "target": schema.TARGET,
        "unit": "INR lakhs",
        "input_features": schema.INPUT_FEATURES,
        "valid_ranges": {k: list(v) for k, v in schema.VALID_RANGES.items()},
        "localities": list(schema.LOCALITIES),
        "reference_year": schema.REFERENCE_YEAR,
        "split": {"test_size": cfg["training"]["test_size"], "random_state": cfg["training"]["random_state"]},
        "train_rows": data_report["train_rows"],
        "test_rows": data_report["test_rows"],
        "raw_data_md5": data_report["raw_md5"],
        "metrics": metrics,
        "cv": cv,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    meta_path = resolve(cfg["artifacts"]["metadata_path"])
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_to_mlflow(cfg: dict, pipeline, metadata: dict, cv: dict, X_train: pd.DataFrame) -> str | None:
    """Log the run if MLflow is installed and enabled. Imported lazily: the API image does not need it."""
    try:
        import mlflow
        import mlflow.sklearn
        import skops.io as sio
        from mlflow.models import infer_signature
    except ImportError:
        logger.warning("mlflow not installed; skipping experiment tracking")
        return None

    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).is_absolute():
        uri = "sqlite:///" + resolve(uri[len("sqlite:///"):]).as_posix()
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        location = resolve("mlartifacts").as_uri() if uri.startswith("sqlite") else None
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)

    rdir = resolve(cfg["artifacts"]["reports_dir"])
    with mlflow.start_run(run_name=f"{metadata['model_type']}-v{metadata['model_version']}") as run:
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "HEAD"),
            "git_code_dirty": str(bool(_git("status", "--porcelain", "--", "src", "configs"))).lower(),
            "raw_data_md5": metadata["raw_data_md5"],
            "model_sha256": metadata["model_sha256"],
            "lab": "7-9",
        })
        mlflow.log_params({"model_type": metadata["model_type"], **metadata["params"],
                           "test_size": metadata["split"]["test_size"],
                           "random_state": metadata["split"]["random_state"],
                           "train_rows": metadata["train_rows"], "test_rows": metadata["test_rows"]})
        mlflow.log_metrics({k: v for k, v in {**metadata["metrics"], **cv}.items() if isinstance(v, (int, float))})
        for f in ["metrics.json", "evaluation.md", "data_quality.json", "figures/residuals.png"]:
            if (rdir / f).exists():
                mlflow.log_artifact(str(rdir / f), artifact_path="reports")
        mlflow.log_artifact(str(resolve(cfg["artifacts"]["model_path"])), artifact_path="joblib_model")
        mlflow.log_artifact(str(resolve(cfg["artifacts"]["metadata_path"])), artifact_path="joblib_model")
        # Also log as an MLflow model (skops format). We trained this pipeline in this
        # process, so its types are trusted; src/ is shipped for the custom transformer.
        trusted = sio.get_untrusted_types(data=sio.dumps(pipeline))
        sample = X_train.head(3)
        mlflow.sklearn.log_model(pipeline, name="model", signature=infer_signature(sample, pipeline.predict(sample)),
                                 input_example=sample, code_paths=[str(ROOT / "src")],
                                 skops_trusted_types=trusted)
        return run.info.run_id


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config YAML (default configs/config.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set model.type=random_forest")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    train, test, data_report = prepare(cfg)
    X_train, y_train = xy(train)
    X_test, y_test = xy(test)
    logger.info("data: %s raw rows -> %s clean (train %s / test %s)", data_report["raw_rows"],
                data_report["clean_rows"], len(train), len(test))

    cv = cross_validate(X_train, y_train, cfg)
    pipeline = train_model(X_train, y_train, cfg)
    metrics = evaluate_model(pipeline, X_test, y_test)
    metadata = save_model(pipeline, cfg, metrics, data_report, cv)
    write_reports(metrics, cv, metadata, data_report, cfg["artifacts"]["reports_dir"], pipeline, X_test, y_test)
    logger.info("model saved: %s (sha256 %s...)", cfg["artifacts"]["model_path"], metadata["model_sha256"][:12])
    logger.info("test RMSE %.3f | MAE %.3f | R2 %.4f | CV RMSE %.3f +/- %.3f", metrics["test_rmse"],
                metrics["test_mae"], metrics["test_r2"], cv["cv_rmse_mean"], cv["cv_rmse_std"])

    if cfg["mlflow"]["enabled"] and not args.no_mlflow:
        run_id = log_to_mlflow(cfg, pipeline, metadata, cv, X_train)
        if run_id:
            logger.info("MLflow run %s logged to %s", run_id,
                        os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"])


if __name__ == "__main__":
    main()
