"""A4: validate the modular pipeline and fill the evidence table (reports/pipeline_validation.md).

    python scripts/validate_pipeline.py

Runs every stage from the terminal in separate Python processes, retrains from scratch
into a temporary folder (to prove the saved model is reproducible), re-evaluates,
scores records, checks for leakage and import side effects, and runs the test suite.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPORTS, ROOT, md_table, now

from src import schema
from src.config import load_config
from src.data import REQUIRED_COLUMNS, file_md5, load_data, out_of_range_mask, preprocess_data
from src.predict import load_model, sha256_of


def py(*args: str, env=None) -> subprocess.CompletedProcess:
    print("$ python", " ".join(args), flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def main() -> None:
    cfg = load_config()
    rows, failed = [], False

    def check(name, expected, evidence, ok):
        nonlocal failed
        failed |= not ok
        rows.append({"Check": name, "Expected result": expected, "Evidence": evidence, "Status": "PASS" if ok else "FAIL"})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}")

    # 1. data loading
    raw_path = cfg["data"]["raw_path"]
    raw = load_data(raw_path)
    check("Data loading", "Expected file and columns",
          f"`{raw_path}` (md5 `{file_md5(raw_path)[:8]}`): {len(raw)} rows; all {len(REQUIRED_COLUMNS)} required columns present",
          all(c in raw.columns for c in REQUIRED_COLUMNS))

    # 2. preprocessing
    clean, rep = preprocess_data(raw)
    model = load_model()
    train = pd.read_csv(ROOT / cfg["data"]["processed_dir"] / "train.csv")
    test = pd.read_csv(ROOT / cfg["data"]["processed_dir"] / "test.csv")
    transformed = model.pipeline[:-1].transform(pd.concat([train, test])[schema.INPUT_FEATURES])
    nan_after = int(np.isnan(transformed).sum())
    oor_left = int(out_of_range_mask(clean).to_numpy().sum())
    check("Preprocessing", "No unexpected missing/invalid values",
          f"{rep['exact_duplicates_removed']} duplicates and {rep['invalid_target_rows_removed']} invalid-target rows removed; "
          f"{sum(rep['out_of_range_set_to_nan'].values())} out-of-range values set to NaN; out-of-range left: {oor_left}; "
          f"target missing: {rep['target_missing_after_cleaning']}; NaN after the pipeline's imputation: {nan_after}",
          oor_left == 0 and rep["target_missing_after_cleaning"] == 0 and nan_after == 0)

    # 3. training from the terminal, into a temp folder -> must reproduce the saved artifact
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp).as_posix()
        start = time.time()
        res = py("-m", "src.train", "--no-mlflow", "--set", f"artifacts.model_path={t}/model.joblib",
                 "--set", f"artifacts.metadata_path={t}/model_metadata.json", "--set", f"artifacts.reports_dir={t}/reports",
                 "--set", f"data.processed_dir={t}/processed")
        retrained = Path(tmp) / "model.joblib"
        same = retrained.exists() and sha256_of(retrained) == model.metadata["model_sha256"]
        check("Training", "Model saved successfully",
              f"`python -m src.train` exit {res.returncode} in {time.time() - start:.0f}s; saved pipeline "
              f"(features + preprocessing + {model.metadata['model_type']}); retrained SHA-256 identical to "
              f"models/model.joblib: {same}", res.returncode == 0 and same)

    # 4. evaluation on held-out data, in a separate process
    res = py("-m", "src.evaluate")
    m = model.metadata["metrics"]
    check("Evaluation", "Metrics on held-out data",
          f"{m['test_rows']} test rows: RMSE {m['test_rmse']:.3f}, MAE {m['test_mae']:.3f}, R2 {m['test_r2']:.4f} "
          f"(mean-baseline RMSE {m['baseline_rmse_predict_mean']:.2f}); `python -m src.evaluate` reproduces them: "
          f"{'True' in res.stdout}", res.returncode == 0 and "True" in res.stdout)

    # 5. prediction in a separate process (>= 5 records)
    out_csv = REPORTS / "batch_predictions.csv"
    res = py("-m", "src.predict", "--input", "examples/batch_houses.csv", "--output", str(out_csv))
    preds = pd.read_csv(out_csv)
    ok_preds = preds[preds.status == "ok"]
    check("Prediction", "Valid output for samples",
          f"new process loaded the model and scored {len(ok_preds)} records (all finite, > 0); "
          f"{int((preds.status != 'ok').sum())} invalid rows rejected with reasons",
          res.returncode == 0 and len(ok_preds) >= 5 and bool((ok_preds.predicted_price_lakhs > 0).all()))

    # 6. importing modules must not start training
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MODEL_PATH": str(Path(tmp) / "x.joblib")}
        start = time.time()
        res = py("-c", "import src.train, src.data, src.features, src.evaluate, src.predict, src.api", env=env)
        took = time.time() - start
        check("Import side effects", "Importing a module does not train",
              f"importing all modules took {took:.1f}s, created no files, printed nothing",
              res.returncode == 0 and not res.stdout.strip() and not list(Path(tmp).iterdir()))

    # 7. leakage: preprocessing statistics come from the train split only
    num = model.pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    cols = list(model.pipeline.named_steps["preprocess"].transformers_[0][2])
    fe = model.pipeline.named_steps["features"]
    train_med = fe.transform(train[schema.INPUT_FEATURES])[cols].median().to_numpy()
    all_med = fe.transform(pd.concat([train, test])[schema.INPUT_FEATURES])[cols].median().to_numpy()
    overlap = len(set(train[schema.ID_COLUMN]) & set(test[schema.ID_COLUMN]))
    check("No test-data leakage", "Test data not used to fit preprocessing/model",
          f"imputer medians equal train-only medians: {np.allclose(num.statistics_, train_med)}; "
          f"differ from train+test medians: {not np.allclose(num.statistics_, all_med)}; house_id overlap: {overlap}",
          np.allclose(num.statistics_, train_med) and overlap == 0)

    # 8. tests
    res = py("-m", "pytest", "-p", "no:cacheprovider")
    summary = (re.findall(r"\d+ passed[^\n]*", res.stdout) or [res.stdout.strip().splitlines()[-1]])[-1]
    check("Tests", "Tests pass", f"`pytest`: {summary}", res.returncode == 0)

    pred_rows = [{"house_id": r.house_id, "area_sqft (raw)": r.area_sqft, "locality (raw)": str(r.locality).strip(),
                  "prediction (lakhs)": f"{r.predicted_price_lakhs:.2f}" if r.status == "ok" else "-",
                  "status": r.status if r.status == "ok" else f"rejected: {r.error}"} for r in preds.itertuples()]
    text = "\n".join([
        "# Pipeline validation (A4)", "",
        f"Run {now()} with `python scripts/validate_pipeline.py` (Python {sys.version.split()[0]}).", "",
        md_table(rows, ["Check", "Expected result", "Evidence", "Status"]),
        "## Predictions from a separate process (`python -m src.predict --input examples/batch_houses.csv`)", "",
        "The batch file deliberately mixes formats (`2,450`, `980 sqft`, `Y`, `central`) and includes two invalid rows.", "",
        md_table(pred_rows, ["house_id", "area_sqft (raw)", "locality (raw)", "prediction (lakhs)", "status"]),
    ])
    (REPORTS / "pipeline_validation.md").write_text(text + "\n", encoding="utf-8")
    print("wrote reports/pipeline_validation.md")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
