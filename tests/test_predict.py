"""Saved-model loading and prediction (A4 checks: training, prediction, import side effects)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.predict import ModelLoadError, load_model, predict
from tests.conftest import VALID_HOUSE

ROOT = Path(__file__).resolve().parents[1]


def five_houses() -> list[dict]:
    houses = []
    for i, (area, loc) in enumerate([(850, "East"), (1450, "West"), (1850, "North"), (2600, "South"), (3400, "Central")]):
        houses.append({**VALID_HOUSE, "area_sqft": area, "locality": loc, "bedrooms": 1 + i,
                       "bathrooms": max(1, i), "year_built": 1990 + 6 * i, "has_garage": i % 2})
    return houses


def test_prediction_output_shape_and_values(loaded):
    pred = predict(loaded, five_houses())
    assert pred.shape == (5,)
    assert pred.dtype == float and np.all(np.isfinite(pred))
    assert np.all(pred > 0)
    assert pred[-1] > pred[0], "a large central house should cost more than a small one"


def test_prediction_is_deterministic_and_ignores_column_order(loaded):
    import pandas as pd
    df = pd.DataFrame(five_houses())
    a = predict(loaded, df)
    b = predict(loaded, df[list(reversed(df.columns))])
    np.testing.assert_array_equal(a, b)


def test_missing_feature_column_is_rejected(loaded):
    bad = [{k: v for k, v in VALID_HOUSE.items() if k != "locality"}]
    with pytest.raises(ValueError, match="missing required feature columns"):
        predict(loaded, bad)


def test_tampered_model_file_is_refused(model_path, tmp_path):
    src = Path(model_path)
    shutil.copy(src, tmp_path / "model.joblib")
    shutil.copy(src.with_name("model_metadata.json"), tmp_path / "model_metadata.json")
    with open(tmp_path / "model.joblib", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ModelLoadError, match="checksum"):
        load_model(tmp_path / "model.joblib")


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "nope.joblib")


def test_saved_model_loads_in_a_separate_process(model_path):
    """A4.4: a fresh Python process loads the file and predicts (no shared in-memory state)."""
    code = ("import json,sys; from src.predict import load_model, predict; "
            "m = load_model(sys.argv[1]); "
            f"print(json.dumps(predict(m, [{VALID_HOUSE!r}]).tolist()))")
    out = subprocess.run([sys.executable, "-c", code, str(model_path)], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert len(eval(out.stdout.strip())) == 1


def test_importing_modules_does_not_start_training(tmp_path):
    """A4.2: importing src.train / src.api must not train, write files or block."""
    env = {**os.environ, "MODEL_PATH": str(tmp_path / "absent.joblib")}
    start = time.time()
    out = subprocess.run([sys.executable, "-c", "import src.train, src.data, src.evaluate, src.predict, src.api"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr
    assert time.time() - start < 30
    assert not list(tmp_path.iterdir()), "import created files"
    assert "model saved" not in out.stdout + out.stderr
