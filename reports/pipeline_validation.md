# Pipeline validation (A4)

Run 2026-11-06 10:42:04 UTC with `python scripts/validate_pipeline.py` (Python 3.11.17).

| Check | Expected result | Evidence | Status |
|---|---|---|---|
| Data loading | Expected file and columns | `data/raw/houses_raw.csv` (md5 `71a9a5cf`): 1230 rows; all 9 required columns present | PASS |
| Preprocessing | No unexpected missing/invalid values | 30 duplicates and 10 invalid-target rows removed; 6 out-of-range values set to NaN; out-of-range left: 0; target missing: 0; NaN after the pipeline's imputation: 0 | PASS |
| Training | Model saved successfully | `python -m src.train` exit 0 in 5s; saved pipeline (features + preprocessing + gradient_boosting); retrained SHA-256 identical to models/model.joblib: True | PASS |
| Evaluation | Metrics on held-out data | 238 test rows: RMSE 8.918, MAE 5.536, R2 0.9641 (mean-baseline RMSE 47.08); `python -m src.evaluate` reproduces them: True | PASS |
| Prediction | Valid output for samples | new process loaded the model and scored 6 records (all finite, > 0); 2 invalid rows rejected with reasons | PASS |
| Import side effects | Importing a module does not train | importing all modules took 1.8s, created no files, printed nothing | PASS |
| No test-data leakage | Test data not used to fit preprocessing/model | imputer medians equal train-only medians: True; differ from train+test medians: True; house_id overlap: 0 | PASS |
| Tests | Tests pass | `pytest`: 29 passed in 3.82s | PASS |

## Predictions from a separate process (`python -m src.predict --input examples/batch_houses.csv`)

The batch file deliberately mixes formats (`2,450`, `980 sqft`, `Y`, `central`) and includes two invalid rows.

| house_id | area_sqft (raw) | locality (raw) | prediction (lakhs) | status |
|---|---|---|---|---|
| B1 | 1850 | North | 104.42 | ok |
| B2 | 2,450 | central | 151.56 | ok |
| B3 | 980 sqft | West | 40.22 | ok |
| B4 | 3300 | SOUTH | 194.31 | ok |
| B5 | 700 | East | 28.09 | ok |
| B6 | 1400 | West | 68.88 | ok |
| B7 | 24500 | North | - | rejected: area_sqft out of range (300, 10000) |
| B8 | 1100 | East | - | rejected: bathrooms exceed bedrooms + 1 |

