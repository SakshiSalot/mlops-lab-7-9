# MLOps Lab 7-9: ML Pipelines, FastAPI and Docker

House-price prediction service continued from Sessions 3-6: the model is refactored into a modular,
config-driven pipeline (Part A), served through a FastAPI REST API (Part B), and packaged with Docker
and Docker Compose next to an MLflow tracking server (Part C), with an end-to-end reproducibility check
(Part D). Data is synthetic (no personal data) and versioned with DVC.

```
data/raw/houses_raw.csv (DVC)
   -> src/data.py       load, validate, clean (duplicates, formats, invalid values), split
   -> src/features.py   feature engineering + impute/scale/one-hot   \
   -> src/train.py      fit on train split, CV, save, log to MLflow    }  one sklearn Pipeline
   -> src/evaluate.py   held-out metrics + reproducible report        /   -> models/model.joblib
   -> src/predict.py    verified loading, predict(), batch CSV scoring
   -> src/api.py        FastAPI: GET /health, GET /model-info, POST /predict
   -> Dockerfile        python:3.11-slim image with code + model
   -> docker-compose.yml   api + mlflow (shared network, persistent volume)
```

| Result (reference run) | Value |
|---|---|
| Test metrics (238 held-out houses) | RMSE 8.918, MAE 5.536, R2 0.964, MAPE 6.5% (baseline RMSE 47.08) |
| Tests | 29 passed (data, leakage, saved-model loading, prediction, API) |
| A4 pipeline checks | 8/8 pass; retraining reproduces the same model SHA-256 |
| B5 API checks | 8/8 behave as expected (200 / 422 / health) |
| Local vs container predictions | max absolute difference 0.0 on 6 inputs (tolerance 1e-6) |
| Compose | API reaches MLflow at `http://mlflow:5000`; runs persist across `down`/`up` |

Detailed evidence: [`reports/`](reports/). Written answers (A1, B1, B6, C1, challenges, reflection):
[`docs/theory.md`](docs/theory.md).

---

## Repository layout

```
mlops-lab-7-9/
├── configs/config.yaml          every tunable value (paths, split, model, MLflow)
├── data/raw/houses_raw.csv.dvc  DVC pointer to the raw export (the CSV itself is in DVC)
├── data/processed/              cleaned train/test split (DVC output)
├── models/                      model.joblib + model_metadata.json (DVC outputs)
├── src/
│   ├── __init__.py
│   ├── schema.py                data contract: columns, ranges, localities, business rule
│   ├── config.py                config loading + --set overrides
│   ├── make_dataset.py          generates the raw export (deterministic)
│   ├── data.py                  load_data(), normalize_formats(), preprocess_data(), split_data()
│   ├── features.py              HouseFeatureEngineer, build_preprocessor(), build_pipeline()
│   ├── train.py                 train_model(), save_model(), log_to_mlflow(), main()
│   ├── evaluate.py              evaluate_model(), write_reports()
│   ├── predict.py               load_model(), predict(), batch scoring CLI
│   └── api.py                   FastAPI app
├── tests/                       pytest: test_data.py, test_predict.py, test_api.py
├── scripts/                     evidence scripts (validate_pipeline, api_check, docker_check, compose_check, compare_predictions)
├── examples/                    request JSON files, batch CSV, Postman collection
├── reports/                     metrics, evaluation report, validation tables, evidence/
├── docs/                        theory.md (written answers), screenshots/
├── Dockerfile, .dockerignore, docker-compose.yml
├── dvc.yaml, dvc.lock           pipeline stage `train` and its exact hashes
├── requirements.txt             runtime (API image)
├── requirements-dev.txt         + training, MLflow, DVC, tests
└── README.md, .gitignore
```

---

## 1. Setup (Windows PowerShell; macOS/Linux in comments)

Requirements: Python 3.11, 3.12 or 3.13, Git, Docker Desktop. Keep the project in a path **without
spaces** (e.g. `C:\mlops\`), not under OneDrive.

```powershell
cd C:\mlops\mlops-lab-7-9
python -m venv .venv
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If PowerShell blocks `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Get the data and the model

The DVC remote is a local folder **next to** the repo: `C:\mlops\dvc-storage`.

```powershell
dvc pull          # restores data/raw, data/processed and models/model.joblib
dvc status        # "Data and pipelines are up to date."
```

No `dvc-storage` folder? Regenerate everything instead (outputs are deterministic):

```powershell
python -m src.make_dataset        # data/raw/houses_raw.csv (md5 71a9a5cf...)
python -m src.train               # data/processed, models/, reports/
```

---

## 2. Part A: modular pipeline

```powershell
python -m src.train                                   # clean -> split -> CV -> fit -> save -> evaluate -> MLflow
python -m src.train --set model.type=random_forest    # override any config key (A3)
python -m src.train --no-mlflow                       # skip experiment tracking
python -m src.evaluate                                # re-score the saved model; must match training-time metrics
python -m src.predict --input examples/batch_houses.csv              # batch scoring in a separate process
python -m pytest                                      # 29 tests
python scripts/validate_pipeline.py                   # fills the A4 table -> reports/pipeline_validation.md
dvc repro                                             # reruns `train` only if data, code or config changed
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Key design points: preprocessing and the estimator are saved as **one** scikit-learn Pipeline, so
the API applies exactly the training transformations. Duplicates are removed before the split; all
fitting happens on the train split only (checked by a test). `models/model_metadata.json` records the
model's SHA-256 and library versions; `load_model()` refuses a file whose checksum does not match.

---

## 3. Part B: FastAPI service

```powershell
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs (Swagger UI). Endpoints:

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| GET | `/health` | Model loaded? MLflow reachable (if configured)? | 200 ok, 503 model not loaded |
| GET | `/model-info` | Version, SHA-256, metrics, input schema | 200, 503 |
| POST | `/predict` | Price of one house (INR lakhs) | 200, 422 invalid input, 503, 500 |

### Request schema (`POST /predict`)

| Field | Type | Rule |
|---|---|---|
| `area_sqft` | number | 300 to 10000 |
| `bedrooms` | integer | 1 to 10 |
| `bathrooms` | integer | 1 to 10, and at most `bedrooms + 1` (business rule) |
| `year_built` | integer | 1900 to 2026 |
| `distance_to_center_km` | number | 0 to 100 |
| `has_garage` | boolean | `true` / `false` |
| `locality` | string | one of `Central`, `East`, `North`, `South`, `West` |

Types are strict (`"1850"` as a string is rejected) and unknown fields are rejected.

Valid request (`examples/request_valid.json`) and response:

```json
{"area_sqft": 1850, "bedrooms": 3, "bathrooms": 2, "year_built": 2012,
 "distance_to_center_km": 6.5, "has_garage": true, "locality": "North"}
```
```json
{"prediction": 104.41561579290465, "unit": "INR lakhs",
 "model_version": "1.0.0+b09c57dd3b34", "request_id": "1398b9f625e1"}
```

Invalid request (1 bedroom, 4 bathrooms) -> `422`:

```json
{"error": "validation_error", "message": "Request body failed validation",
 "details": [{"field": "body", "message": "bathrooms (4) cannot exceed bedrooms + 1 (2)", "type": "value_error"}],
 "request_id": "02e69c87d9ea"}
```

### curl (PowerShell: use `curl.exe`, and send JSON from a file to avoid quoting problems)

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request_valid.json"
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request_missing_field.json"
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request_wrong_type.json"
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request_business_rule.json"
```

### Postman

File -> Import -> `examples/postman_collection.json`. It contains health, valid, missing field, wrong
type, business rule and model-info requests, each with a status-code test. The `baseUrl` variable
defaults to `http://127.0.0.1:8000`.

### Automated checks

```powershell
python -m pytest tests/test_api.py                                     # TestClient tests
python scripts/api_check.py --url http://127.0.0.1:8000 --label "local uvicorn"   # B5 table -> reports/api_validation.md
```

---

## 4. Part C: Docker

Needs Docker Desktop running and `models/model.joblib` present (step 1).

```powershell
docker build -t ml-inference-api:1.0 .
docker image ls ml-inference-api
docker run --name ml-api -p 8000:8000 ml-inference-api:1.0
```

In a second terminal:

```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@examples/request_valid.json"
docker ps
docker logs ml-api
docker stop ml-api
docker rm ml-api
```

Port 8000 busy? Use `-p 8001:8000` and browse http://localhost:8001/docs.

Run the whole validation (build, run, health, predict, invalid input, local vs container, logs,
stop/remove/recreate, clean up) and save the evidence:

```powershell
python scripts/docker_check.py              # -> reports/docker_validation.md, reports/evidence/docker_*.txt
python scripts/docker_check.py --no-cache   # same, with a full rebuild
```

Image details: `python:3.11-slim` base, dependency layer cached before code, model baked in (it is
small and verified by checksum), non-root user, Docker `HEALTHCHECK` on `/health`. `.dockerignore`
keeps `.git`, `.venv`, data, MLflow stores, tests and secrets out of the build context (about 560 KB).

Configuration via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/app/models/model.joblib` | Model file (metadata must be next to it) |
| `MLFLOW_TRACKING_URI` | unset | If set to an http URL, `/health` reports whether MLflow is reachable |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning` |

Using a model from the host instead of the baked-in one (volume):

```powershell
docker run --name ml-api -p 8000:8000 -v "${PWD}\models:/models:ro" -e MODEL_PATH=/models/model.joblib ml-inference-api:1.0
```

### Docker Compose: API + MLflow

```powershell
docker compose config                  # validate
docker compose up --build -d           # API http://localhost:8000/docs   MLflow http://localhost:5000
docker compose ps
docker compose logs
curl.exe http://localhost:8000/health  # "mlflow": {"uri": "http://mlflow:5000", "reachable": true}
```

Log a training run to the Compose MLflow server from the host:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m src.train
Remove-Item Env:MLFLOW_TRACKING_URI
```

Stop: `docker compose down` (the `mlflow-data` volume and all runs are kept) or
`docker compose down -v` (also deletes the volume).

Automated check of connectivity and persistence:

```powershell
python scripts/compose_check.py        # -> reports/compose_validation.md, reports/evidence/compose_*.txt
```

Two Compose details verified against MLflow 3.17.1: MLflow rejects unknown `Host` headers with 403, so
the service name `mlflow` is in `--allowed-hosts`; and the SQLite backend runs with `--workers=1`.

---

## 5. Part D: complete workflow (how a teammate reproduces it)

```powershell
git clone https://github.com/SakshiSalot/mlops-lab-7-9.git
cd mlops-lab-7-9
python -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
dvc pull                                   # or: python -m src.make_dataset; python -m src.train
python -m src.train                        # 1-3: data -> train -> evaluate -> save pipeline -> MLflow
uvicorn src.api:app --port 8000            # 4: start the API, then POST /predict (Swagger/curl/Postman)
docker build -t ml-inference-api:1.0 .     # 5: build the image (stop uvicorn first to free port 8000)
docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0     # 6: run it
python scripts/compare_predictions.py --url http://localhost:8000  # local vs container
```

The comparison loads `models/model.joblib` in the local Python process, sends the same 6 houses to
the container, and checks `|local - container| <= 1e-6` lakhs. Same artifact (same SHA-256) and the
same pinned library versions give a difference of exactly 0 in the reference run. Retraining on
Python 3.13 produced a byte-identical model, so results also match across Python versions. The
tolerance covers CPU differences (e.g. an `arm64` image built on Apple Silicon).

---

## About the committed evidence

`reports/` and `docs/screenshots/` hold a reference run recorded while building the project. Docker
Hub and ghcr.io were not reachable in that environment, so the Docker steps used locally built
stand-ins for `python:3.11-slim` and `ghcr.io/mlflow/mlflow:v3.17.1` through the Dockerfile's
`BASE_IMAGE` build argument (visible in `reports/evidence/docker_02_build.txt`). On a normal machine
the defaults are used unchanged. Rerun the scripts to regenerate every report on your own machine:

```powershell
python scripts/validate_pipeline.py
python scripts/api_check.py --url http://127.0.0.1:8000            # with uvicorn running
python scripts/docker_check.py
python scripts/compose_check.py
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `model file not found` / `/health` returns 503 | Run `dvc pull` or `python -m src.train`, then rebuild the image |
| `model checksum does not match` | `models/` is out of sync; `dvc checkout` or retrain |
| `trained with scikit-learn X but Y is installed` | `pip install -r requirements.txt` (versions are pinned) |
| `curl` in PowerShell shows `Invoke-WebRequest` errors | Use `curl.exe`, and pass JSON with `--data-binary "@file.json"` |
| Port already in use | Stop uvicorn or the other container, or map `-p 8001:8000` |
| `/health` shows MLflow `http_status: 403` | The MLflow host name is missing from `--allowed-hosts` |
| `dvc pull` cannot find files | `dvc-storage` must sit next to the repo folder (`..\dvc-storage`) |
