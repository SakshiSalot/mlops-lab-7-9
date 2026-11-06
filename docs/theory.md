# Written answers: Lab 7-9 (ML Pipelines, FastAPI, Docker)

Every number quoted here comes from a file in `reports/` produced by the code in this repository.
Versions: Python 3.11 (container) / 3.13 (also tested), scikit-learn 1.9.1, FastAPI 0.143.3,
MLflow 3.17.1, DVC 3.67.1, Docker Engine 29.8.

---

## A1. The pipeline stages

| Stage | Module / function | Input | Output | Purpose | One possible failure (and the guard here) |
|---|---|---|---|---|---|
| Data loading | `src/data.py: load_data()` | `data/raw/houses_raw.csv` (DVC) | DataFrame of text columns | Read the source and check the contract | File missing or a column renamed upstream. Guard: clear error naming the missing columns; DVC pins the exact file (md5 `71a9a5cf`). |
| Cleaning | `normalize_formats()`, `preprocess_data()` | Raw DataFrame | Clean DataFrame + `reports/data_quality.json` | Fix formats (`"2,450"`, `"980 sqft"`, `Y/no/1`, `" north "`), drop duplicates and invalid targets, turn out-of-range values into NaN | A plausible typo passes every range check: house 404 has area 6740 instead of 674 and is the largest test error (+62 lakhs). Range checks cannot catch values that look valid. |
| Feature engineering | `src/features.py: HouseFeatureEngineer` + `build_preprocessor()` | Clean feature columns | Model-ready matrix (house_age, rooms_total, area_per_room, imputed, scaled, one-hot) | Turn raw columns into features the same way in training and serving | Using "today" for house age would make predictions drift every year. Guard: fixed `REFERENCE_YEAR = 2026`. Unknown locality would crash a strict encoder. Guard: `handle_unknown="ignore"`. |
| Training | `src/train.py: train_model()` | Train split + config | Fitted `Pipeline` -> `models/model.joblib` + metadata | Fit preprocessing and estimator on training data only | Leakage if the split happens after fitting the imputer/scaler. Guard: split first, everything fitted inside the Pipeline on `X_train`; a test asserts imputer medians equal train-only medians. |
| Evaluation | `src/evaluate.py: evaluate_model()` | Fitted pipeline + held-out test split | `reports/metrics.json`, `reports/evaluation.md`, residual plot | Estimate performance on unseen data, compare with a baseline | Reporting training-set metrics, or tuning on the test set. Guard: CV only on the train split; test split used once; `python -m src.evaluate` must reproduce the recorded numbers. |
| Prediction | `src/predict.py: load_model(), predict()` | Saved pipeline + new records | Prices (INR lakhs) | Score new data in another process, the API or batch | Loading a tampered or incompatible pickle. Guard: SHA-256 check against `model_metadata.json`, refuse a different scikit-learn minor version. |

**Why separate responsibilities.** Each stage can be tested on its own (29 tests), rerun on its own
(`python -m src.data`, `src.train`, `src.evaluate`, `src.predict`) and replaced without touching the
others (switching to `random_forest` is one config change). It also makes the *boundary* between
training-only work and inference work explicit. Row filtering (dropping duplicates) belongs to
training data only. Everything the model needs at prediction time (feature engineering, imputation,
encoding) lives inside the saved Pipeline, so the API cannot apply different preprocessing from
training. Finally, importing a module has no side effects (checked by a test), so the API can import
`src.predict` without accidentally retraining.

**Imputation choice.** With 5-fold CV on the training split only: median imputation 7.76 +/- 1.13 RMSE,
median + missing indicators 7.80 +/- 1.17, KNN (k=5) 7.59 +/- 1.37, iterative 7.56 +/- 1.50. The
differences are smaller than one standard deviation, so the simplest option (median) was kept.

---

## B1. REST and inference APIs

| Concept | Meaning | In this API |
|---|---|---|
| REST API | Operations exposed as resources over HTTP, stateless request/response | `/health`, `/model-info`, `/predict` |
| GET | Read something, no side effects, safe to repeat | `GET /health`, `GET /model-info` |
| POST | Submit data for processing | `POST /predict` with a JSON body |
| PUT / DELETE | Replace / remove a resource | Not needed; would fit e.g. `DELETE /batch-jobs/{id}` in the batch design below |
| Request schema | Required fields, types and constraints the server accepts | `HouseFeatures` (Pydantic, strict types, ranges, allowed localities, bathrooms <= bedrooms + 1, unknown fields rejected) |
| Response schema | What the client gets back | `PredictionResponse`: `prediction`, `unit`, `model_version`, `request_id`; errors use `ErrorResponse` |

| Aspect | Online inference (`POST /predict`) | Batch inference (`python -m src.predict --input file.csv`) |
|---|---|---|
| Input | One house per request | A CSV of many houses, raw formats allowed |
| Response | Immediate: 13-31 ms per request in the container log (422s in about 1 ms) | A file after the whole batch is processed |
| Typical use | A pricing form in a web app | Re-scoring all listings every night |
| Main concern | Latency, availability, input validation | Throughput, scheduling, partial failures |
| Invalid rows | 422 for the request | Row marked `rejected` with a reason; other rows still scored |

---

## B6. Online versus batch design for a CSV of many records

| Option | When it fits | Problems |
|---|---|---|
| Synchronous endpoint (`POST /predict-batch` returns all predictions) | Small files (up to a few hundred rows) | The request blocks a worker for the whole file; large files hit client/proxy timeouts and memory limits; one retry rescores everything |
| Background job in the API (FastAPI `BackgroundTasks`) | Medium files, single instance | Jobs are lost if the container restarts; status lives in one process's memory |
| Separate batch process (scheduled job, queue worker) | Large or periodic scoring | More moving parts (storage, scheduler), but isolated from the online API's latency |

**Decision.** A separate batch process, which is implemented: `python -m src.predict --input <csv>
--output <csv>` loads the same verified pipeline, repairs formats with the same `normalize_formats()`,
validates each row with the same `schema.py` rules, scores the valid rows and writes per-row status and
error messages. It can run from cron, a CI job or `docker run ... python -m src.predict` without
touching the online service.

If clients must upload through HTTP, the proposed (not implemented) design is asynchronous:
`POST /batch-jobs` stores the file (object storage) and returns `202 Accepted` with a job id; a worker
container consumes a queue, scores in chunks and writes results; `GET /batch-jobs/{id}` returns status
and a download link. This keeps `/predict` fast, survives restarts and allows retries per chunk.

---

## C1. Docker concepts

| Concept | Meaning | In this project |
|---|---|---|
| Image | Read-only template with OS userland, Python, packages, code and config | `ml-inference-api:1.0` (python:3.11-slim + requirements.txt + `src/` + `models/`) |
| Container | A running instance of an image, with its own writable layer | `docker run --name ml-api -p 8000:8000 ml-inference-api:1.0` |
| Dockerfile | Ordered build instructions; each creates a cached layer | Requirements copied and installed before the code, so code edits rebuild in seconds |
| Port mapping | Connects a host port to a container port | `-p 8000:8000`, or `-p 8001:8000` if 8000 is busy |
| Volume | Storage that outlives the container's writable layer | `mlflow-data` keeps the MLflow database and artifacts across `docker compose down` |
| Environment variable | Runtime configuration without rebuilding | `MODEL_PATH`, `MLFLOW_TRACKING_URI`, `LOG_LEVEL` |
| Docker Compose | Defines related services, a shared network and volumes in one file | `api` + `mlflow`; the API reaches MLflow at `http://mlflow:5000` |

**Why software works on one laptop and fails on another.** Different Python versions (here: numpy
2.5 and scipy 1.18 require Python 3.12, so the "latest" versions would not even install on the
`python:3.11-slim` base), different library versions (a pickle made with one scikit-learn minor version
may not load in another, which is why the loader refuses a mismatch), different operating systems (path
separators, CRLF line endings changing file hashes, case sensitivity), missing system libraries, files
that exist only on one machine (a model that was never committed; a host path that is not visible
inside a container), different environment variables, and ports already in use.

**How containers reduce this.** The image fixes the OS userland, the Python version, every package
version, the code and the model file. Everyone who runs `ml-inference-api:1.0` runs the same bits. The
reference run shows this: predictions from the container and from a local Python process are
identical (max difference 0.0 on 6 inputs).

**What containers do not guarantee.**
- Same CPU architecture: an image built on an Apple Silicon Mac is `arm64`, a Windows PC builds `amd64`;
  float results can differ in the last digits (hence the documented 1e-6 tolerance).
- Same kernel or hardware: containers share the host kernel; GPU, memory and CPU limits differ.
- Reproducible builds over time: `python:3.11-slim` is a moving tag and unpinned transitive packages
  can change; pin a digest and use a lock file for strict reproducibility.
- External dependencies: MLflow, databases and networks must still be reachable and configured.
- Persistence: anything written inside a container is lost when it is removed, unless it is on a volume.
- Correctness or security: a container runs wrong code or a vulnerable package just as faithfully.

---

## Challenges and resolutions

| Challenge | What happened | Resolution |
|---|---|---|
| Latest numpy/scipy drop Python 3.11 | `numpy 2.5.4` and `scipy 1.18.1` require Python >= 3.12; pinning them would break `pip install` in `python:3.11-slim` | Pinned `numpy 2.4.6` and `scipy 1.17.1`, the newest versions with Python 3.11 wheels for Linux and Windows; verified on 3.11 and 3.13 |
| MLflow rejects the Compose service name | API health showed `reachable: false, http_status: 403`; MLflow log: `Rejected request with invalid Host header: mlflow:5000` (MLflow 3.x DNS-rebinding protection) | Added `--allowed-hosts=mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*`; evidence in `reports/evidence/challenge_mlflow_allowed_hosts.txt` |
| MLflow `/health` hides that problem | `/health` skips the Host check, so it returned 200 even when real API calls got 403 | The API's connectivity check calls `/api/2.0/mlflow/experiments/search` instead |
| Pickle safety | `joblib.load` can execute code from a tampered file | Model loaded only if its SHA-256 matches `model_metadata.json`; scikit-learn minor version must match; tested with a tampered file |
| Training/serving skew | Cleaning code duplicated in the API could drift from training | Feature engineering, imputation and encoding live inside the saved Pipeline; ranges and the business rule live in one module (`src/schema.py`) used by both |
| Plausible typos | House 404: area 6740 instead of 674 is inside the valid range, so it passed validation and produced the largest test error | Documented as a limitation; RMSE without that row is 7.97 vs 8.92; a cross-field plausibility rule or outlier detection would be the next step |
| Starlette TestClient deprecation | Starlette 1.x warns that plain `httpx` is deprecated for `TestClient` | Dev requirements use `httpx2` |
| MLflow on SQLite with several workers | The server defaults to 4 workers; SQLite allows one writer | `--workers=1` in Compose |
| Registries blocked in the build environment | Docker Hub and ghcr.io were not reachable where the reference run was recorded | Dockerfile takes `BASE_IMAGE` as a build argument; the reference run used locally built stand-ins for `python:3.11-slim` and `ghcr.io/mlflow/mlflow:v3.17.1`; on a normal machine the defaults are used unchanged |

---

## Reflection

The most useful change was treating the model as a *pipeline artifact* rather than an estimator. Once
feature engineering, imputation and encoding are fitted and saved together, the API code becomes small:
validate the request, build a one-row DataFrame, call `predict`. Training and serving cannot disagree,
and the local-vs-container comparison confirms it with a difference of exactly zero.

Validation turned out to be layered. Pydantic catches structure and types (missing fields, strings for
numbers, unknown localities), the business rule catches impossible combinations, and the cleaning
report shows what the raw data contained. But the largest error came from a value that passed every
rule, which shows that schema validation is necessary but not sufficient; monitoring and plausibility
checks are still needed.

Docker removed most "works on my machine" problems, but only after the dependency list was made exact:
choosing the base image first and then pinning versions that support it, splitting runtime and
development requirements (the API image needs no MLflow, DVC or matplotlib), and baking a verified
model into the image. Compose added a new class of problem: networking and security between services.
The MLflow Host-header rejection would have been easy to miss because MLflow's own health check still
passed; testing a real API call from inside the API container exposed it.

Finally, writing the checks as scripts (`validate_pipeline.py`, `api_check.py`, `docker_check.py`,
`compose_check.py`) made the evidence reproducible: a teammate can regenerate every table in this report
on their own machine with four commands.

## References

- FastAPI documentation: request bodies, Pydantic models, error handling, testing with TestClient, https://fastapi.tiangolo.com/
- Pydantic v2 documentation: strict mode, validators, https://docs.pydantic.dev/
- scikit-learn documentation: Pipeline, ColumnTransformer, SimpleImputer, model persistence, https://scikit-learn.org/stable/
- Docker documentation: Dockerfile reference, HEALTHCHECK, .dockerignore, Compose file reference, https://docs.docker.com/
- MLflow 3.17 documentation: `mlflow server` options (`--allowed-hosts`, `--artifacts-destination`), official image `ghcr.io/mlflow/mlflow`, https://mlflow.org/docs/latest/
- DVC documentation: `dvc add`, pipelines (`dvc.yaml`), remotes, https://dvc.org/doc
