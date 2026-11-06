# Docker Compose validation (C6, C7.19)

Run 2026-11-06 10:43:45 UTC with `python scripts/compose_check.py`. Raw output: `reports/evidence/compose_*.txt`.

| # | Step | Command | Result |
|---|---|---|---|
| 1 | Validate Compose file | `docker compose config` | valid |
| 2 | Start both services | `docker compose up --build -d` | API healthy after 2s |
| 3 | Service status | `docker compose ps` | api  Up 2 seconds (health: starting)  0.0.0.0:8000->8000/tcp ; mlflow  Up 13 seconds (healthy)  0.0.0.0:5000->5000/tcp |
| 4 | API -> MLflow by service name (C7.19) | `GET http://localhost:8000/health` | model_loaded=True, mlflow uri=http://mlflow:5000, reachable=True |
| 5 | Direct call inside the api container | `docker compose exec api python -c ...` | HTTP 200 |
| 6 | Log a run from the host | `mlflow.set_tracking_uri('http://localhost:5000')` | run 6a53f2a9b9314d96843fa3f1b6847f45 (param, metric, artifact) |
| 7 | Remove and recreate containers | `docker compose down && docker compose up -d` | both healthy again |
| 8 | Persistence check | `MlflowClient().get_run('6a53f2a9...')` | run still present with its metric: True; artifact downloadable: True |
| 9 | Inspect logs | `docker compose logs` | saved; error lines: 0 |
| 10 | Stop | `docker compose down` | containers removed; volume mlflow-data kept (`down -v` deletes it) |

