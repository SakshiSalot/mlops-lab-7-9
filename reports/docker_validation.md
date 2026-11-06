# Docker validation (C4, C7, Part D)

Run 2026-11-06 10:43:03 UTC with `python scripts/docker_check.py`. Raw command output: `reports/evidence/docker_*.txt`.

| # | Step | Command | Result |
|---|---|---|---|
| 1 | Docker available | `docker version` | client 29.8.2 / server 29.8.2 (linux/amd64) |
| 2 | Build image (C7.14) | `docker build --no-cache --build-arg BASE_IMAGE=local/python:3.11-slim-standin -t ml-inference-api:1.0 .` | success in 37s |
| 3 | Image details | `docker image ls ml-inference-api:1.0` | ml-inference-api:1.0  id=c90660e01d37  size=750MB |
| 4 | Run container | `docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0` | container 5fb17405b164; /health 200 after 2.0s; Docker HEALTHCHECK status: healthy |
| 5 | Health from host (C7.15) | `GET http://localhost:8000/health` | status=ok, model 1.0.0+b09c57dd3b34 |
| 6 | Prediction from host (C7.16) | `POST http://localhost:8000/predict (examples/request_valid.json)` | 200: 104.415616 lakhs |
| 7 | Invalid input handled in container | `POST http://localhost:8000/predict (missing bedrooms)` | 422: bedrooms - Field required |
| 8 | Local vs container (Part D) | `scripts/compare_predictions.py` | 6 inputs, max abs diff 0.00e+00, same artifact: True |
| 9 | Running containers | `docker ps` | ml-api  ml-inference-api:1.0  Up 7 seconds (healthy)  0.0.0.0:8000->8000/tcp |
| 10 | Inspect logs (C7.17) | `docker logs ml-api` | startup: 'model loaded: version=1.0.0+b09c57dd3b34 path=/app/models/model.joblib'; error lines: 0 |
| 11 | Stop and remove | `docker stop ml-api && docker rm ml-api` | removed |
| 12 | Recreate and re-test (C7.18) | `docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0` | healthy after 2.0s, prediction 104.415616 (identical: True) |
| 13 | Clean up | `docker stop ml-api && docker rm ml-api` | container removed (image kept) |

