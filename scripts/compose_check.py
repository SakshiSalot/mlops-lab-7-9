"""C6 + C7.19: run API + MLflow with Docker Compose and verify connectivity and persistence.

    python scripts/compose_check.py

Steps: validate the Compose file -> up --build -> API health shows MLflow reachable
(by service name, from inside the API container) -> log a run from the host ->
`docker compose down` (volume kept) -> up again -> the run and its artifact are still
there -> save logs -> down. Writes reports/compose_validation.md and
reports/evidence/compose_*.txt. Needs Docker and the dev requirements (mlflow) on the host.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from _common import EVIDENCE, REPORTS, ROOT, http, md_table, now, run, wait_for

EXPERIMENT = "compose-persistence-check"


def log_check_run(tracking_uri: str) -> str:
    """Log a tiny run (param, metric, artifact) to the Compose MLflow server from the host."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="host-to-compose") as r:
        mlflow.log_param("source", "scripts/compose_check.py")
        mlflow.log_metric("check_value", 1.0)
        mlflow.log_artifact(str(ROOT / "models" / "model_metadata.json"), artifact_path="check")
        return r.info.run_id


def run_exists(tracking_uri: str, run_id: str) -> tuple[bool, bool]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        r = client.get_run(run_id)
    except Exception:  # noqa: BLE001 - run not found
        return False, False
    with tempfile.TemporaryDirectory() as d:
        path = client.download_artifacts(run_id, "check/model_metadata.json", d)
        artifact_ok = json.load(open(path, encoding="utf-8")).get("model_sha256") is not None
    return r.data.metrics.get("check_value") == 1.0, artifact_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--override", action="append", default=[], help="extra compose file(s), e.g. a local override")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--mlflow-port", type=int, default=5000)
    args = ap.parse_args()

    compose = ["docker", "compose", "-f", "docker-compose.yml"]
    for f in args.override:
        compose += ["-f", f]
    api, mlf = f"http://localhost:{args.api_port}", f"http://localhost:{args.mlflow_port}"
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for f in EVIDENCE.glob("compose_*.txt"):
        f.unlink()
    steps = []

    def step(n, what, cmd, result):
        steps.append({"#": n, "Step": what, "Command": f"`{cmd}`", "Result": result})
        print(f"[{n}] {what}: {result}")

    run(compose + ["config", "--quiet"], EVIDENCE / "compose_01_config.txt")
    run(compose + ["config"], EVIDENCE / "compose_01_config.txt")
    step(1, "Validate Compose file", "docker compose config", "valid")

    run(compose + ["up", "--build", "-d"], EVIDENCE / "compose_02_up.txt")
    t_api = wait_for(api + "/health", timeout=180)
    wait_for(mlf + "/health", timeout=180)
    step(2, "Start both services", "docker compose up --build -d", f"API healthy after {t_api:.0f}s")

    ps = run(compose + ["ps", "--format", "{{.Service}}  {{.Status}}  {{.Ports}}"], EVIDENCE / "compose_03_ps.txt")
    step(3, "Service status", "docker compose ps", ps.stdout.strip().replace("\n", " ; "))

    _, health = http("GET", api + "/health")
    m = health["mlflow"]
    step(4, "API -> MLflow by service name (C7.19)", f"GET {api}/health",
         f"model_loaded={health['model_loaded']}, mlflow uri={m.get('uri')}, reachable={m.get('reachable')}")

    probe = ("import urllib.request as u; r = u.urlopen('http://mlflow:5000/api/2.0/mlflow/experiments/search"
             "?max_results=1', timeout=5); print('HTTP', r.status)")
    out = run(compose + ["exec", "-T", "api", "python", "-c", probe], EVIDENCE / "compose_05_network.txt")
    step(5, "Direct call inside the api container", "docker compose exec api python -c ...", out.stdout.strip())

    run_id = log_check_run(mlf)
    step(6, "Log a run from the host", f"mlflow.set_tracking_uri('{mlf}')", f"run {run_id} (param, metric, artifact)")

    run(compose + ["down"], EVIDENCE / "compose_07_down_up.txt")
    run(compose + ["up", "-d"], EVIDENCE / "compose_07_down_up.txt")
    wait_for(api + "/health", timeout=180)
    wait_for(mlf + "/health", timeout=180)
    step(7, "Remove and recreate containers", "docker compose down && docker compose up -d", "both healthy again")

    metric_ok, artifact_ok = run_exists(mlf, run_id)
    step(8, "Persistence check", f"MlflowClient().get_run('{run_id[:8]}...')",
         f"run still present with its metric: {metric_ok}; artifact downloadable: {artifact_ok}")

    logs = run(compose + ["logs", "--no-color", "--tail", "40"], EVIDENCE / "compose_09_logs.txt")
    errors = [ln for ln in (logs.stdout + logs.stderr).splitlines() if " ERROR " in ln or "Traceback" in ln]
    step(9, "Inspect logs", "docker compose logs", f"saved; error lines: {len(errors)}")

    run(compose + ["down"], EVIDENCE / "compose_10_down.txt")
    step(10, "Stop", "docker compose down", "containers removed; volume mlflow-data kept (`down -v` deletes it)")

    (REPORTS / "compose_validation.md").write_text("\n".join([
        "# Docker Compose validation (C6, C7.19)", "",
        f"Run {now()} with `python scripts/compose_check.py`. Raw output: `reports/evidence/compose_*.txt`.", "",
        md_table(steps, ["#", "Step", "Command", "Result"]),
    ]) + "\n", encoding="utf-8")
    print("wrote reports/compose_validation.md")
    if not (m.get("reachable") and metric_ok and artifact_ok) or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
