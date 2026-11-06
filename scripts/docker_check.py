"""C4 + C7 + Part D: build, run, test, inspect, recreate and clean up the API container.

    python scripts/docker_check.py                 # full run, writes reports/docker_validation.md
    python scripts/docker_check.py --port 8001     # if host port 8000 is busy

Every command and its output is also saved in reports/evidence/docker_*.txt, ready to
quote or screenshot. Needs Docker running and models/model.joblib present
(`dvc pull` or `python -m src.train`).
"""
from __future__ import annotations

import argparse
import json
import time

from _common import EVIDENCE, REPORTS, ROOT, http, load_json, md_table, now, run, wait_for
from compare_predictions import compare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="ml-inference-api:1.0")
    ap.add_argument("--name", default="ml-api")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--base-image", help="override the Dockerfile BASE_IMAGE build argument")
    ap.add_argument("--keep", action="store_true", help="leave the container running at the end")
    args = ap.parse_args()
    if not (ROOT / "models" / "model.joblib").exists():
        raise SystemExit("models/model.joblib not found: run `dvc pull` or `python -m src.train` first")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for f in EVIDENCE.glob("docker_*.txt"):
        f.unlink()
    url = f"http://localhost:{args.port}"
    steps = []

    def step(n, what, cmd, result):
        steps.append({"#": n, "Step": what, "Command": f"`{cmd}`", "Result": result})
        print(f"[{n}] {what}: {result}")

    v = run(["docker", "version", "--format", "client {{.Client.Version}} / server {{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})"],
            EVIDENCE / "docker_01_version.txt")
    step(1, "Docker available", "docker version", v.stdout.strip())

    build = ["docker", "build", "-t", args.image, "."]
    if args.base_image:
        build[2:2] = ["--build-arg", f"BASE_IMAGE={args.base_image}"]
    t0 = time.time()
    run(build, EVIDENCE / "docker_02_build.txt")
    step(2, "Build image (C7.14)", " ".join(build), f"success in {time.time() - t0:.0f}s")

    ls = run(["docker", "image", "ls", args.image, "--format", "{{.Repository}}:{{.Tag}}  id={{.ID}}  size={{.Size}}"],
             EVIDENCE / "docker_03_image_ls.txt")
    step(3, "Image details", f"docker image ls {args.image}", ls.stdout.strip())

    run(["docker", "rm", "-f", args.name], check=False)
    run_cmd = ["docker", "run", "-d", "--name", args.name, "-p", f"{args.port}:8000", args.image]
    cid = run(run_cmd, EVIDENCE / "docker_04_run.txt").stdout.strip()[:12]
    waited = wait_for(url + "/health")
    health_state = "unknown"
    for _ in range(40):                                   # Docker's own HEALTHCHECK (every 10 s)
        health_state = run(["docker", "inspect", "--format", "{{.State.Health.Status}}", args.name]).stdout.strip()
        if health_state == "healthy":
            break
        time.sleep(1)
    step(4, "Run container", " ".join(run_cmd),
         f"container {cid}; /health 200 after {waited:.1f}s; Docker HEALTHCHECK status: {health_state}")

    _, health = http("GET", url + "/health")
    (EVIDENCE / "docker_05_health.json").write_text(json.dumps(health, indent=2) + "\n", encoding="utf-8")
    step(5, "Health from host (C7.15)", f"GET {url}/health", f"status={health['status']}, model {health['model_version']}")

    status, pred = http("POST", url + "/predict", body=load_json("request_valid.json"))
    (EVIDENCE / "docker_06_predict.json").write_text(json.dumps(pred, indent=2) + "\n", encoding="utf-8")
    first_prediction = pred["prediction"]
    step(6, "Prediction from host (C7.16)", f"POST {url}/predict (examples/request_valid.json)",
         f"{status}: {first_prediction:.6f} lakhs")

    status, bad = http("POST", url + "/predict", body=load_json("request_missing_field.json"))
    step(7, "Invalid input handled in container", f"POST {url}/predict (missing bedrooms)",
         f"{status}: {bad['details'][0]['field']} - {bad['details'][0]['message']}")

    result = compare(url)
    step(8, "Local vs container (Part D)", "scripts/compare_predictions.py",
         f"{len(result['rows'])} inputs, max abs diff {result['max_diff']:.2e}, same artifact: "
         f"{result['local_sha'] == result['remote_sha']}")

    ps = run(["docker", "ps", "--filter", f"name={args.name}", "--format", "{{.Names}}  {{.Image}}  {{.Status}}  {{.Ports}}"],
             EVIDENCE / "docker_09_ps.txt")
    step(9, "Running containers", "docker ps", ps.stdout.strip())

    time.sleep(1)
    logs = run(["docker", "logs", args.name], EVIDENCE / "docker_10_logs.txt")
    text = logs.stdout + logs.stderr
    errors = [ln for ln in text.splitlines() if " ERROR " in ln or "Traceback" in ln]
    loaded = next((ln for ln in text.splitlines() if "model loaded" in ln), "model loaded line not found")
    step(10, "Inspect logs (C7.17)", f"docker logs {args.name}",
         f"startup: '{loaded.split(': ', 1)[-1]}'; error lines: {len(errors)}")

    run(["docker", "stop", args.name], EVIDENCE / "docker_11_stop_rm.txt")
    run(["docker", "rm", args.name], EVIDENCE / "docker_11_stop_rm.txt")
    step(11, "Stop and remove", f"docker stop {args.name} && docker rm {args.name}", "removed")

    run(run_cmd, EVIDENCE / "docker_12_recreate.txt")
    waited = wait_for(url + "/health")
    _, again = http("POST", url + "/predict", body=load_json("request_valid.json"))
    same = abs(again["prediction"] - first_prediction) <= 1e-12
    step(12, "Recreate and re-test (C7.18)", " ".join(run_cmd),
         f"healthy after {waited:.1f}s, prediction {again['prediction']:.6f} (identical: {same})")

    if not args.keep:
        run(["docker", "stop", args.name], EVIDENCE / "docker_13_cleanup.txt")
        run(["docker", "rm", args.name], EVIDENCE / "docker_13_cleanup.txt")
        step(13, "Clean up", f"docker stop {args.name} && docker rm {args.name}", "container removed (image kept)")

    report = "\n".join([
        "# Docker validation (C4, C7, Part D)", "",
        f"Run {now()} with `python scripts/docker_check.py`. Raw command output: `reports/evidence/docker_*.txt`.", "",
        md_table(steps, ["#", "Step", "Command", "Result"]),
    ])
    (REPORTS / "docker_validation.md").write_text(report + "\n", encoding="utf-8")
    print("wrote reports/docker_validation.md")
    if result["max_diff"] > 1e-6 or not same or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
